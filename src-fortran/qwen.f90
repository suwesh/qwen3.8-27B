! Qwen3.8-27B (MLX 8-bit) inference in Fortran: the program of src-cuda/main.c. LFortran runs the
! kernels (kernels.f90) on the GPU with --gpu=cuda.
!
!   ./qwen -m /tmp/certik/model -p "Why is the sky blue?" [-n 512] [--no-think]
!
! Pipeline: tokenizer -> chat template -> one token at a time through the model on the GPU (the prompt
! too) -> greedy decoding -> detokenize and print as we go.
module model
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real64, real32
    use model_io_support, only: io_error
    use safetensors_io, only: tensor_archive, open_archive, find_tensor, read_u32, read_bf16, read_f32
    use qwen_config, only: dense_config, read_config
    use kernels
    implicit none
    private
    public :: weights, state, load_model, alloc_state, step

    type :: layer
        logical :: linear
        real, allocatable :: in_norm(:), post_norm(:)
        type(qmat) :: in_proj    ! linear: [qkv | z | b | a]; full: [q+gate | k | v]
        type(qmat) :: out_proj   ! linear_attn.out_proj or self_attn.o_proj
        type(qmat) :: gate_up    ! [gate; up], rows interleaved in pairs (see gemv_norm)
        type(qmat) :: down
        real, allocatable :: conv_w(:,:), a_log(:), dt_bias(:), norm_w(:)   ! linear attention
        real, allocatable :: q_norm(:), k_norm(:)                           ! full attention
    end type

    ! The layers are fixed-size arrays, not allocatable: the host reads their descriptors at every launch,
    ! and an allocatable array of them (more than 4 KB) would be managed memory, whose pages would move
    ! between the host and the GPU with the small arrays next to them.
    integer, parameter :: max_layers = 64

    ! The parts of the argmax, which argmax_final combines on the GPU into the next token.
    integer, parameter :: np = 512

    type :: weights
        type(dense_config) :: c
        type(layer) :: l(max_layers)
        type(qmat) :: embed, lm_head
        real, allocatable :: final_norm(:)
    end type

    type :: layer_state
        real, allocatable :: conv(:,:), ssm(:,:,:)   ! linear attention: conv inputs, recurrent state
        real, allocatable :: kc(:,:,:), vc(:,:,:)    ! full attention: KV cache, kc(d, kh, t), vc(d, kh, t)
    end type

    type :: state
        real, allocatable :: x(:), xw(:)     ! residual stream, x * next norm weight
        integer(int64), allocatable :: ss(:) ! sum x**2 in fixed point (scale 2**24), in slot k of two (padded)
        integer :: k = 1
        real, allocatable :: y(:), o(:), mid(:), logits(:)
        real, allocatable :: pm(:,:,:), pl(:,:,:), pa(:,:,:,:), rc(:,:), rs(:,:)   ! attention: chunks, RoPE cos/sin
        real, allocatable :: pv(:)                   ! argmax parts (see np)
        integer, allocatable :: pi(:), tokbuf(:), hist(:)
        type(layer_state) :: l(max_layers)
    end type

    type(tensor_archive) :: archive

contains

    subroutine check(e)
        type(io_error), intent(in) :: e
        if (e%failed()) error stop e%message
    end subroutine

    integer function tensor(name)
        character(*), intent(in) :: name
        tensor = find_tensor(archive, name)
        if (tensor == 0) error stop 'tensor not found: '//name
    end function

    ! The size of a small array the kernels read: at least 4 KB, which LFortran's CUDA runtime puts in
    ! managed memory, used in place. A smaller array is copied from pageable host memory at every
    ! launch, which makes the host wait for the GPU.
    integer function pad(n)
        integer, intent(in) :: n
        pad = max(n, 1024)
    end function

    ! A bf16 vector, as reals (padded, see pad)
    subroutine vec(name, n, v)
        character(*), intent(in) :: name
        integer, intent(in) :: n
        real, allocatable, intent(out) :: v(:)
        integer(int16) :: h(n)
        type(io_error) :: e
        allocate(v(pad(n)))
        call read_bf16(archive%tensor(tensor(name)), h, e)
        call check(e)
        v = 0
        v(1:n) = bf(h)
    end subroutine

    subroutine vec_f32(name, n, v)
        character(*), intent(in) :: name
        integer, intent(in) :: n
        real(real32), allocatable, intent(out) :: v(:)
        real(real32), allocatable :: x(:)
        type(io_error) :: e
        allocate(x(n))
        allocate(v(int(pad(n), int32)))
        call read_f32(archive%tensor(tensor(name)), x, e)
        call check(e)

        v = 0.0
        v(1:n) = real(x, kind=kind(v))
        deallocate(x)
    end subroutine

    ! The q8 matrices `names` (same K), stacked along N into one matrix, so one GEMV computes them all.
    subroutine load_q8(names, k, m)
        character(*), intent(in) :: names(:)
        integer, intent(in) :: k
        type(qmat), intent(out) :: m
        type(io_error) :: e
        integer :: i, n, rows(size(names))
        do i = 1, size(names)
            rows(i) = int(archive%tensor(tensor(trim(names(i))//'.weight'))%shape(1))
        end do
        allocate(m%w(k/4, sum(rows)), m%s(k/64, sum(rows)), m%b(k/64, sum(rows)))
        n = 0
        do i = 1, size(names)
            call read_u32(archive%tensor(tensor(trim(names(i))//'.weight')), m%w(:, n+1:n+rows(i)), e)
            call check(e)
            call read_bf16(archive%tensor(tensor(trim(names(i))//'.scales')), m%s(:, n+1:n+rows(i)), e)
            call check(e)
            call read_bf16(archive%tensor(tensor(trim(names(i))//'.biases')), m%b(:, n+1:n+rows(i)), e)
            call check(e)
            n = n + rows(i)
        end do
    end subroutine

    ! [gate; up] with rows interleaved in pairs: g1 g2 u1 u2 g3 g4 u3 u4 ..., so that one GEMV
    ! iteration (4 rows) has both inputs of silu(gate_i) * up_i for two outputs i.
    subroutine load_q8_swiglu(gate, up, k, m)
        character(*), intent(in) :: gate, up
        integer, intent(in) :: k
        type(qmat), intent(out) :: m
        type(qmat) :: g, u
        integer :: j, n
        call load_q8([gate], k, g)
        call load_q8([up], k, u)
        n = size(g%w, 2)
        allocate(m%w(k/4, 2*n), m%s(k/64, 2*n), m%b(k/64, 2*n))
        do j = 1, n/2
            m%w(:, 4*j-3:4*j-2) = g%w(:, 2*j-1:2*j)
            m%w(:, 4*j-1:4*j) = u%w(:, 2*j-1:2*j)
            m%s(:, 4*j-3:4*j-2) = g%s(:, 2*j-1:2*j)
            m%s(:, 4*j-1:4*j) = u%s(:, 2*j-1:2*j)
            m%b(:, 4*j-3:4*j-2) = g%b(:, 2*j-1:2*j)
            m%b(:, 4*j-1:4*j) = u%b(:, 2*j-1:2*j)
        end do
    end subroutine

    subroutine load_model(dir, w)
        character(*), intent(in) :: dir
        type(weights), intent(out) :: w
        character(*), parameter :: P = 'language_model.model'
        character(:), allocatable :: a
        character(16) :: num
        type(io_error) :: e
        integer(int16), allocatable :: cw(:)
        integer :: l, h
        call read_config(dir//'/config.json', w%c, e)
        call check(e)
        call open_archive(dir, archive, e)
        call check(e)
        associate (c => w%c)
            if (c%head_dim /= 256 .or. c%linear_key_head_dim /= 128 .or. c%linear_value_head_dim /= 128 &
                .or. c%conv_kernel /= 4 .or. c%rotary_dim /= 64 .or. c%group_size /= 64 .or. c%quant_bits /= 8) &
                error stop 'unsupported model dimensions'
            h = c%hidden
            call load_q8([P//'.embed_tokens'], h, w%embed)
            if (find_tensor(archive, 'language_model.lm_head.weight') /= 0) then
                call load_q8(['language_model.lm_head'], h, w%lm_head)
            else
                call load_q8([P//'.embed_tokens'], h, w%lm_head)
            end if
            ! write(0, '(a,2(1x,i0))') 'embed w shape:', shape(w%embed%w)
            ! write(0, '(a,2(1x,i0))') 'lm_head w shape:', shape(w%lm_head%w)
            ! write(0, '(a,2(1x,i0))') 'embed s shape:', shape(w%embed%s)
            ! write(0, '(a,2(1x,i0))') 'lm_head s shape:', shape(w%lm_head%s)
            call vec(P//'.norm.weight', h, w%final_norm)
            if (c%layers > max_layers) error stop 'too many layers'
            do l = 1, c%layers
                write(num, '(i0)') l - 1
                a = P//'.layers.'//trim(num)
                write(0, '(a,i0,a,i0)', advance='no') achar(13)//'loading layer ', l, '/', c%layers
                associate (y => w%l(l))
                    y%linear = .not. c%full_attention(l)
                    call vec(a//'.input_layernorm.weight', h, y%in_norm)
                    call vec(a//'.post_attention_layernorm.weight', h, y%post_norm)
                    if (y%linear) then
                        call load_q8([character(80) :: a//'.linear_attn.in_proj_qkv', a//'.linear_attn.in_proj_z', &
                            a//'.linear_attn.in_proj_b', a//'.linear_attn.in_proj_a'], h, y%in_proj)
                        call load_q8([a//'.linear_attn.out_proj'], c%linear_value_dim, y%out_proj)
                        allocate(cw(c%conv_kernel*c%conv_dim))
                        call read_bf16(archive%tensor(tensor(a//'.linear_attn.conv1d.weight')), cw, e)
                        call check(e)
                        allocate(y%conv_w(c%conv_kernel, c%conv_dim))
                        y%conv_w = reshape(bf(cw), [c%conv_kernel, c%conv_dim])   ! oldest input first
                        deallocate(cw)
                        call vec_f32(a//'.linear_attn.A_log', c%linear_value_heads, y%a_log)
                        ! if (l == 1) then
                        !     write(0, '(a,8(1x,es15.7))') &
                        !    'Fortran A_log first 8:', y%a_log(1:8)
                        ! end if
                        call vec(a//'.linear_attn.dt_bias', c%linear_value_heads, y%dt_bias)
                        call vec(a//'.linear_attn.norm.weight', c%linear_value_head_dim, y%norm_w)
                    else
                        call load_q8([character(80) :: a//'.self_attn.q_proj', a//'.self_attn.k_proj', &
                            a//'.self_attn.v_proj'], h, y%in_proj)
                        call load_q8([a//'.self_attn.o_proj'], c%query_dim, y%out_proj)
                        call vec(a//'.self_attn.q_norm.weight', c%head_dim, y%q_norm)
                        call vec(a//'.self_attn.k_norm.weight', c%head_dim, y%k_norm)
                    end if
                    call load_q8_swiglu(a//'.mlp.gate_proj', a//'.mlp.up_proj', h, y%gate_up)
                    call load_q8([a//'.mlp.down_proj'], c%intermediate, y%down)
                end associate
            end do
        end associate
        write(0, '()')
    end subroutine

    subroutine alloc_state(w, s, max_ctx)
        type(weights), intent(in) :: w
        type(state), intent(out) :: s
        integer, intent(in) :: max_ctx
        integer :: l, j, t
        if (max_ctx > AMAX) error stop 'the context is at most AMAX positions (kernels.f90)'
        associate (c => w%c)
            allocate(s%x(c%hidden), s%xw(c%hidden), s%ss(pad(2)), s%mid(c%intermediate), s%logits(c%vocab), &
                s%y(max(2*c%intermediate, c%conv_dim + c%linear_value_dim + 2*c%linear_value_heads, &
                        2*c%query_dim + 2*c%kv_dim)), &
                s%o(max(c%linear_value_dim, c%query_dim)), &
                s%pm(AG, c%kv_heads, max_ctx/AC + 1), s%pl(AG, c%kv_heads, max_ctx/AC + 1), &
                s%pa(c%head_dim, AG, c%kv_heads, max_ctx/AC + 1), s%rc(c%rotary_dim/2, max_ctx), s%rs(c%rotary_dim/2, max_ctx), &
                s%pv(pad(np)), s%pi(pad(np)), s%tokbuf(pad(1)), s%hist(max(max_ctx, 1024)))
            s%ss = 0
            do t = 1, max_ctx   ! RoPE angles of the positions t - 1
                do j = 1, c%rotary_dim/2
                    s%rc(j,t) = real(cos(mod((t-1) * real(c%rope_theta, real64)**(-2.0_real64*(j-1)/c%rotary_dim), &
                        6.283185307179586_real64)))
                    s%rs(j,t) = real(sin(mod((t-1) * real(c%rope_theta, real64)**(-2.0_real64*(j-1)/c%rotary_dim), &
                        6.283185307179586_real64)))
                end do
            end do
            do l = 1, c%layers
                if (w%l(l)%linear) then
                    allocate(s%l(l)%conv(c%conv_dim, 2*(c%conv_kernel-1)), s%l(l)%ssm(c%linear_key_head_dim, &
                        c%linear_value_head_dim, c%linear_value_heads))
                    s%l(l)%conv = 0
                    s%l(l)%ssm = 0
                else
                    allocate(s%l(l)%kc(c%head_dim, c%kv_heads, max_ctx), s%l(l)%vc(c%head_dim, c%kv_heads, max_ctx))
                end if
            end do
        end associate
    end subroutine

    ! One token's forward pass at position `pos` (from 0): the token in s%tokbuf(1) -> the next one (greedy),
    ! into s%tokbuf(1) and s%hist(pos+1), on the GPU. 5 GEMVs-and-friends per layer: RMSNorm, residual adds
    ! and SwiGLU are fused into the GEMVs.
    subroutine step(w, s, pos)
        type(weights), intent(in) :: w
        type(state), intent(inout) :: s
        integer, intent(in) :: pos
        integer :: l! , debug_h, debug_d, debug_heads, debug_dim
        ! real :: debug_norm
        associate (c => w%c)
            s%k = 3 - s%k
            call embed(w%embed%w, w%embed%s, w%embed%b, s%tokbuf, w%l(1)%in_norm, s%x, s%xw, s%ss, s%k)
            ! if (pos == 21) then
            !     write(0, '(a,8(1x,es16.8))') &
            !     'embedding_residual', s%x(1:8)

            !     write(0, '(a,8(1x,es16.8))') &
            !     'embedding_normalized', s%xw(1:8)
            ! end if
            do l = 1, c%layers
                associate (y => w%l(l), t => s%l(l))
                    call k_gemv_norm(y%in_proj, s%xw, s%ss, s%k, c%rms_eps, .false., s%y)   ! y = W_in rmsnorm(x)
                    if (y%linear) then   ! Gated DeltaNet
                        call gdn(s%y, y%conv_w, y%a_log, y%dt_bias, y%norm_w, c%linear_key_heads, pos, c%rms_eps, &
                            t%conv, t%ssm, s%o)
                    else                 ! full attention
                        call attn_part(s%y, y%q_norm, y%k_norm, s%rc, s%rs, c%rms_eps, pos, t%kc, t%vc, &
                            s%pm, s%pl, s%pa)
                        call attn_combine(s%y, s%pm, s%pl, s%pa, pos, s%o)
                    end if
                    !if (pos == 21) then
                    !    if (y%linear) then
                    !        debug_dim = c%linear_value_head_dim
                    !        debug_heads = c%linear_value_heads
                    !    else
                    !    debug_dim = c%head_dim
                    !    debug_heads = c%query_heads
                    !end if

                    !do debug_h = 1, debug_heads
                    !    debug_norm = 0.0

                    !    do debug_d = 1, debug_dim
                    !        debug_norm = debug_norm + &
                    !        s%o((debug_h-1)*debug_dim + debug_d)**2
                    !    end do

                    !    debug_norm = sqrt(debug_norm)

                    !    write(0, '(a,i2.2,a,i2.2,a,l1,a,es16.8,a,4(1x,es16.8))') &
                    !        'HEAD layer=', l-1, &
                    !        ' head=', debug_h-1, &
                    !        ' linear=', y%linear, &
                    !        ' norm=', debug_norm, &
                    !        ' first4=', &
                    !        s%o((debug_h-1)*debug_dim+1: &
                    !            (debug_h-1)*debug_dim+4)
                    !    end do
                    !end if
                    s%k = 3 - s%k
                    call k_gemv_resid(y%out_proj, s%o, y%post_norm, s%x, s%xw, s%ss, s%k)                ! x += W_out o
                    call k_gemv_norm(y%gate_up, s%xw, s%ss, s%k, c%rms_eps, .true., s%mid)          ! mid = silu(gate) * up
                    s%k = 3 - s%k
                    if (l < c%layers) then
                        call k_gemv_resid(y%down, s%mid, w%l(l+1)%in_norm, s%x, s%xw, s%ss, s%k)         ! x += W_down mid
                    else
                        call k_gemv_resid(y%down, s%mid, w%final_norm, s%x, s%xw, s%ss, s%k)
                    end if
                    !if (pos == 21) then
                    !    write(0, '(a,i2.2,8(1x,es16.8))') &
                    !   'LAYER layer=', l-1, s%x(1:8)
                    !end if
                end associate
            end do
            !if (pos == 21) then
            !    write(0, '(a,8(1x,es16.8))') &
            !        'final_normalized', s%xw(1:8)
            !end if
            call k_gemv_norm(w%lm_head, s%xw, s%ss, s%k, c%rms_eps, .false., s%logits)
            !if (pos == 21) then
            !    write(0, '(a,8(1x,es15.7))') &
            !        'final xw first 8:', s%xw(1:8)

            !    write(0, '(a,8(1x,es15.7))') &
            !        'logits first 8:', s%logits(1:8)

            !    write(0, '(a,es15.7)') &
            !        'logit token 43:', s%logits(43+1)

            !    write(0, '(a,es15.7)') &
            !        'logit token 90562:', s%logits(90562+1)
            !end if
            call argmax_part(s%logits, np, s%pv, s%pi)
            call argmax_final(s%pv, s%pi, np, pos, s%tokbuf, s%hist)
        end associate
    end subroutine

    subroutine k_gemv_norm(m, xw, ss, k, eps, swiglu, y)
        type(qmat), intent(in) :: m
        real, intent(in) :: xw(:), eps
        integer(int64), intent(inout) :: ss(:)
        integer, intent(in) :: k
        logical, intent(in) :: swiglu
        real, intent(inout) :: y(:)
        call gemv_norm(m%w, m%s, m%b, xw, ss, k, eps, swiglu, y)
    end subroutine

    subroutine k_gemv_resid(m, v, wnext, x, xw, ss, k)
        type(qmat), intent(in) :: m
        real, intent(in) :: v(:), wnext(:)
        real, intent(inout) :: x(:), xw(:)
        integer(int64), intent(inout) :: ss(:)
        integer, intent(in) :: k
        call gemv_resid(m%w, m%s, m%b, v, wnext, x, xw, ss, k)
    end subroutine

end module model

program qwen
    use, intrinsic :: iso_fortran_env, only: int32, int64, output_unit, error_unit
    use qwen_tokenizer
    use model
    implicit none
    type(weights) :: w
    type(state) :: s
    type(bpe_tokenizer) :: tok
    type(text_stream) :: stream
    character(:), allocatable :: dir, prompt, text, arg, err, buf
    integer(int32), allocatable :: ids(:), out(:)
    integer :: i, max_new, max_ctx, n_out, queued, nb
    integer, parameter :: ahead = 16   ! decode steps queued ahead of the tokens printed
    logical :: think, raw, show_ids
    integer(int64) :: t0, t1, t2, t3, t4, rate

    dir = '/tmp/certik/model'
    prompt = 'Hello, who are you?'
    max_new = 512
    max_ctx = 8192
    think = .true.
    raw = .false.
    show_ids = .false.
    i = 1
    do while (i <= command_argument_count())
        arg = argument(i)
        select case (arg)
        case ('-m'); i = i + 1; dir = argument(i)
        case ('-p'); i = i + 1; prompt = argument(i)
        case ('-n'); i = i + 1; read(argument(i), *) max_new
        case ('-c'); i = i + 1; read(argument(i), *) max_ctx
        case ('--no-think'); think = .false.
        case ('--raw'); raw = .true.
        case ('--ids'); show_ids = .true.
        case default
            write(error_unit, '(a)') 'usage: qwen [-m DIR] [-p PROMPT] [-n MAX_NEW] [-c MAX_CTX] [--no-think] [--raw] [--ids]'
            error stop
        end select
        i = i + 1
    end do

    call system_clock(t0, rate)
    call load_tokenizer(dir, tok, err)
    if (allocated(err)) error stop err
    if (raw) then
        text = prompt
    else
        text = chat_prompt(prompt, '', think, '')   ! the model's chat template, one user message
    end if
    call encode_text(tok, text, ids)
    if (show_ids) write(error_unit, '(a,i0,a,*(1x,i0))') 'prompt ids (', size(ids), '):', ids
    max_new = min(max_new, max_ctx - size(ids))
    if (max_new <= 0) error stop 'prompt too long for the context'
    call system_clock(t1)
    call load_model(dir, w)
    call alloc_state(w, s, max_ctx)
    call system_clock(t2)
    write(error_unit, '(a,f0.1,a,f0.1,a,i0,a)') 'tokenizer ', real(t1-t0)/rate, ' s, weights + state in ', &
        real(t2-t1)/rate, ' s, prompt ', size(ids), ' tokens'

    ! Prefill: the prompt goes through the same single-token forward pass, one token at a time.
    do i = 1, size(ids)
        s%tokbuf(1) = ids(i)
        call step(w, s, i - 1)
    end do
    call system_clock(t3)

    ! Decode
    allocate(out(max_new))
    allocate(character(64) :: buf)
    call stream_reset(stream)
    if (.not. raw) write(output_unit, '(a)', advance='no') prompt//achar(10)//achar(10)
    if (.not. raw .and. think) write(output_unit, '(a)', advance='no') '<think>'//achar(10)
    flush(output_unit)
    ! The steps run `ahead` tokens ahead of the host, which reads and prints the tokens behind them: the GPU
    ! never waits for the host. Step q (at position size(ids) + q - 1) takes token out(q) and gives out(q+1).
    call system_clock(t4)
    out(1) = s%hist(size(ids))
    n_out = 1
    queued = 0
    do
        if (is_stop_token(tok, out(n_out))) exit
        call stream_feed(tok, stream, out(n_out), buf, nb)
        if (nb > 0) write(output_unit, '(a)', advance='no') buf(1:nb)
        flush(output_unit)
        if (n_out >= max_new) exit
        if (queued < n_out) then
            do while (queued < min(n_out + ahead - 1, max_new - 1))
                queued = queued + 1
                call step(w, s, size(ids) + queued - 1)
            end do
        end if
        n_out = n_out + 1
        out(n_out) = s%hist(size(ids) + n_out - 1)
    end do
    call system_clock(t1)
    write(output_unit, '()')
    flush(output_unit)
    if (show_ids) write(error_unit, '(a,i0,a,*(1x,i0))') 'output ids (', n_out, '):', out(:n_out)
    write(error_unit, '(a,i0,a,f0.3,a,f0.1,a,i0,a,f0.3,a,f0.1,a)') 'prefill: ', size(ids), ' tokens in ', &
        real(t3-t2)/rate, ' s (', size(ids)/(real(t3-t2)/rate), ' tok/s); decode: ', n_out - 1, ' tokens in ', &
        real(t1-t4)/rate, ' s (', (n_out - 1)/(real(t1-t4)/rate), ' tok/s)'

contains

    function argument(i) result(a)
        integer, intent(in) :: i
        character(:), allocatable :: a
        integer :: n
        if (i > command_argument_count()) error stop 'missing argument value'
        call get_command_argument(i, length=n)
        allocate(character(n) :: a)
        call get_command_argument(i, a)
    end function

end program qwen
