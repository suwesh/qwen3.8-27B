! GPU kernels for Qwen3.8-27B (MLX 8-bit) single-token decode: the kernels of src-cuda/kernels.cu as
! do concurrent loops, which LFortran offloads to the GPU (--gpu=cuda). fp32 activations; the weights
! stay in the checkpoint's q8 + bf16 scale/bias format.
!
! A layer is 4 GEMVs plus a Gated DeltaNet or an attention: RMSNorm, the residual adds and SwiGLU are
! fused into the GEMVs. The residual stream and its RMSNorm (as `resid` in model.h):
!   x    residual (H)
!   xw   x * (weight of the RMSNorm that comes next), the input of the next GEMV
!   ss   sum(x**2) in 64-bit fixed point (scale 2**24), in slot k of two: the producer of x (embed,
!        gemv_resid) adds its rows into ss(k) with a reduce; the next gemv_norm reads ss(k), multiplies
!        its rows by rsqrt(ss(k)/H + eps), which gives W (rmsnorm(x) * weight), and zeroes the other
!        slot for the producer after it. Integer sums are the same in any order: the result does not
!        depend on the order the GPU adds them in.
!
! In a do concurrent over 2 or 3 indices the last one varies fastest between GPU threads. An inner
! sequential loop that only accumulates into scalars is shared out over the lanes of the iteration (a
! warp, or the threads of a block). The Gated DeltaNet and the attention give an iteration to a head and
! write its steps as phases, inner loops the lanes share, over arrays local to the iteration (which
! LFortran keeps in the block's shared memory).
module kernels
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real64
    implicit none
    private
    public :: qmat, bf, embed, gemv_norm, gemv_resid, gdn
    public :: attn_part, attn_combine, argmax_part, argmax_final, AC, AG, AMAX

    ! One MLX affine 8-bit matrix (group size 64) in the checkpoint's own layout: w(:, n) holds the K
    ! bytes of row n (the U32 tensor [N, K/4]), s(:, n) and b(:, n) its K/64 bf16 scales and biases.
    ! Element (k, n) = byte * scale + bias.
    type :: qmat
        integer(int32), allocatable :: w(:,:)
        integer(int16), allocatable :: s(:,:), b(:,:)
    end type

    integer, parameter :: GD = 128   ! linear attention key/value head dim
    integer, parameter :: HD = 256   ! full attention head dim
    integer, parameter :: AG = 4     ! full attention query heads per kv head - suwesh changed this hardcoded value for 27b model to hardcoded value for 0.8b model
    integer, parameter :: AC = 8     ! positions per chunk of the attention (attn_part)
    integer, parameter :: AMAX = 8192 ! the most positions of the attention (attn_combine)

contains

    ! bf16 bits -> real
    pure elemental real function bf(v)
        integer(int16), intent(in) :: v
        bf = transfer(shiftl(int(v, int32), 16), 0.0)
    end function

    pure real function sigmoid(x)
        real, intent(in) :: x
        sigmoid = 1 / (1 + exp(-x))
    end function

    ! acc + the 4 bytes of w dotted with x0..x3, as a chain of fused multiply-adds. A byte as a real:
    ! 2**23 + byte has it as its low mantissa bits.
    pure real function dot4(acc, w, x0, x1, x2, x3)
        real, intent(in) :: acc, x0, x1, x2, x3
        integer(int32), intent(in) :: w
        dot4 = acc + (transfer(ior(iand(w, 255), 1258291200), 0.0) - 8388608.0) * x0 &
             + (transfer(ior(iand(shiftr(w, 8), 255), 1258291200), 0.0) - 8388608.0) * x1 &
             + (transfer(ior(iand(shiftr(w, 16), 255), 1258291200), 0.0) - 8388608.0) * x2 &
             + (transfer(ior(shiftr(w, 24), 1258291200), 0.0) - 8388608.0) * x3
    end function

    ! ------------------------------------------------------------ embedding

    ! Element (k, n) of the q8 matrix (w, s, b)
    pure real function q8(w, s, b, k, n)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        integer, intent(in) :: k, n
        q8 = ibits(w((k-1)/4+1, n), 8*modulo(k-1, 4), 8) * bf(s((k-1)/64+1, n)) + bf(b((k-1)/64+1, n))
    end function

    ! x = E(:, tokbuf(1)); xw = x * wn; ss(k) += sum(x**2) in fixed point (scale 2**24)
    subroutine embed(w, s, b, tokbuf, wn, x, xw, ss, k)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        integer, intent(in) :: tokbuf(:), k
        real, intent(in) :: wn(:)
        real, intent(inout) :: x(:), xw(:)
        integer(int64), intent(inout) :: ss(:)
        do concurrent (integer :: i = 1:size(x)/4) reduce(+: ss)
            x(4*i-3) = q8(w, s, b, 4*i-3, tokbuf(1)+1)
            x(4*i-2) = q8(w, s, b, 4*i-2, tokbuf(1)+1)
            x(4*i-1) = q8(w, s, b, 4*i-1, tokbuf(1)+1)
            x(4*i) = q8(w, s, b, 4*i, tokbuf(1)+1)
            xw(4*i-3) = x(4*i-3) * wn(4*i-3)
            xw(4*i-2) = x(4*i-2) * wn(4*i-2)
            xw(4*i-1) = x(4*i-1) * wn(4*i-1)
            xw(4*i) = x(4*i) * wn(4*i)
            ss(k) = ss(k) + nint(((x(4*i-3)*x(4*i-3) + x(4*i-2)*x(4*i-2)) + (x(4*i-1)*x(4*i-1) + x(4*i)*x(4*i))) &
                * 2.0**24, int64)
        end do
    end subroutine

    ! ------------------------------------------------------------ q8 GEMV
    ! An iteration computes 4 consecutive rows; its lanes (a warp, or two for a matrix with few rows)
    ! take its chunks of 16 bytes, c = lane, lane + 32, ..., and combine their sums at the end.

    ! Chunk c (16 bytes) of row n dotted with x: scale * sum(byte * x) + bias * sum(x), with one scale
    ! and bias per 4 chunks.
    pure real function qdot(w, s, b, x, n, c)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        real, intent(in) :: x(:)
        integer, intent(in) :: n, c
        qdot = bf(s((c+3)/4,n)) * dot4(dot4(dot4(dot4(0.0, w(4*c-3,n), x(16*c-15), x(16*c-14), x(16*c-13), x(16*c-12)), &
            w(4*c-2,n), x(16*c-11), x(16*c-10), x(16*c-9), x(16*c-8)), w(4*c-1,n), x(16*c-7), x(16*c-6), x(16*c-5), x(16*c-4)), &
            w(4*c,n), x(16*c-3), x(16*c-2), x(16*c-1), x(16*c)) &
            + bf(b((c+3)/4,n)) * ((((x(16*c-15) + x(16*c-14)) + (x(16*c-13) + x(16*c-12))) &
            + ((x(16*c-11) + x(16*c-10)) + (x(16*c-9) + x(16*c-8)))) + (((x(16*c-7) + x(16*c-6)) &
            + (x(16*c-5) + x(16*c-4))) + ((x(16*c-3) + x(16*c-2)) + (x(16*c-1) + x(16*c)))))
    end function

    ! y = rsqrt(sum(ssp)/H + eps) * (W xw). swiglu: the rows are interleaved gate/up pairs
    ! (g1 g2 u1 u2 g3 g4 u3 u4 ...) and y(i) = silu(gate_i) * up_i, for N/2 outputs.
    subroutine gemv_norm(w, s, b, xw, ss, k, eps, swiglu, y)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        real, intent(in) :: xw(:), eps
        integer(int64), intent(inout) :: ss(:)
        integer, intent(in) :: k
        logical, intent(in) :: swiglu
        real, intent(inout) :: y(:)
        do concurrent (integer :: i = 1:size(w, 2)/4)
            block
                real :: a1, a2, a3, a4, r
                integer :: c, n
                n = 4*i - 3
                a1 = 0; a2 = 0; a3 = 0; a4 = 0
                do c = 1, size(w, 1)/4
                    a1 = a1 + qdot(w, s, b, xw, n, c)
                    a2 = a2 + qdot(w, s, b, xw, n+1, c)
                    a3 = a3 + qdot(w, s, b, xw, n+2, c)
                    a4 = a4 + qdot(w, s, b, xw, n+3, c)
                end do
                r = 1 / sqrt(real(real(ss(k), real64) / 2.0_real64**24) / size(xw) + eps)
                if (swiglu) then
                    y(2*i-1) = a1 * r * sigmoid(a1 * r) * (a3 * r)
                    y(2*i) = a2 * r * sigmoid(a2 * r) * (a4 * r)
                else
                    y(n) = a1 * r
                    y(n+1) = a2 * r
                    y(n+2) = a3 * r
                    y(n+3) = a4 * r
                end if
                if (i == 1) ss(3 - k) = 0   ! the other slot, for the next producer of x
            end block
        end do
    end subroutine

    ! x += W v; then xw = x * wnext and ssp, for the next RMSNorm (weight wnext)
    subroutine gemv_resid(w, s, b, v, wnext, x, xw, ss, k)
        integer(int32), intent(in) :: w(:,:)
        integer(int16), intent(in) :: s(:,:), b(:,:)
        real, intent(in) :: v(:), wnext(:)
        real, intent(inout) :: x(:), xw(:)
        integer(int64), intent(inout) :: ss(:)
        integer, intent(in) :: k
        do concurrent (integer :: i = 1:size(w, 2)/4) reduce(+: ss)
            block
                real :: a1, a2, a3, a4
                integer :: c, n
                n = 4*i - 3
                a1 = 0; a2 = 0; a3 = 0; a4 = 0
                do c = 1, size(w, 1)/4
                    a1 = a1 + qdot(w, s, b, v, n, c)
                    a2 = a2 + qdot(w, s, b, v, n+1, c)
                    a3 = a3 + qdot(w, s, b, v, n+2, c)
                    a4 = a4 + qdot(w, s, b, v, n+3, c)
                end do
                x(n) = x(n) + a1
                x(n+1) = x(n+1) + a2
                x(n+2) = x(n+2) + a3
                x(n+3) = x(n+3) + a4
                xw(n) = x(n) * wnext(n)
                xw(n+1) = x(n+1) * wnext(n+1)
                xw(n+2) = x(n+2) * wnext(n+2)
                xw(n+3) = x(n+3) * wnext(n+3)
                ss(k) = ss(k) + nint(((x(n)*x(n) + x(n+1)*x(n+1)) + (x(n+2)*x(n+2) + x(n+3)*x(n+3))) * 2.0**24, int64)
            end block
        end do
    end subroutine

    ! ------------------------------------------------------------ linear attention (Gated DeltaNet)
    ! y = in_proj output [qkv (C) | z (nv*128) | b (nv) | a (nv)], C = 2*nk*128 + nv*128. The state
    ! S(k, v, h) of value head h is updated as S = g*S + khat * delta^T, delta = (v - g*S^T khat)*beta,
    ! and the output o = S^T qhat comes from the old state: o = g*(S^T qhat) + delta*(khat . qhat).

    ! One value head per iteration: the conv of its q, k, v channels, the Gated DeltaNet step and the gated
    ! RMSNorm. st(v, j, h): the state, v fastest. cs(c, 1:6): conv inputs, double buffered by position parity.
    subroutine gdn(y, cw, a_log, dt_bias, nw, nk, p, eps, cs, st, out)
        real, intent(in) :: y(:), cw(:,:), a_log(:), dt_bias(:), nw(:), eps
        integer, intent(in) :: nk, p
        real, intent(inout) :: cs(:,:), st(:,:,:), out(:)
        integer :: nv, c, h
        nv = size(st, 3)
        c = size(cw, 2)
        do concurrent (h = 1:nv)
            block
                real :: q(GD), k(GD), v(GD), o(GD)
                real :: kk, qq, kn, qn, sp, g, beta, ss, r, sk, sq, delta, z
                integer :: j, i, kq0, kk0, vc0, pi, po
                kq0 = ((h-1) / (nv/nk)) * GD
                kk0 = nk*GD + kq0
                vc0 = 2*nk*GD + (h-1)*GD
                pi = 3*mod(p, 2)
                po = 3 - pi
                do concurrent (j = 1:GD)
                    q(j) = cw(1,kq0+j)*cs(kq0+j,pi+1) + cw(2,kq0+j)*cs(kq0+j,pi+2) + cw(3,kq0+j)*cs(kq0+j,pi+3) + cw(4,kq0+j)*y(kq0+j)
                    q(j) = q(j) * sigmoid(q(j))
                    k(j) = cw(1,kk0+j)*cs(kk0+j,pi+1) + cw(2,kk0+j)*cs(kk0+j,pi+2) + cw(3,kk0+j)*cs(kk0+j,pi+3) + cw(4,kk0+j)*y(kk0+j)
                    k(j) = k(j) * sigmoid(k(j))
                    v(j) = cw(1,vc0+j)*cs(vc0+j,pi+1) + cw(2,vc0+j)*cs(vc0+j,pi+2) + cw(3,vc0+j)*cs(vc0+j,pi+3) + cw(4,vc0+j)*y(vc0+j)
                    v(j) = v(j) * sigmoid(v(j))
                end do
                kk = 0; qq = 0
                do j = 1, GD
                    kk = kk + k(j) * k(j)
                    qq = qq + q(j) * q(j)
                end do
                kn = 1 / sqrt(kk + 1e-6)
                qn = 1 / sqrt(qq + 1e-6) / sqrt(real(GD))
                sp = y(c + nv*GD + nv + h) + dt_bias(h)
                if (sp <= 20) sp = log(1 + exp(sp))
                g = exp(-exp(a_log(h)) * sp)
                beta = sigmoid(y(c + nv*GD + h))
                do concurrent (i = 1:GD)
                    sk = 0; sq = 0
                    do j = 1, GD
                        sk = sk + st(i,j,h) * k(j)
                    end do
                    delta = (v(i) - g * sk * kn) * beta
                    do j = 1, GD
                        st(i,j,h) = g * st(i,j,h) + k(j) * kn * delta
                        sq = sq + st(i,j,h) * q(j)
                    end do
                    o(i) = sq * qn
                end do
                ss = 0
                do j = 1, GD
                    ss = ss + o(j) * o(j)
                end do
                r = 1 / sqrt(ss / GD + eps)
                do concurrent (i = 1:GD)
                    z = y(c + (h-1)*GD + i)
                    out((h-1)*GD + i) = o(i) * r * nw(i) * z * sigmoid(z)
                end do
                if (mod(h-1, nv/nk) == 0) then
                    do concurrent (j = 1:GD)
                        cs(kq0+j,po+1) = cs(kq0+j,pi+2); cs(kq0+j,po+2) = cs(kq0+j,pi+3); cs(kq0+j,po+3) = y(kq0+j)
                        cs(kk0+j,po+1) = cs(kk0+j,pi+2); cs(kk0+j,po+2) = cs(kk0+j,pi+3); cs(kk0+j,po+3) = y(kk0+j)
                    end do
                end if
                do concurrent (j = 1:GD)
                    cs(vc0+j,po+1) = cs(vc0+j,pi+2); cs(vc0+j,po+2) = cs(vc0+j,pi+3); cs(vc0+j,po+3) = y(vc0+j)
                end do
            end block
        end do
    end subroutine

    ! ------------------------------------------------------------ full attention
    ! y = [q|gate per head (nq*512) | k (nkv*256) | v (nkv*256)]. q and k get an RMSNorm and a
    ! rotate-half RoPE on their first 64 dims (cos/sin of the 32 angles of position t: rc(:, t+1), rs(:, t+1)).

    ! The attention of the new token (at position p, from 0) by kv head and chunk of AC positions: the
    ! RMSNorm and RoPE of the kv head's AG query heads and of its key, the scores of the chunk's positions,
    ! their softmax within the chunk, and the weighted sum of the chunk's values (flash decoding). The chunk
    ! with the new position first appends its key and value to the caches kc(d, kh, t) and vc(d, kh, t).
    ! The maximum, the sum of the exponentials and the weighted sum of each query head and chunk go to
    ! pm(i, kh, c), pl(i, kh, c) and pa(d, i, kh, c), which attn_combine merges.
    subroutine attn_part(y, qw, kw, rc, rs, eps, p, kc, vc, pm, pl, pa)
        real, intent(in) :: y(:), qw(:), kw(:), rc(:,:), rs(:,:), eps
        integer, intent(in) :: p
        real, intent(inout) :: kc(:,:,:), vc(:,:,:), pm(:,:,:), pl(:,:,:), pa(:,:,:,:)
        integer :: nkv, nq
        nkv = size(kc, 2)
        nq = AG*nkv
        do concurrent (integer :: kh = 1:nkv, c = 1:p/AC + 1)
            block
                real :: x(HD, AG+1), w(AC, AG), nrm(AG+1)
                real :: ss, cs, sn, s, m, l
                integer :: i, j, d, dp, u, t, b
                ! the norms of the AG query heads and of the key
                do concurrent (i = 1:AG+1)
                    b = merge(((kh-1)*AG + i - 1)*2*HD, nq*2*HD + (kh-1)*HD, i <= AG)
                    ss = 0
                    do j = 1, HD
                        ss = ss + y(b+j)**2
                    end do
                    nrm(i) = 1 / sqrt(ss/HD + eps)
                end do
                ! normalized and rotated (rotate-half RoPE on dims 1..64), the queries scaled by 1/sqrt(HD)
                do concurrent (d = 1:HD)
                    if (d <= 32) then
                        dp = d + 32; cs = rc(d,p+1); sn = -rs(d,p+1)
                    else if (d <= 64) then
                        dp = d - 32; cs = rc(d-32,p+1); sn = rs(d-32,p+1)
                    else
                        dp = d; cs = 1; sn = 0
                    end if
                    do i = 1, AG
                        b = ((kh-1)*AG + i - 1)*2*HD
                        x(d,i) = (y(b+d)*qw(d)*cs + y(b+dp)*qw(dp)*sn) * nrm(i) / 16
                    end do
                    b = nq*2*HD + (kh-1)*HD
                    x(d,AG+1) = (y(b+d)*kw(d)*cs + y(b+dp)*kw(dp)*sn) * nrm(AG+1)
                end do
                if (c == p/AC + 1) then
                    do concurrent (d = 1:HD)
                        kc(d,kh,p+1) = x(d,AG+1)
                        vc(d,kh,p+1) = y(nq*2*HD + nkv*HD + (kh-1)*HD + d)
                    end do
                end if
                ! the scores of the chunk's positions (past the new one, again the new one)
                do concurrent (u = 1:AC)
                    t = min((c-1)*AC + u, p+1)
                    do i = 1, AG
                        s = 0
                        do d = 1, HD
                            s = s + x(d,i)*kc(d,kh,t)
                        end do
                        w(u,i) = s
                    end do
                end do
                ! their softmax within the chunk: no weight past the new position
                do concurrent (i = 1:AG)
                    m = -huge(1.0)
                    do u = 1, AC
                        m = max(m, merge(w(u,i), -huge(1.0), (c-1)*AC + u <= p+1))
                    end do
                    l = 0
                    do u = 1, AC
                        w(u,i) = merge(exp(w(u,i) - m), 0.0, (c-1)*AC + u <= p+1)
                        l = l + w(u,i)
                    end do
                    pm(i,kh,c) = m
                    pl(i,kh,c) = l
                end do
                ! the weighted sums of the chunk's values, by dimension: each value read once for all the
                ! query heads
                do concurrent (d = 1:HD)
                    block
                        real :: v(AC)
                        do u = 1, AC
                            v(u) = vc(d,kh,min((c-1)*AC + u, p+1))
                        end do
                        do i = 1, AG
                            s = 0
                            do u = 1, AC
                                s = s + w(u,i)*v(u)
                            end do
                            pa(d,i,kh,c) = s
                        end do
                    end block
                end do
            end block
        end do
    end subroutine

    ! out = softmax(scores) V * sigmoid(gate), from the chunks of attn_part: by query head, the weight of
    ! each chunk (its exponentials' share of the sum of all), then the weighted sum of the chunks' sums.
    subroutine attn_combine(y, pm, pl, pa, p, out)
        real, intent(in) :: y(:), pm(:,:,:), pl(:,:,:), pa(:,:,:,:)
        integer, intent(in) :: p
        real, intent(inout) :: out(:)
        integer :: nkv
        nkv = size(pm, 2)
        do concurrent (integer :: h = 1:AG*nkv)
            block
                real :: f(AMAX/AC + 1)
                real :: mm, den, num
                integer :: c, i, kh, d
                kh = (h-1)/AG + 1
                i = h - (kh-1)*AG
                mm = -huge(1.0)
                do c = 1, p/AC + 1
                    mm = max(mm, pm(i,kh,c))
                end do
                do concurrent (c = 1:p/AC + 1)
                    f(c) = exp(pm(i,kh,c) - mm)
                end do
                den = 0
                do c = 1, p/AC + 1
                    den = den + f(c)*pl(i,kh,c)
                end do
                do concurrent (d = 1:HD)
                    num = 0
                    do c = 1, p/AC + 1
                        num = num + f(c)*pa(d,i,kh,c)
                    end do
                    out((h-1)*HD + d) = num/den * sigmoid(y((h-1)*2*HD + HD + d))
                end do
            end block
        end do
    end subroutine

    ! ------------------------------------------------------------ argmax
    ! Part i of np: the largest of its x(j) and the first j it is at.
    subroutine argmax_part(x, np, pv, pi)
        real, intent(in) :: x(:)
        integer, intent(in) :: np
        real, intent(inout) :: pv(:)
        integer, intent(inout) :: pi(:)
        integer :: len
        len = (size(x) + np - 1) / np
        do concurrent (integer :: i = 1:np)
            block
                real :: m
                integer :: j, bi
                m = -huge(m)
                do j = (i-1)*len + 1, min(i*len, size(x))
                    m = max(m, x(j))
                end do
                bi = huge(bi)
                do j = (i-1)*len + 1, min(i*len, size(x))
                    if (x(j) == m) bi = min(bi, j)
                end do
                pv(i) = m
                pi(i) = bi
            end block
        end do
    end subroutine


    ! The token of the largest of the parts (pv, pi) into tokbuf(1) (for the next embed) and hist(pos+1)
    subroutine argmax_final(pv, pi, np, pos, tokbuf, hist)
        real, intent(in) :: pv(:)
        integer, intent(in) :: pi(:), np, pos
        integer, intent(inout) :: tokbuf(:), hist(:)
        do concurrent (integer :: i = 1:1)
            block
                real :: m
                integer :: j, bi
                m = -huge(m)
                do j = 1, np
                    m = max(m, pv(j))
                end do
                bi = huge(bi)
                do j = 1, np
                    if (pv(j) == m) bi = min(bi, pi(j))
                end do
                tokbuf(i) = bi - 1
                hist(pos+i) = bi - 1
            end block
        end do
    end subroutine

end module kernels
