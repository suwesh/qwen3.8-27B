module qwen_config
    use, intrinsic :: iso_fortran_env, only: int64, real32, real64
    use model_io_support
    use json_tokens
    implicit none
    private
    public :: dense_config, read_config

    type :: dense_config
        integer :: layers=0, hidden=0, intermediate=0, vocab=0, full_interval=0
        integer :: query_heads=0, kv_heads=0, head_dim=0, query_dim=0, kv_dim=0
        integer :: linear_key_heads=0, linear_value_heads=0, linear_key_head_dim=0, linear_value_head_dim=0
        integer :: linear_key_dim=0, linear_value_dim=0, conv_dim=0, conv_kernel=0
        integer :: rotary_dim=0, quant_bits=0, group_size=0
        logical, allocatable :: full_attention(:)
        real(real32) :: rms_eps=0, rope_theta=0, rotary_fraction=0
    end type

contains

    subroutine read_config(path, config, error)
        character(*), intent(in) :: path
        type(dense_config), intent(out) :: config
        type(io_error), intent(out) :: error
        type(json_document) :: doc
        integer :: tc, rp, q, i, types, item, arch, index, ios
        integer(int64) :: value, derived
        real(real64) :: fraction, rotary
        character(:), allocatable :: name
        call read_json(path,doc,error)
        if (error%failed()) return
        call expect_kind(doc,1,json_object,error)
        call string_is(doc,1,'model_type','qwen3_5',error)
        tc=member(doc,1,'text_config')
        call expect_kind(doc,tc,json_object,error)
        call string_is(doc,tc,'model_type','qwen3_5_text',error)
        arch=member(doc,1,'architectures')
        call expect_kind(doc,arch,json_array,error)
        if (error%failed()) return
        if (doc%token(arch)%count /= 1) call fail(error,'config: unsupported architectures')
        name=json_string(doc,element(doc,arch,1),error)
        if (.not. same_text(name,'Qwen3_5ForConditionalGeneration')) call fail(error,'config: unsupported architecture')
        call string_is(doc,tc,'hidden_act','silu',error)
        ! call string_is(doc,tc,'output_gate_type','swish',error) ! for loading the qwen 3.5 0.8b model which doesnot have this param in config.json
        call string_is(doc,tc,'dtype','bfloat16',error)
        call string_is(doc,tc,'mamba_ssm_dtype','float32',error)
        call boolean_is(doc,tc,'attention_bias',.false.,error)
        ! call boolean_is(doc,tc,'tie_word_embeddings',.false.,error)
        ! call boolean_is(doc,1,'tie_word_embeddings',.false.,error)
        call boolean_is(doc,tc,'attn_output_gate',.true.,error)
        value=optional_integer(doc,tc,'num_experts',0_int64,error)
        if (value /= 0) call fail(error,'config: MoE is not supported by this dense loader')
        value=optional_integer(doc,tc,'num_experts_per_tok',0_int64,error)
        if (value /= 0) call fail(error,'config: dense FFN cannot select experts')
        q=member(doc,1,'quantization')
        if (q == 0) q=member(doc,1,'quantization_config')
        call quantization(doc,q,error)
        index=member(doc,1,'quantization_config')
        if (index /= 0) call quantization(doc,index,error)
        if (error%failed()) return
        config%quant_bits=8
        config%group_size=64
        config%layers=positive(doc,tc,'num_hidden_layers',error)
        config%hidden=positive(doc,tc,'hidden_size',error)
        config%intermediate=positive(doc,tc,'intermediate_size',error)
        config%vocab=positive(doc,tc,'vocab_size',error)
        config%full_interval=positive(doc,tc,'full_attention_interval',error)
        config%query_heads=positive(doc,tc,'num_attention_heads',error)
        config%kv_heads=positive(doc,tc,'num_key_value_heads',error)
        config%head_dim=positive(doc,tc,'head_dim',error)
        config%linear_key_heads=positive(doc,tc,'linear_num_key_heads',error)
        config%linear_value_heads=positive(doc,tc,'linear_num_value_heads',error)
        config%linear_key_head_dim=positive(doc,tc,'linear_key_head_dim',error)
        config%linear_value_head_dim=positive(doc,tc,'linear_value_head_dim',error)
        config%conv_kernel=positive(doc,tc,'linear_conv_kernel_dim',error)
        config%rms_eps=positive_real(doc,tc,'rms_norm_eps',error)
        rp=member(doc,tc,'rope_parameters')
        call expect_kind(doc,rp,json_object,error)
        item=member(doc,tc,'rope_scaling')
        if (item /= 0) then
            if (doc%token(item)%kind /= json_null) call fail(error,'config: unsupported rope_scaling')
        end if
        call string_is(doc,rp,'rope_type','default',error)
        config%rope_theta=positive_real(doc,rp,'rope_theta',error)
        config%rotary_fraction=positive_real(doc,rp,'partial_rotary_factor',error)
        fraction=real(config%rotary_fraction,real64)
        if (fraction > 1) call fail(error,'config: partial_rotary_factor exceeds one')
        item=member(doc,tc,'partial_rotary_factor')
        if (item /= 0) then
            rotary=json_real(doc,item,error)
            if (abs(rotary-fraction) > 0) call fail(error,'config: conflicting partial_rotary_factor')
        end if
        if (error%failed()) return
        config%query_dim=dimension(config%query_heads,config%head_dim,error)
        config%kv_dim=dimension(config%kv_heads,config%head_dim,error)
        config%linear_key_dim=dimension(config%linear_key_heads,config%linear_key_head_dim,error)
        config%linear_value_dim=dimension(config%linear_value_heads,config%linear_value_head_dim,error)
        derived=checked_multiply(2_int64,int(config%linear_key_dim,int64),error)
        derived=checked_add(derived,int(config%linear_value_dim,int64),error)
        if (derived > huge(0)) call fail(error,'config: convolution dimension exceeds kernel integer range')
        if (error%failed()) return
        config%conv_dim=int(derived)
        derived=checked_multiply(2_int64,int(config%query_dim,int64),error)
        if (derived > huge(0)) call fail(error,'config: gated query dimension exceeds kernel integer range')
        rotary=real(config%head_dim,real64)*fraction
        config%rotary_dim=int(rotary)
        if (abs(rotary-real(config%rotary_dim,real64)) > 0) call fail(error,'config: nonintegral rotary dimension')
        if (config%rotary_dim < 2 .or. modulo(config%rotary_dim,2) /= 0) &
            call fail(error,'config: rotary dimension must be positive and even')
        if (modulo(config%query_heads,config%kv_heads) /= 0 .or. &
                modulo(config%linear_value_heads,config%linear_key_heads) /= 0 .or. config%conv_kernel < 2 .or. &
                modulo(config%hidden,64) /= 0 .or. modulo(config%intermediate,64) /= 0 .or. &
                modulo(config%query_dim,64) /= 0 .or. modulo(config%linear_value_dim,64) /= 0) &
            call fail(error,'config: incompatible head/group/quantization dimensions')
        if (config%full_interval < 2 .or. config%full_interval > config%layers) &
            call fail(error,'config: expected hybrid recurrent/full-attention schedule')
        types=member(doc,tc,'layer_types')
        call expect_kind(doc,types,json_array,error)
        if (error%failed()) return
        if (doc%token(types)%count /= config%layers) then
            call fail(error,'config: layer_types must cover every layer')
            return
        end if
        allocate(config%full_attention(config%layers),stat=ios)
        if (ios /= 0) then
            call fail(error,'config: layer schedule allocation failed')
            return
        end if
        do i=1,config%layers
            name=json_string(doc,element(doc,types,i),error)
            config%full_attention(i)=modulo(i,config%full_interval) == 0
            if (config%full_attention(i)) then
                if (.not. same_text(name,'full_attention')) call fail(error,'config: layer_types/interval mismatch')
            else
                if (.not. same_text(name,'linear_attention')) call fail(error,'config: layer_types/interval mismatch')
            end if
        end do
    end subroutine

    subroutine quantization(doc, object, error)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        type(io_error), intent(inout) :: error
        integer(int64) :: value
        call expect_kind(doc,object,json_object,error)
        value=json_integer(doc,member(doc,object,'bits'),error)
        if (value /= 8) call fail(error,'config: only 8-bit quantization is supported')
        value=json_integer(doc,member(doc,object,'group_size'),error)
        if (value /= 64) call fail(error,'config: only group_size=64 is supported')
        call string_is(doc,object,'mode','affine',error)
    end subroutine

    subroutine string_is(doc, object, key, expected, error)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        character(*), intent(in) :: key, expected
        type(io_error), intent(inout) :: error
        character(:), allocatable :: value
        if (error%failed()) return
        value=json_string(doc,member(doc,object,key),error)
        if (error%failed()) then
            error%message='config '//key//': '//error%message
        else if (.not. same_text(value,expected)) then
            call fail(error,'config: unsupported '//key//' (expected '//expected//')')
        end if
    end subroutine

    subroutine boolean_is(doc, object, key, expected, error)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        character(*), intent(in) :: key
        logical, intent(in) :: expected
        type(io_error), intent(inout) :: error
        logical :: value
        if (error%failed()) return
        value=json_boolean(doc,member(doc,object,key),error)
        if (error%failed()) error%message='config '//key//': '//error%message
        if (value .neqv. expected) call fail(error,'config: unsupported '//key)
    end subroutine

    integer function positive(doc, object, key, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        character(*), intent(in) :: key
        type(io_error), intent(inout) :: error
        integer(int64) :: n
        value=0
        if (error%failed()) return
        n=json_integer(doc,member(doc,object,key),error)
        if (error%failed()) then
            error%message='config '//key//': '//error%message
            return
        end if
        if (n < 1 .or. n > huge(0)) then
            call fail(error,'config: '//key//' must be positive and fit kernel integer range')
        else
            value=int(n)
        end if
    end function

    function optional_integer(doc, object, key, default, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        character(*), intent(in) :: key
        integer(int64), intent(in) :: default
        type(io_error), intent(inout) :: error
        integer(int64) :: value
        integer :: index
        value=default
        if (error%failed()) return
        index=member(doc,object,key)
        if (index /= 0) value=json_integer(doc,index,error)
    end function

    function positive_real(doc, object, key, error) result(value)
        type(json_document), intent(in) :: doc
        integer, intent(in) :: object
        character(*), intent(in) :: key
        type(io_error), intent(inout) :: error
        real(real32) :: value
        real(real64) :: n
        value=0
        if (error%failed()) return
        n=json_real(doc,member(doc,object,key),error)
        if (error%failed()) then
            error%message='config '//key//': '//error%message
            return
        end if
        if (n < real(tiny(value),real64) .or. n > real(huge(value),real64)) then
            call fail(error,'config: '//key//' must be finite, positive and representable as real32')
        else
            value=real(n,real32)
        end if
    end function

    integer function dimension(a, b, error) result(value)
        integer, intent(in) :: a,b
        type(io_error), intent(inout) :: error
        integer(int64) :: n
        value=0
        n=checked_multiply(int(a,int64),int(b,int64),error)
        if (n > huge(0)) call fail(error,'config: derived dimension exceeds kernel integer range')
        if (.not. error%failed()) value=int(n)
    end function
end module
