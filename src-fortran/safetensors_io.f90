module safetensors_io
    use, intrinsic :: iso_fortran_env, only: int16, int32, int64, real32
    use model_io_support
    use json_tokens
    implicit none
    private
    public :: tensor_archive, open_archive, find_tensor
    public :: read_bf16, read_u32, read_f32

    type :: tensor_info
        character(:), allocatable :: name, path, dtype
        integer(int64), allocatable :: shape(:)
        integer(int64) :: byte_offset=0, nbytes=0, elements=0
    end type
    type :: shard_info
        character(:), allocatable :: path
    end type
    type :: tensor_archive
        type(tensor_info), allocatable :: tensor(:)
    end type

    interface read_bf16
        module procedure read_bf16_flat, read_bf16_matrix
    end interface
    interface read_u32
        module procedure read_u32_flat, read_u32_matrix
    end interface
    interface read_f32
        module procedure read_f32_flat
    end interface

contains

    subroutine inspect_shard(path, tensors, error)
        character(*), intent(in) :: path
        type(tensor_info), allocatable, intent(out) :: tensors(:)
        type(io_error), intent(out) :: error
        type(json_document) :: doc
        character(8) :: prefix
        character(:), allocatable :: header
        integer :: unit, ios, i, j, key, info, arr, off, count, item_bytes, k, metadata
        integer, allocatable :: order(:)
        integer(int64) :: file_bytes, length, base, lo, hi, bytes, dim, finish, cursor
        call require_byte_io(error)
        if (error%failed()) return
        open(newunit=unit, file=path, status='old', access='stream', form='unformatted', action='read', iostat=ios)
        if (ios /= 0) then
            call fail(error, 'cannot open safetensors shard: '//path)
            return
        end if
        inquire(unit=unit, size=file_bytes, iostat=ios)
        if (ios /= 0) then
            call fail(error, 'cannot inquire shard size: '//path)
        else if (file_bytes < 8) then
            call fail(error, 'truncated safetensors length prefix: '//path)
        end if
        if (.not. error%failed()) then
            read(unit, pos=1_int64, iostat=ios) prefix
            if (ios /= 0) call fail(error, 'truncated safetensors length prefix: '//path)
        end if
        length=0
        if (.not. error%failed()) then
            if (btest(iachar(prefix(8:8)),7)) call fail(error, 'safetensors header length exceeds signed 64-bit range')
            do i=1,8
                do j=0,7
                    if (btest(iachar(prefix(i:i)),j)) length=ibset(length,8*(i-1)+j)
                end do
            end do
            if (length < 2 .or. length > max_json_bytes) call fail(error, 'safetensors header length outside 2..16 MiB')
            if (length > file_bytes-8) call fail(error, 'truncated safetensors header: '//path)
        end if
        if (.not. error%failed()) then
            allocate(character(int(length)) :: header, stat=ios)
            if (ios /= 0) then
                call fail(error, 'safetensors header allocation failed')
            else
                read(unit, pos=9_int64, iostat=ios) header
                if (ios /= 0) call fail(error, 'truncated safetensors header: '//path)
            end if
        end if
        close(unit)
        if (error%failed()) return
        base=8+length
        call parse_json(header, doc, error)
        if (error%failed()) return
        call expect_kind(doc,1,json_object,error)
        if (error%failed()) return
        metadata=member(doc,1,'__metadata__')
        count=doc%token(1)%count
        if (metadata /= 0) then
            call expect_kind(doc,metadata,json_object,error)
            if (error%failed()) return
            key=doc%token(metadata)%first
            do i=1,doc%token(metadata)%count
                call expect_kind(doc,key+1,json_text,error)
                key=doc%token(key+1)%next
            end do
            count=count-1
        end if
        if (error%failed()) return
        if (count < 1) then
            call fail(error, 'safetensors shard contains no tensors')
            return
        end if
        allocate(tensors(count),order(count),stat=ios)
        if (ios /= 0) then
            call fail(error, 'safetensors descriptor allocation failed')
            return
        end if
        key=doc%token(1)%first
        i=0
        do k=1,doc%token(1)%count
            info=key+1
            if (info == metadata) then
                key=doc%token(info)%next
                cycle
            end if
            i=i+1
            tensors(i)%name=doc%token(key)%text
            tensors(i)%path=path
            if (len(tensors(i)%name) == 0) call fail(error, 'empty tensor name')
            call expect_kind(doc,info,json_object,error)
            tensors(i)%dtype=json_string(doc,member(doc,info,'dtype'),error)
            select case (tensors(i)%dtype)
            case ('BF16')
                item_bytes=2
                if (len(tensors(i)%dtype) /= 4) call fail(error, 'unsupported tensor dtype: '//tensors(i)%name)
            case ('U32')
                item_bytes=4
                if (len(tensors(i)%dtype) /= 3) call fail(error, 'unsupported tensor dtype: '//tensors(i)%name)
            case ('F32')
                item_bytes=4
                if (len(tensors(i)%dtype) /= 3) call fail(error, 'unsupported tensor dtype: '//tensors(i)%name)
            case default
                call fail(error, 'unsupported tensor dtype (expected BF16 or U32 or F32): '//tensors(i)%name)
                return
            end select
            if (error%failed()) return
            arr=member(doc,info,'shape')
            call expect_kind(doc,arr,json_array,error)
            if (error%failed()) return
            if (doc%token(arr)%count > 6) then
                call fail(error, 'tensor rank exceeds 6: '//tensors(i)%name)
                return
            end if
            allocate(tensors(i)%shape(doc%token(arr)%count),stat=ios)
            if (ios /= 0) then
                call fail(error, 'tensor shape allocation failed')
                return
            end if
            tensors(i)%elements=1
            do j=1,size(tensors(i)%shape)
                dim=json_integer(doc,element(doc,arr,j),error)
                if (dim <= 0) call fail(error, 'tensor dimensions must be positive: '//tensors(i)%name)
                tensors(i)%shape(j)=dim
                tensors(i)%elements=checked_multiply(tensors(i)%elements,dim,error)
            end do
            bytes=checked_multiply(tensors(i)%elements,int(item_bytes,int64),error)
            if (error%failed()) return
            off=member(doc,info,'data_offsets')
            call expect_kind(doc,off,json_array,error)
            if (error%failed()) return
            if (doc%token(off)%count /= 2) then
                call fail(error, 'data_offsets must have two integers: '//tensors(i)%name)
                return
            end if
            lo=json_integer(doc,element(doc,off,1),error)
            hi=json_integer(doc,element(doc,off,2),error)
            if (error%failed()) return
            if (lo < 0 .or. hi < lo .or. hi > file_bytes-base) then
                call fail(error, 'tensor data_offsets outside shard: '//tensors(i)%name)
                return
            end if
            if (hi-lo /= bytes) then
                call fail(error, 'tensor shape/bytecount mismatch: '//tensors(i)%name)
                return
            end if
            tensors(i)%byte_offset=checked_add(base,lo,error)
            tensors(i)%nbytes=bytes
            order(i)=i
            key=doc%token(info)%next
        end do
        ! Safetensors data is a complete, nonoverlapping buffer, not arbitrary file ranges.
        do i=2,count
            k=order(i)
            j=i-1
            do while (j >= 1)
                if (tensors(order(j))%byte_offset <= tensors(k)%byte_offset) exit
                order(j+1)=order(j)
                j=j-1
            end do
            order(j+1)=k
        end do
        cursor=base
        do i=1,count
            k=order(i)
            if (tensors(k)%byte_offset /= cursor) then
                call fail(error, 'tensor data has overlap or hole: '//tensors(k)%name)
                return
            end if
            finish=checked_add(cursor,tensors(k)%nbytes,error)
            if (error%failed()) return
            cursor=finish
        end do
        if (cursor /= file_bytes) call fail(error, 'unindexed trailing safetensors data')
    end subroutine

    subroutine open_archive(directory, archive, error)
        character(*), intent(in) :: directory
        type(tensor_archive), intent(out) :: archive
        type(io_error), intent(out) :: error
        type(json_document) :: doc
        type(tensor_info), allocatable :: tensors(:)
        type(shard_info), allocatable :: shards(:)
        character(:), allocatable :: filename, path
        integer :: map, key, i, j, s, n, ns, nt, index, ios, metadata, total
        integer(int64) :: declared_bytes, payload_bytes
        integer, allocatable :: mapping(:)
        logical, allocatable :: seen(:)
        call read_json(trim(directory)//'/model.safetensors.index.json',doc,error)
        if (error%failed()) return
        call expect_kind(doc,1,json_object,error)
        map=member(doc,1,'weight_map')
        call expect_kind(doc,map,json_object,error)
        if (error%failed()) return
        declared_bytes=-1
        metadata=member(doc,1,'metadata')
        if (metadata /= 0) then
            call expect_kind(doc,metadata,json_object,error)
            total=member(doc,metadata,'total_size')
            if (total /= 0) then
                declared_bytes=json_integer(doc,total,error)
                if (declared_bytes < 0) call fail(error, 'index metadata total_size must be nonnegative')
            end if
        end if
        if (error%failed()) return
        n=doc%token(map)%count
        if (n < 1) then
            call fail(error, 'weight_map must not be empty')
            return
        end if
        allocate(archive%tensor(n),shards(n),mapping(n),seen(n),stat=ios)
        if (ios /= 0) then
            call fail(error, 'archive metadata allocation failed')
            return
        end if
        seen=.false.
        ns=0
        key=doc%token(map)%first
        do i=1,n
            archive%tensor(i)%name=doc%token(key)%text
            filename=json_string(doc,key+1,error)
            if (error%failed()) return
            if (.not. safe_shard_name(filename)) then
                call fail(error, 'weight_map shard must be a local .safetensors basename')
                return
            end if
            path=trim(directory)//'/'//filename
            s=0
            do j=1,ns
                if (shards(j)%path == path) s=j
            end do
            if (s == 0) then
                ns=ns+1
                s=ns
                shards(s)%path=path
            end if
            mapping(i)=s
            key=doc%token(key+1)%next
        end do
        nt=0
        payload_bytes=0
        do s=1,ns
            call inspect_shard(shards(s)%path,tensors,error)
            if (error%failed()) return
            do j=1,size(tensors)
                index=find_tensor(archive,tensors(j)%name)
                if (index == 0) then
                    call fail(error, 'shard tensor absent from weight_map: '//tensors(j)%name)
                    return
                end if
                if (seen(index)) then
                    call fail(error, 'duplicate tensor across shards: '//tensors(j)%name)
                    return
                end if
                if (mapping(index) /= s) then
                    call fail(error, 'weight_map shard mismatch: '//tensors(j)%name)
                    return
                end if
                archive%tensor(index)=tensors(j)
                seen(index)=.true.
                nt=nt+1
                payload_bytes=checked_add(payload_bytes,tensors(j)%nbytes,error)
            end do
            if (error%failed()) return
        end do
        if (nt /= n .or. .not. all(seen)) call fail(error, 'weight_map references missing tensor')
        if (declared_bytes >= 0) then
            if (declared_bytes /= payload_bytes) call fail(error, 'index metadata total_size disagrees with tensor payload')
        end if
    end subroutine

    logical function safe_shard_name(name) result(ok)
        character(*), intent(in) :: name
        integer :: i, c
        ok=.false.
        if (len(name) <= 12) return
        if (name(len(name)-11:) /= '.safetensors') return
        do i=1,len(name)
            c=iachar(name(i:i))
            if (.not. ((c >= 65 .and. c <= 90) .or. (c >= 97 .and. c <= 122) .or. &
                    (c >= 48 .and. c <= 57) .or. name(i:i) == '.' .or. &
                    name(i:i) == '-' .or. name(i:i) == '_')) return
        end do
        if (index(name,'..') /= 0) return
        ok=.true.
    end function

    integer function find_tensor(archive, name) result(index)
        type(tensor_archive), intent(in) :: archive
        character(*), intent(in) :: name
        integer :: i
        index=0
        if (.not. allocated(archive%tensor)) return
        do i=1,size(archive%tensor)
            if (same_text(archive%tensor(i)%name,name)) then
                index=i
                return
            end if
        end do
    end function

    subroutine payload_unit(tensor, dtype, count, unit, position, error)
        type(tensor_info), intent(in) :: tensor
        character(*), intent(in) :: dtype
        integer(int64), intent(in) :: count
        integer, intent(out) :: unit
        integer(int64), intent(out) :: position
        type(io_error), intent(inout) :: error
        integer(int64) :: bytes, finish, length, width, tensor_finish
        integer :: ios
        call require_byte_io(error)
        unit=0
        position=0
        if (error%failed()) return
        if (.not. allocated(tensor%dtype) .or. .not. allocated(tensor%path)) then
            call fail(error, 'payload requires initialized tensor metadata')
            return
        end if
        if (.not. same_text(tensor%dtype,dtype)) call fail(error, 'typed payload dtype mismatch: '//tensor%name)
        if (count < 1) call fail(error, 'payload requires positive count')
        if (error%failed()) return
        if (count > tensor%elements) then
            call fail(error, 'payload range exceeds tensor elements')
            return
        end if
        select case (dtype)
        case ('BF16')
            width=2
        case ('U32', 'F32')
            width=4
        case default
            call fail(error, 'unsupported payload dtype: '//tensor%name)
            return
        end select
        bytes=checked_multiply(tensor%elements,width,error)
        if (bytes /= tensor%nbytes) call fail(error, 'payload descriptor bytecount mismatch')
        tensor_finish=checked_add(tensor%byte_offset,tensor%nbytes,error)
        position=tensor%byte_offset
        bytes=checked_multiply(count,width,error)
        finish=checked_add(position,bytes,error)
        position=checked_add(position,1_int64,error)
        if (error%failed()) return
        open(newunit=unit,file=tensor%path,status='old',access='stream',form='unformatted',action='read',iostat=ios)
        if (ios /= 0) then
            call fail(error, 'cannot open tensor payload: '//tensor%path)
            return
        end if
        inquire(unit=unit,size=length,iostat=ios)
        if (ios /= 0) then
            call fail(error, 'cannot inquire tensor payload size')
        else if (finish > length .or. tensor_finish > length) then
            call fail(error, 'truncated tensor payload: '//tensor%name)
        end if
        if (error%failed()) close(unit)
    end subroutine

    subroutine read_bf16_flat(tensor, values, error)
        type(tensor_info), intent(in) :: tensor
        integer(int16), intent(out) :: values(:)
        type(io_error), intent(out) :: error
        character(65536) :: buffer
        integer :: unit, ios, n, j, p, high
        integer(int64) :: position, done, count
        count=size(values,kind=int64)
        call payload_unit(tensor,'BF16',count,unit,position,error)
        if (error%failed()) return
        done=0
        do while (done < count)
            n=int(min(count-done,int(len(buffer)/2,int64)))
            read(unit,pos=position,iostat=ios) buffer(:2*n)
            if (ios /= 0) then
                call fail(error, 'truncated/read error in BF16 payload: '//tensor%name)
                exit
            end if
            do j=1,n
                p=2*j
                high=iachar(buffer(p:p))
                values(done+j)=ior(int(iachar(buffer(p-1:p-1)),int16),shiftl(int(iand(high,127),int16),8))
                if (btest(high,7)) values(done+j)=ibset(values(done+j),15)
            end do
            done=done+n
            if (done < count) position=checked_add(position,int(2*n,int64),error)
            if (error%failed()) exit
        end do
        close(unit)
    end subroutine

    subroutine read_u32_flat(tensor, values, error)
        type(tensor_info), intent(in) :: tensor
        integer(int32), intent(out) :: values(:)
        type(io_error), intent(out) :: error
        character(65536) :: buffer
        integer :: unit, ios, n, j, p, high
        integer(int64) :: position, done, count
        count=size(values,kind=int64)
        call payload_unit(tensor,'U32',count,unit,position,error)
        if (error%failed()) return
        done=0
        do while (done < count)
            n=int(min(count-done,int(len(buffer)/4,int64)))
            read(unit,pos=position,iostat=ios) buffer(:4*n)
            if (ios /= 0) then
                call fail(error, 'truncated/read error in U32 payload: '//tensor%name)
                exit
            end if
            do j=1,n
                p=4*j
                high=iachar(buffer(p:p))
                values(done+j)=ior(ior(int(iachar(buffer(p-3:p-3)),int32), &
                    shiftl(int(iachar(buffer(p-2:p-2)),int32),8)), &
                    ior(shiftl(int(iachar(buffer(p-1:p-1)),int32),16), &
                    shiftl(int(iand(high,127),int32),24)))
                if (btest(high,7)) values(done+j)=ibset(values(done+j),31)
            end do
            done=done+n
            if (done < count) position=checked_add(position,int(4*n,int64),error)
            if (error%failed()) exit
        end do
        close(unit)
    end subroutine

    subroutine read_f32_flat(tensor, values, error)
        type(tensor_info), intent(in) :: tensor
        real(real32), intent(out) :: values(:)
        type(io_error), intent(out) :: error
        integer :: unit, ios
        integer(int64) :: position, count
        count=size(values,kind=int64)
        call payload_unit(tensor,'F32',count,unit,position,error)
        if (error%failed()) return
        read(unit,pos=position,iostat=ios) values
        if (ios /= 0) then
            call fail(error, 'truncated/read error in F32 payload: '//tensor%name)
        end if
        close(unit)
    end subroutine

    subroutine require_layout(tensor, fortran_shape, error)
        type(tensor_info), intent(in) :: tensor
        integer(int64), intent(in) :: fortran_shape(:)
        type(io_error), intent(inout) :: error
        integer :: i, rank
        if (.not. allocated(tensor%shape)) then
            call fail(error, 'typed layout requires initialized tensor shape')
            return
        end if
        rank=size(fortran_shape)
        if (size(tensor%shape) /= rank) then
            call fail(error, 'typed layout rank mismatch')
            return
        end if
        do i=1,rank
            if (fortran_shape(i) /= tensor%shape(rank+1-i)) call fail(error, 'typed layout dimensions mismatch')
        end do
    end subroutine

    subroutine read_bf16_matrix(tensor, values, error)
        type(tensor_info), intent(in) :: tensor
        integer(int16), contiguous, target, intent(out) :: values(:,:)
        type(io_error), intent(out) :: error
        integer(int16), pointer :: flat(:)
        call require_layout(tensor,shape(values,kind=int64),error)
        if (error%failed()) return
        flat(1:size(values,kind=int64)) => values
        call read_bf16_flat(tensor,flat,error)
    end subroutine

    subroutine read_u32_matrix(tensor, values, error)
        type(tensor_info), intent(in) :: tensor
        integer(int32), contiguous, target, intent(out) :: values(:,:)
        type(io_error), intent(out) :: error
        integer(int32), pointer :: flat(:)
        call require_layout(tensor,shape(values,kind=int64),error)
        if (error%failed()) return
        flat(1:size(values,kind=int64)) => values
        call read_u32_flat(tensor,flat,error)
    end subroutine

end module
