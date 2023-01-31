"""
    Onda.read_byte_range(path::S3Path, byte_offset, byte_count)

Implement method needed for Onda to read a byte range from an S3 path.  Uses
`AWSS3.s3_get` under the hood.

!!! note
    This is technically type piracy, but neither Onda nor AWSS3 makes sense to 
    house this method so until such a package exists, here it shall remain.
"""
function Onda.read_byte_range(path::S3Path, byte_offset, byte_count)
    # s3_get byte_range is 1-indexed, so we need to add one
    byte_range = range(byte_offset + 1; length=byte_count)
    return read(path; byte_range)
end

# avoid method ambiguity
function Onda.read_byte_range(path::S3Path, ::Missing, ::Missing)
    return read(path)
end

# glue together a bunch of N-d arrays on the N+1th dimension; used to create
# N+1-dim tensors during batch materialization
function _glue(stuff)
    sizes = unique(size.(stuff))
    if length(sizes) > 1
        sizes_str = join(sizes, ", ")
        throw(ArgumentError("all elements must have the same size, " *
                            "got >1 unique sizes: $(sizes_str)"))
    end
    stuff_size = only(unique(size.(stuff)))
    # use collect to ensure this is a vector-of-vectors
    stuff_vec = reduce(vcat, vec.(collect(stuff)))
    return reshape(stuff_vec, stuff_size..., length(stuff))
end

#####
##### channel wrangling
#####

is_closed_ex(ex) = false
is_closed_ex(ex::InvalidStateException) = ex.state == :closed
is_closed_ex(ex::RemoteException) = is_closed_ex(ex.captured.ex)
# for use in retry:
check_is_closed_ex(_, ex) = is_closed_ex(ex)

function retry_materialize_batch(batch; retries=4)
    @debug "worker $(myid()) materializing batch" batch
    return Base.retry(materialize_batch; delays=ExponentialBackOff(; n=retries),
                      check=!check_is_closed_ex)(batch)
end

"""
    with_channel(f, channel; closed_msg="channel close, stopping")

Run `f(channel)`, handling channel closure gracefully and closing the channel if
an error is caught.

If the channel is closed, the `InvalidStateException` is caught, the
`closed_msg` is logged as `@info`, and `:closed` is returned.

If any other error occurs, the channel is closed before rethrowing (with a
`@debug` log message reporting the error + stacktrace).

Otherwise, the return value is `f(channel)`.
"""
function with_channel(f, channel; closed_msg="channel close, stopping")
    try
        return f(channel)
    catch e
        if is_closed_ex(e)
            @info closed_msg
            return :closed
        else
            # close the channel to communicate to anyone waiting on these
            # batches that a problem has occurred
            msg = sprint(showerror, e, catch_backtrace())
            @debug "caught exception, closing channel and re-throwing: $msg"
            close(channel)
            rethrow()
        end
    end
end

#####
##### WorkerPool workarounds
#####

"""
    reset!(pool::AbstractWorkerPool)

Restore worker pool to something like the state it would be in after
construction, with the channel populated with one instance of each worker
managed by the pool.

This has two phases: first, the contents of the channel are cleared out to avoid
double-adding workers to the channel.  Second, the contents of `pool.workers` is
sorted, checked against the list of active processes with `procs()`, and then
live PIDs `put!` into the pool one-by-one.  Dead workers are removed from the
set of workers held by the pool.

For a `WorkerPool`, this operation is forwarded to the process holding the
original pool (as with `put!`, `take!`, etc.) so it is safe to call on
serialized copies of the pool.

`nothing` is returned.
"""
reset!(pool::AbstractWorkerPool) = reset!(pool)
function reset!(pool::WorkerPool)
    if pool.ref.where != myid()
        return remotecall_fetch(ref -> _local_reset!(fetch(ref).value),
                                pool.ref.where,
                                pool.ref)::Nothing
    else
        return _local_reset!(pool)
    end
end

function _local_reset!(pool::AbstractWorkerPool)
    # clean out existing workers so that we're not double-put!ing workers into
    # the channel.  we work directly with teh channel to work around
    # https://github.com/JuliaLang/julia/issues/48255
    while isready(pool.channel)
        take!(pool.channel)
    end
    live_procs = Set(procs())
    for worker in sort!(collect(pool.workers))
        # don't put worker back in pool if it's dead
        if worker in live_procs
            put!(pool, worker)
        else
            delete!(pool.workers, worker)
        end
    end
    return nothing
end

# there isn't a Base.wait method for worker pools.  `take!` blocks but removes a
# worker and we don't want that.  teh workaround here is to `wait` on the
# `.channel` field, which is consistent with the docs description of relying on
# a `.channel` field for fallback implementations of the API methods like
# `take!`:
# https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.AbstractWorkerPool
#
# this is contributed upstream but it may be ... some time before it's usable
# here so in the mean time...
# 
# https://github.com/JuliaLang/julia/pull/48238
_wait(p::AbstractWorkerPool) = wait(p.channel)

# but we also need to handle when a WorkerPool has been serialized, so this is
# basically copy-pasta from Distributed.jl stdlib:
function _wait(pool::WorkerPool)
    # just for my own understanding later on: WorkerPool handles serialization
    # by storing a "ref" as a RemoteChannel which contains the workerpool
    # itself.  this is created when the WorkerPool is created so the "where"
    # field is the worker id which holds the actual pool; deserializing the pool
    # somewhere else creates a dummy Channel (for available workers) and
    # deserializes the ref.
    #
    # so in order to wait on a worker, we have to first determine whether our
    # copy of the pool is the "real" one
    if pool.ref.where != myid()
        # this is the "remote" branch, so we remotecall `wait` on 
        return remotecall_fetch(ref -> wait(fetch(ref).value.channel),
                                pool.ref.where,
                                pool.ref)::Nothing
    else
        return wait(pool.channel)
    end
end
