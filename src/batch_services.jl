#####
##### Single-worker batching
#####

"""
    start_batching(channel::RemoteChannel, batches, state)

Begin loading batches onto a `RemoteChannel` based on batches (e.g.,
[`RandomBatches`](@ref)) and initial state.

This will run an infinite loop which loads one batch at a time with
[`iterate_batch`](@ref) and [`materialize_batch`](@ref), and `put!`s the
resulting `(x, y)` and `state` values into the channel.

Batching continues until the channel is closed or an error is encountered.  When
the channel is closed, the `InvalidStateException` is caught and `:closed` is
returned from the function.  Other errors are rethrown.  If somehow the loop is
exited without an error (honestly not sure how this would happen), `:done` is
returned.

This function is intended to used with `@async` or `remotecall` (e.g., in a
[`Batcher`](@ref)); the `Future` that `remotecall` returns can be monitored
with [`get_status`](@ref).

Calls to [`materialize_batch`](@ref) are wrapped in `Base.retry` to add some
measure of resiliency to transient network interruptions.

Runs on the batching manager (i.e. `Batcher.manager`), but only when
`Batcher.workers` is empty.
"""
function start_batching(channel::RemoteChannel, batches, state)
    @debug "Starting batching..."
    closed_msg = "batch channel closed, stopping batching"
    return with_channel(channel; closed_msg) do channel
        batch_counter = 1
        # setup: grab prev state and next batch+state to start
        prev_state = deepcopy(state)
        next = iterate_batch(batches, state)
        # iteration loop: go until iterate_batch returns nothing
        while next !== nothing
            # unpack iterated pair
            batch, state = next
            # probably (xx, yy) but who knows
            batch_out = retry_materialize_batch(batch; retries=4)
            next_out = (batch_out, deepcopy(state))
            @debug "loaded batch $(batch_counter):" batch state
            put!(channel, (next_out, prev_state))
            @debug "put batch on channel:" batch state
            batch_counter += 1
            # next iteration: grab prev state, iterate next pair
            prev_state = deepcopy(state)
            next = iterate_batch(batches, state)
        end
        # we need this last thing in order to support synchronization.
        # consumers call `take!` on the batcher, which first fetches the
        # previous state, and if it's consistent with the requested state,
        # proceeds to fetch the next batch+state.
        put!(channel, (nothing, prev_state))
        close(channel)
        return :done
    end
end

#####
##### Multi-worker batching
#####

# represents a single `materialize_batch` job
struct BatchJob
    worker::Union{Int,Nothing}
    batch_future::Union{Future,Nothing}
    state::Any
    prev_state::Any
end

"""
    _feed_jobs!(jobs::Channel, batches, state, workers)

Function that iterates `batches` starting from `state`, creating a `BatchJob` to
materialize each one using the pool of `workers`.  Each job holds is put onto
the `jobs` channel in the order they were iterated, and is a struct with fields
- `worker` PID of the worker loading this batch
- `batch_future` a `Future` containing the output of `materialize_batch`
- `state` the iteration state after iterating this batch
- `prev_state` the iteration state before iterating this batch (i.e., the input
  to `iterate_batch(batches, state)` required to reproduce this batch

When batch iteration is complete (as indicated by `iterate_batch` returning
`nothing`, a final placeholder job will be placed on the jobs channel, with
values of `nothing` everywhere except for `prev_state`, which is required to
support synchronization on the client end (i.e., to confirm that the user really
did ask for the final batch with `take!`).

Returns `nothing`.

Runs on the batching manager (i.e., `Batcher.manager`), in an async Task created
in `start_batching`.
"""
function _feed_jobs!(jobs::Channel, batches, state, workers)
    prev_state = deepcopy(state)
    next = iterate_batch(batches, state)
    # iteration loop: go until iterate_batch returns nothing
    while next !== nothing
        batch, state = next
        # why wait here?  we don't want to take a worker if the jobs channel
        # isn't still open, but we do want to block creating more jobs until a
        # worker is available to run it.  so we wait...(_wait actually)
        _wait(workers)
        # ...check that the jobs channel is still open, returning if not...
        isopen(jobs) || return nothing
        # ...and if is, we take the worker that (hopefully) is still ready
        @debug "feeder: taking worker from pool..."
        worker = take!(workers)
        @debug "feeder: materializing batch on worker $(worker) for state $(state)"
        batch_future = remotecall(retry_materialize_batch,
                                  worker, batch; retries=4)
        job = BatchJob(worker, batch_future, deepcopy(state), prev_state)
        put!(jobs, job)
        prev_state = deepcopy(state)
        next = iterate_batch(batches, state)
    end
    @debug "finished: feeder exited at state $(state)"
    # we always need to put a final batch on the output with the correct
    # previous state to support synchronization by the consumer, so
    # rather than just closing the channel, put a whole lotta nothing w/
    # the correct prev_state onto our internal channel
    put!(jobs, BatchJob(nothing, nothing, nothing, prev_state))
    return nothing
end

"""
    start_batching(channel::RemoteChannel, batches, state, workers)

Start batching loop, utilizing multiple workers to load batches in parallel.
This method will yield batches in the same order that `start_batching` without
`workers` will, using a [`_feed_jobs!`](@ref) feed batch materialization jobs to
an internal channel (maintaining iteration order while distributing work across
`workers`).

Runs on the batching manager (i.e. `Batcher.manager`)
"""
function start_batching(channel::RemoteChannel, batches, state, workers)
    # we need to be sure that the worker pool has active workers or else we may
    # deadlock waiting on a job below... #54 
    reset!(workers)
    @debug "Starting batching on $(length(workers)) workers..."
    # batches are assigned to workers to materialize using a worker pool: the
    # feeder task takes a worker from the pool, and the consumer loop returns
    # the worker to the pool when the batch is ready.  this controls the number
    # of batches that are being worked on simultaneously.
    jobs = Channel{BatchJob}(Inf)
    feeder = @async begin
        # wrap this in a `with_channel` to gracefully handle closure of the jobs
        # channel
        with_channel(jobs) do jobs
            _feed_jobs!(jobs, batches, state, workers)
        end
    end

    # this will ensure that the jobs channel is closed when teh feeder task
    # completes AND forward any errors thrown on the feeder task to anyone
    # waiting on `jobs` (i.e. the main loop below)
    bind(jobs, feeder)

    # create a sentinel task that will close the jobs channel if/when the output
    # channel is closed.  this is necessary because we are waiting on the jobs
    # channel below which may block if there's resource starvation, but we still
    # need to be able to handle the closure of the output channel since that's
    # how the client communicates that batching should be stopped prematurely.
    sentinel = @async begin
        while isopen(channel)
            sleep(1)
        end
        @debug "output channel closed, closing jobs channel"
        close(jobs)
    end
    Base.errormonitor(sentinel)

    try
        closed_msg = "batch channel closed, stopping batching"
        status = with_channel(channel; closed_msg) do channel
            for (; worker, batch_future, state, prev_state) in jobs
                if batch_future === nothing
                    # to support synchronization from the consumer, we need to put
                    # one "final" "batch" on the channel with `nothing` in place of
                    # the materialized batch + next state tuple.
                    put!(channel, (nothing, prev_state))
                else
                    # TODO: consider only `wait` here, and put future directly onto
                    # channel.
                    local materialized
                    try
                        materialized = fetch(batch_future)
                        @debug "returning worker $(worker) to pool"
                        put!(workers, worker)
                    catch
                        # in the event of an error, close the jobs channel to
                        # stop the feeder task
                        @debug "caught exception, closing jobs channel"
                        close(jobs)
                        rethrow()
                    end
                    @debug "putting batch onto channel" state
                    put!(channel, ((materialized, state), prev_state))
                end
            end
            # because we may be waiting on jobs, we may exit the `for ... in
            # jobs` loop on channel closure without hitting an
            # InvalidStateException which would cause `with_channel` to return
            # `:closed`.  so we need an additional manual check here...
            return isopen(channel) ? :done : :closed
        end

        # rethrow possible task failed exception once we finish with all the
        # good batches (task will close channel on failure).
        if istaskfailed(feeder)
            @debug "feeder task failed, fetching to rethrow"
            fetch(feeder)
        end

        # if the feeder task is not done and we've gotten here, something has
        # gone wrong and we should notify the external world
        status != :closed && !istaskdone(feeder) &&
            error("`start_batching` feeder task is not done but internal job channel closed!")

        return status
    finally
        # always make sure the jobs channel is closed and all workers for
        # in-flight jobs are returned to the pool
        close(jobs)
        reset!(workers)
    end
end

"""
A struct that provides control of batching process on one or more remote
workers.  This struct keeps track of

- `manager::Int` the PID where `start_batching` will be called.
- `workers` an `AbstractWorkerPool` for the worker process(es).
- `channel::RemoteChannel` the channel that batches are loaded into.
- `status::Future` the return value of the `start_batching` function as a
  Future; see [`get_status`](@ref) for a convenient accessor.
- `batches` the iterator of batches that will be materialized; only requirement
  is that [`iterate_batch`](@ref) be defined; see [`RandomBatches`](@ref) for an
  example
- `state::Any` batcher state (passed to [`iterate_batch`](@ref), updated with
  each new batch that's yielded by the batcher.
- `buffer::Int` the size of the batch buffer to keep locally (e.g., the capacity 
  of `channel`).

Use [`start!`](@ref) to start the batching service, [`stop!`](@ref) to stop it,
and [`get_status`](@ref) to check the status.

Once the batcher is started, the sequence of materialized batches (the output of
[`materialize_batch`](@ref)) and corresponding batcher states can be retrieved
by [`take!`](@ref).

## Architecture

A `Batcher` is meant to run in an architecture where remote workers are created
with a Distributed.jl cluster manager.  We use the following terminology to
describe the roles these different processes play:

- "Batch worker": one or more processes that are used to actually load batches
  (via [`materialize_batch`](@ref))

- "Batch manager": the process which coordinates the loading of batches,
  ensuring consistent iteration order, distributing work to the batch workers,
  and populating the output channel.  [`start_batching`](@ref) runs on this
  process.

- "Client": the process which is consuming batches via `take!(::Batcher, state)`
  (which OndaBatches.jl is generally agnostic about and does not manage)

- "Manager": the process on which the `Batcher` is initially created, and holds
  the reference for the worker pool (for multi-worker batching).

!!! note
    We try hard to make `Batcher`s relocatable to other processes (e.g.,
    serializing to the Client after initialization on the Manager).  However,
    since a new `RemoteChannel` is created each time the batcher is started
    (including when the desired state does not match the `Batcher`'s current
    state), some care needs to be taken if it matters where that channel is
    hosted (although this behavior may change in the future).

    Also note that while a running (i.e. `start!`ed) `Batcher` can be relocated
    to another process, the `status` and `channel` fields are not guaranteed to
    stay in sync on the two copies.
"""
Base.@kwdef mutable struct Batcher
    manager::Int
    workers::AbstractWorkerPool
    channel::RemoteChannel
    status::Future
    batches::Any
    buffer::Int
end

function Batcher(manager::Int, workers::Vector{Int}, batches; kwargs...)
    # why not a CachingPool?  they're not serializable, and it's generally
    # important to be able to serialize a Batcher.  so this is a sensible
    # default that users can override as they need to.  also, in general we are
    # not doing a lot of `remotecall`s with chonky closures, so that negates
    # most of the benefits of a CachingPool.
    pool = WorkerPool(workers)
    return Batcher(manager, pool, batches; kwargs...)
end    

function Batcher(workers::Vector{Int}, batches; kwargs...)
    # first worker is going to be the manager
    manager, workers = Iterators.peel(workers)
    return Batcher(manager, collect(workers), batches; kwargs...)
end

"""
    Batcher([manager::Int,] workers::Vector{Int}, batches; start=true, state=nothing, buffer=2 * length(workers) + 1)
    Batcher(manager::Int, workers::AbstractWorkerPool, batches; start=true, state=nothing, buffer=2 * length(workers) + 1)

Construct a new [`Batcher`](@ref), using worker IDs, batches, and initial state.
The batcher's channel and status will be initialized.

The `workers` may be specified as an `AbstractWorkerPool` or a vector of PIDs
(in which case a `WorkerPool` will be constructed).

!!! warning
    If workers are supplied as an `AbstractWorkerPool`, it is assumed that _all_
    workers managed by the pool are available for loading batches.  Whenever the
    batcher is stopped, the worker pool is reset, and all managed workers are
    returned to the channel of available workers.

See [`RandomBatches`](@ref) for an example of creation of `batches`.

The initial `state` is the state that is used by
[`iterate_batch`](@ref), e.g., the RNG used by [`RandomBatches`](@ref).

If `start=true`, batching is [`start!`](@ref)ed. The `state` keyword argument must be supplied in this case to provide an initial state.

The `buffer` controls the capacity of the batch channel; a value greater than or
equal to the number of workers is recommended so that batch loading workers do
not block waiting for batches to be taken off the channel.
"""
function Batcher(manager::Int, workers::AbstractWorkerPool, batches;
                 start=true, state=nothing,
                 buffer=2 * length(workers) + 1)
    channel = RemoteChannel(() -> Channel{Any}(buffer))

    status = Future()
    put!(status, :stopped)

    batcher = Batcher(; manager, workers, channel, status, batches, buffer)
    if start
        state === nothing &&
            throw(ArgumentError("state must have a value when `start`=true"))
        start!(batcher, state)
    end
    return batcher
end

"""
    get_status(batcher::Batcher)

Check the status of a remote batcher.

Possible return values are
- `:stopped`: the batcher was created but not started
- `:running`: the batching loop is still running
- `:closed`: the batch channel was closed and the batch loop has terminated
- `:done`: the infinite loop in [`start_batching`](@ref) has terminated without 
  error (not expected)
- a `RemoteException` that wraps an error thrown by `start_batching` on the
  batch manager (which may further wrap an exception thrown on a batch worker
"""
get_status(batcher::Batcher) = get_status(batcher.status)
get_status(s::Future) = isready(s) ? fetch_and_catch(s) : :running

function fetch_and_catch(s::Future)
    try 
        return fetch(s)
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        @error "batcher status error: $msg"
        return e
    end
end

"""
    start!(batcher::Batcher, state)

Start the remote process that loads batches into the batcher's channel.  A new
channel is created since the old one cannot always be re-used.

This invokes [`start_batching`](@ref) on `batcher.manager` with `remotecall`.

The (modified) batcher is returned.

If the batcher is already running (as indicated by [`get_status ==
:running`](@ref get_status)), a warning is raised and the batcher is returned.

Runs on the Client.
"""
function start!(batcher::Batcher, state)
    (; manager, workers, batches, status, buffer) = batcher

    # status is a Future that `isready` is not running, and `!isready` if it is
    # still running and needs to be stopped.
    if get_status(batcher) == :running
        @warn "batcher already running; use `stop!` to halt before `start!`"
        return batcher
    end

    # it's not really possible to check whether a channel is closed without
    # (possibly) blocking so we just create a new one every time we start.
    channel = RemoteChannel(() -> Channel{Any}(buffer))

    if length(workers) > 0
        # length is the total number of workers in the pool, regardless of
        # whether they're available for work or not.
        length(workers) == 1 && @warn "only one extra worker to load batches!"
        @info "starting multi-worker batching, with manager $(manager) and workers $(workers), at state $(state)"
        batcher.status = remotecall(start_batching,
                                    manager,
                                    channel,
                                    batches,
                                    state,
                                    workers)
    else
        @info "starting batching on worker $(manager) at state $(state)"
        # TODO: `isready` docs say we should `put!` this future into a Future owned
        # by this process
        batcher.status = remotecall(start_batching,
                                    manager,
                                    channel,
                                    batches,
                                    state)
    end
    batcher.channel = channel
    return batcher
end

"""
    stop!(batcher::Batcher)

Close `batcher.channel` to stop the remote batching.  This blocks on
`fetch(batcher.status)` to wait for channel closure.  If an error is thrown on
the remote worker that is not caught, it will be rethrown here.

The batcher is returned.

Runs on the Client.
"""
function stop!(batcher::Batcher)
    (; channel, status, workers) = batcher
    @info "stopping batcher"
    @debug "closing channel"
    # where = 0 when channel has been finalized. close on a finalize channel
    # throws a pretty opaque error.
    channel.where != 0 && close(channel)
    @debug "waiting for done status"
    # catch errors here so we can stop the batcher even if there was an error
    status = fetch_and_catch(status)
    # need to finalize this in order to release remote refs for GC
    finalize(channel)
    return batcher
end

"""
    Base.take!(batcher::Batcher, state)

Take one batch + state pair from the batcher, starting at the specified state.
If the requested state does not match the batcher's current state, then the
batching process will be restarted with the new state.  If the batcher is not
running (as indicated by [`get_status`](@ref)), it will be started with
[`start!`](@ref).

If an error has occurred on any of the batch loading workers, the next call to
`take!` will immediately throw the wrapped `RemoteException`, even if there are
still good batches on the channel.

Returns an `(x, y), state` tuple, where `x` is the batch signal data, `y` is the
label data (see [`materialize_batch`](@ref)), and `state` is the next batch
iterator state.

Runs on the Client.
"""
function Base.take!(batcher::Batcher, state)
    # we first check the status future so that there was an error, it throws
    # immediately instead of blocking on `fetch` on the channel...
    #
    # we don't use the get_status convenience wrapper because we WANT to throw
    # the error, rather than just logging it and getting the exception itself.
    @debug "checking batcher status before waiting on channel"
    isready(batcher.status) && fetch(batcher.status)

    # wrap the rest of this in a try-catch to handle when batchers close the
    # channel on errors:
    try
        synchronize!(batcher, state)

        @debug "taking materialized batch and next state from channel"
        # next is probably ((xx, yy), state) but could be nothing to indicate that
        # batching is all done
        next, _ = Base.take!(batcher.channel)
        return next
        # TODO: consider allowing next to be a (::Future, state) tuple, to avoid
        # extra data movement in case of multi-worker batching (e.g., put
        # `remotecall` futures directly onto channel).  In that case, we'd need
        # to do: (can broadcase because of fallback `fetch(x::Any) = x`)
        # return fetch.(next)
    catch e
        @debug "caught exception: $e"
        if is_closed_ex(e)
            @debug "is a channel closed exception, getting batcher status..."
            # channel was closed by a worker (or otherwise).
            # figure out why, by synchronizing on the status
            ready = timedwait(() -> isready(batcher.status), 60)
            if ready == :ok
                # if start_batching threw, this will rethrow the
                # RemoteException.  In this case, because we're throwing from
                # inside a catch block, we'll see the whole exception stack,
                # with the RemoteException at the top, and the local exception
                # due to the channel closure at the bottom.
                status = fetch(batcher.status)
            else
                @warn "Waited 1 minute for batcher status to be ready after channel closed, continuing with status=:unknown"
                status = :unknown
            end
            
            @warn "batcher channel closed but batcher status did not throw: $(status)"
        end
        # if we made it through here without throwing a remote exception, then
        # we wanna rethrow the original exception that we caught here.
        rethrow()
    end
end

function synchronize!(batcher::Batcher, state)
    status = get_status(batcher)
    # need to also check that the channel is open, since iteration may be done
    # but batches remaining on the channel
    status == :running || isready(batcher.channel) ||
        throw(ArgumentError("status must be `:running` or channel ready to synchronize state (got $(status))"))
        
    @debug "fetching previous state to synchronize"
    _, prev_state = fetch(batcher.channel)
    if state != prev_state
        @warn("mismatch between requested batch state and Batcher state, restarting",
              state,
              prev_state)
        stop!(batcher)
        start!(batcher, state)
    end
    return batcher
end

# a same-process batcher with the same `take!` interface that `Batcher` uses in
# the training loop, for testing purposes
struct BlockingBatcher
    batches::Any
end

function Base.take!(batcher::BlockingBatcher, state)
    next = iterate_batch(batcher.batches, state)
    next === nothing && return nothing
    batch, state = next
    xx, yy = materialize_batch(batch)
    return (xx, yy), state
end
