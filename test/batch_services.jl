# need these because how the exception gets wrapped depends on julia
# version
unwrap(e) = e
unwrap(e::RemoteException) = unwrap(e.captured)
unwrap(e::CapturedException) = unwrap(e.ex)

@testset "Batcher" begin
    # provision some workers for service tests.
    worker_ids = addprocs(N_WORKERS)
    try
        provision_worker(worker_ids)

        init_batch_state = StableRNG(1338)
        batches = RandomBatches(; labeled_signals,
                                signal_weights=nothing,
                                label_weights=nothing,
                                n_channels=1,
                                batch_size=2,
                                batch_duration=Minute(5))
        
        localbatcher = OndaBatches.BlockingBatcher(batches)

        # state is required when start=true (the default)
        @test_throws ArgumentError Batcher(worker_ids[1:1], batches; start=true, state=nothing)
        @test_throws ArgumentError Batcher(worker_ids[1:1], batches; state=nothing)
        batcher = Batcher(worker_ids[1:1], batches; start=false)

        # test behavior without starting first
        @test get_status(batcher) == :stopped
        @test_throws ArgumentError take!(batcher, copy(init_batch_state))

        start!(batcher, copy(init_batch_state))
        @test get_status(batcher) == :running
        b, s1 = take!(batcher, copy(init_batch_state))

        multibatcher = Batcher(worker_ids, batches; state=copy(init_batch_state))

        @test get_status(batcher) == :running
        @test get_status(multibatcher) == :running

        @test take!(batcher, copy(init_batch_state)) == (b, s1)
        bm, sm1 = take!(multibatcher, copy(init_batch_state))
        bl, sl1 = take!(localbatcher, copy(init_batch_state))

        @test b == bm == bl
        @test s1 == sm1 == sl1

        # make sure it's repeatable:
        # warns that state is wrong, then info to stop, then info to start
        b0 = b
        b, s1 = @test_logs (:warn,) (:info,) (:info,) take!(batcher, copy(init_batch_state))
        bm, sm1 = @test_logs (:warn,) (:info,) (:info,) take!(multibatcher, copy(init_batch_state))
        bl, sl1 = take!(localbatcher, copy(init_batch_state))

        @test b == b0 == bm == bl
        @test s1 == sm1 == sl1

        b, s2 = @test_logs take!(batcher, s1)
        bm, sm2 = take!(multibatcher, sm1)
        bl, sl2 = take!(localbatcher, s1)

        @test b == bm == bl
        @test s2 == sm2 == sl2

        # take more batches to make sure they stay in sync for a few rounds
        let sm=sm2, s=s2
            for _ in 1:10
                bm, sm = take!(multibatcher, sm)
                b, s = take!(batcher, s)
                @test b == bm
                @test s == sm
            end
        end

        # start! on a running batcher is a no-op
        let chan=batcher.channel, status=batcher.status
            @test_throws MethodError start!(batcher)
            
            @test_logs (:warn,) start!(batcher, init_batch_state)
            @test chan === batcher.channel
            @test status === batcher.status

            @test_logs (:info,) stop!(batcher)
            @test_logs (:info,) start!(batcher, copy(init_batch_state))
            @test chan != batcher.channel
            @test status != batcher.status
        end

        # manual close stops batching
        close(batcher.channel)
        wait(batcher.status)
        @test get_status(batcher) == :closed

        @test_logs (:info,) stop!(batcher)
        # check that we can do this twice without error
        @test_logs (:info,) stop!(batcher)
        @test get_status(batcher) == :closed
        @test_logs (:info,) stop!(multibatcher)
        @test get_status(multibatcher) == :closed
        # test worker pool integrity after stop!
        pool = multibatcher.workers
        @test length(pool) == count_ready_workers(pool) == length(worker_ids) - 1

        @testset "start with empty worker pool" begin
            manager, rest = Iterators.peel(worker_ids)
            workers = WorkerPool(collect(rest))
            batcher = Batcher(manager, workers, batches; start=false)
            while isready(workers)
                take!(workers)
            end
            @test !isready(workers)
            start!(batcher, copy(init_batch_state))
            t = @async take!(batcher, copy(init_batch_state))
            # because this may deadlock, we put it behind a timedwait
            @test timedwait(() -> istaskdone(t), 5) == :ok

            # this, alas, may also deadlock (but should be fast):
            t = @async stop!(batcher)
            @test timedwait(() -> istaskdone(t), 1) == :ok
            @test get_status(batcher) == :closed
        end

        @testset "error handling" begin
            # need to do some juggling to test behavior when channel is MANUALLY
            # closed.  we're gonna splice in a future that resolve to `running` to
            # trick everybody...
            new_status = Future()
            put!(new_status, :running)
            batcher.status = new_status
            # ...and then close teh channel
            batcher.channel = RemoteChannel(() -> Channel{Any}(Inf))
            close(batcher.channel)
            # ...and then attempt to take!
            # 
            # warns about "okay"/not error status, and then throws channel closed
            @test_logs (:warn,) @test_throws InvalidStateException take!(batcher, copy(init_batch_state))

            badbatch = deepcopy(batches)
            # replace signals with nonsense paths that'll throw errors
            transform!(badbatch.labeled_signals,
                       :file_path => ByRow(_ -> "blahblah") => :file_path)

            badbatcher = Batcher(worker_ids[1:1], badbatch; state=copy(init_batch_state))
            @test_throws RemoteException wait(badbatcher.status)
            @test_throws RemoteException take!(badbatcher, copy(init_batch_state))
            @test (@test_logs (:error,) get_status(badbatcher)) isa RemoteException
            @test_logs (:info,) (:error,) stop!(badbatcher)
            # check that we can do this twice without error
            @test_logs (:info,) (:error,) stop!(badbatcher)

            badmultibatcher = Batcher(worker_ids, badbatch; state=copy(init_batch_state))
            @test_throws RemoteException wait(badmultibatcher.status)
            @test_throws RemoteException take!(badmultibatcher, copy(init_batch_state))
            @test (@test_logs (:error,) get_status(badmultibatcher)) isa RemoteException
            @test_logs (:info,) (:error,) stop!(badmultibatcher)
            # confirm that worker pool is restored to good state
            pool = badmultibatcher.workers
            @test length(pool) == count_ready_workers(pool) == length(worker_ids) - 1

            # test behavior of error on workers while waiting inside `take!`
            badbatcher = Batcher(worker_ids[1:1], badbatch; start=false)
            try
                start!(badbatcher, copy(init_batch_state))
                take!(badbatcher, copy(init_batch_state))
            catch e
                @test e isa RemoteException
                e = unwrap(e)
                @test e isa SystemError
                @test e.prefix == "opening file \"blahblah\""
            end
        end

        @testset "finite batches" begin
            # for testing, iterate batch items one at a time
            @everywhere begin
                using OndaBatches
                function OndaBatches.iterate_batch(batches::Vector{<:BatchItemV2}, state::Int)
                    state > length(batches) && return nothing
                    batch, next_state = batches[state], state + 1
                    # @show next_state
                    return [batch], next_state
                end
            end
            
            batches = map(eachrow(labeled_signals)) do row
                span = translate(TimeSpan(0, Minute(1)),
                                 start(row.label_span))
                item = sub_label_span(row, span)
                batch_channels = first(row.channels)
                return BatchItemV2(Tables.rowmerge(item, batch_channels))
            end

            # test multiple batcher methodologies against each other
            batchers = map((worker_ids, worker_ids[1:1])) do workers
                batcher = Batcher(workers, batches; start=true, state=1)
                return batcher
            end

            all_batches_rt = map(batchers) do batcher
                state = 1
                next = take!(batcher, state)
                batches_rt = []
                while next !== nothing
                    batch, state = next
                    # @show state
                    push!(batches_rt, batch)
                    next = take!(batcher, state)
                end

                return batches_rt
            end
            
            # need a vector of items to materialize, so we just call vcat on each
            @test all(==(materialize_batch.(vcat.(batches))), all_batches_rt)
            @test all(==(:done), get_status.(batchers))
        end

        @testset "recovery from dead worker" begin
            # we want to test what happens when a worker dies and needs to be
            # replaced...

            # first, make sure taht we CAN remove a worker and get reasonable
            # behavior...
            batches = RandomBatches(; labeled_signals,
                                    signal_weights=nothing,
                                    label_weights=nothing,
                                    n_channels=1,
                                    batch_size=2,
                                    batch_duration=Minute(5))

            manager, loaders = Iterators.peel(worker_ids)
            pool = WorkerPool(collect(loaders))
            batcher = Batcher(manager, pool, batches; state=copy(init_batch_state))
            wait(batcher.channel)
            rmprocs(last(collect(loaders)))

            caught_ex = let s=copy(init_batch_state)
                try
                    while true
                        b, s = take!(batcher, s)
                    end
                catch e
                    e
                end
            end

            @test unwrap(caught_ex) isa ProcessExitedException
            # we're left with one less worker in the pool
            @test length(pool) ==
                  count_ready_workers(pool) == 
                  length(collect(loaders)) - 1
        end
    finally
        rmprocs(worker_ids)
    end
end
