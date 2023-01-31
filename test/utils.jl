@testset "utilities" begin
    @testset "s3 access" begin
        # check that we can actually read/write from test sandbox
        hello_f = joinpath(TEST_ROOT, "hello.txt")
        write(hello_f, "hello, world!")
        @test String(read(hello_f)) == "hello, world!"
    end

    @testset "Onda.read_byte_range" begin
        using Dates
        using TimeSpans
        signal = first(eachrow(signals))
        signal_uncomp = only(filter(:recording => ==(signal.recording),
                                    uncompressed_signals))
        samples_all = Onda.load(signal_uncomp)
        onesec_span = TimeSpan(Second(100), Second(101))
        samples_onesec = Onda.load(signal_uncomp, onesec_span)
        @test samples_onesec.data == samples_all[:, onesec_span].data
        # not sure why this is broken...
        @test_broken samples_onesec.data isa Base.ReshapedArray

        bad_span = TimeSpan(stop(signal_uncomp.span) + Nanosecond(Second(1)),
                            stop(signal_uncomp.span) + Nanosecond(Second(2)))
        # this throws a BoundsError without our code (since Onda falls back to
        # loading EVERYTHING and then indexing.  with our utils, it passes the
        # byte range to AWS which says it's invalid
        @test_throws AWS.AWSException Onda.load(signal_uncomp, bad_span)

        ex = try
            Onda.load(signal_uncomp, bad_span)
        catch e
            e
        end
        @test ex isa AWS.AWSException
        @test ex.code == "InvalidRange"

        # does not hit this path for a compressed format
        samples_compress_onesec = Onda.load(signal, onesec_span)
        @test samples_compress_onesec.data == samples_onesec.data

        # why does this throw an inexact error?  something gets scrambled
        # inbetween the zst decompression and the deserialization code that
        # checks reshapes the array of bytes into a channel-by-sample matrix.
        # it looks like it's returning a single byte, which then gets divided by
        # 6 (number of channels) and tries to covnert that to an Int...
        @test_throws InexactError Onda.load(signal, bad_span)
    end

    @testset "_glue" begin
        using OndaBatches: _glue
        vecs = [rand(10) for _ in 1:11]
        @test _glue(vecs) == reduce(hcat, vecs) == hcat(vecs...)
        @test size(_glue(vecs)) == (10, 11)
        # still get added dimension with only a single element collection
        @test size(_glue(vecs[1:1])) == (10, 1)

        mats = [rand(3, 4) for _ in 1:11]
        @test _glue(mats) == cat(mats...; dims=3)
        @test size(_glue(mats)) == (3, 4, 11)

        bricks = [rand(3, 4, 5) for _ in 1:11]
        @test _glue(bricks) == cat(bricks...; dims=4)
        @test size(_glue(bricks)) == (3, 4, 5, 11)

        tup = Tuple(mats)
        @test _glue(tup) == _glue(mats)

        @test_throws(ArgumentError("all elements must have the same size, got >1 unique sizes: (1, 2), (2,)"),
                     _glue([[1 2], [1; 2]]))
    end

    @testset "with_channel" begin
        using OndaBatches: with_channel

        # normal return values are propagated:
        done = with_channel(Channel{Any}(1)) do channel
            put!(channel, 1)
            return :done
        end
        @test done == :done

        # channel closed errors are caught and return :closed
        # 
        # we need to do some shenanigans here to be able to test what happens when
        # the channel is closed by someone else.  so we create an async task that
        # waits on the channel...
        c = Channel{Any}(Inf)
        task = @async with_channel(c) do channel
            wait(channel)
        end
        # ...then we close it...
        close(c)
        # ...and grab the return value of `with_channel` from the task:
        @test fetch(task) == :closed

        # other errors are propagated as usual:
        c = Channel{Any}(Inf)
        @test_throws ErrorException("AHHHH") with_channel(c) do c
            error("AHHHH")
        end
    end

    @testset "_wait" begin
        using OndaBatches: _wait

        ws = addprocs(1)
        try
            provision_worker(ws)
            @show ws
            pool = WorkerPool(ws)
            w = take!(pool)

            # local call to _wait
            while isready(pool)
                take!(pool)
            end
            @test !isready(pool)
            t = @async _wait(pool)
            @test !istaskdone(t)
            put!(pool, w)
            # avoid race condition
            status = timedwait(() -> istaskdone(t), 10)
            @test status == :ok

            # remote call to _wait
            while isready(pool)
                take!(pool)
            end
            @test !isready(pool)
            f = remotecall(_wait, w, pool)
            @test !isready(f)

            # XXX: debugging test failure here...
            isready(f) && fetch(f)

            put!(pool, w)
            status = timedwait(() -> isready(f), 10)
            @test status == :ok
        finally
            rmprocs(ws)
        end
    end

    @testset "reset!" begin
        using OndaBatches: reset!
        ws = addprocs(2)
        try
            provision_worker(ws)

            pool = WorkerPool(ws)
            @test isready(pool)
            @test length(pool) == count_ready_workers(pool) == 2

            reset!(pool)
            @test isready(pool)
            @test length(pool) == count_ready_workers(pool) == 2

            w = take!(pool)
            # sorted:
            @test w == first(ws)
            reset!(pool)
            @test isready(pool)
            @test length(pool) == count_ready_workers(pool) == 2

            w = take!(pool)
            # sorted (again):
            @test w == first(ws)

            while isready(pool)
                take!(pool)
            end
            @test !isready(pool)
            reset!(pool)
            @test isready(pool)
            @test length(pool) == count_ready_workers(pool) == 2
            
            while isready(pool)
                take!(pool)
            end
            remotecall_fetch(reset!, first(ws), pool)
            @test isready(pool)
            @test length(pool) == count_ready_workers(pool) == 2

            rmprocs(first(ws))
            remotecall_fetch(reset!, last(ws), pool)
            @test isready(pool)
            # reset! drops dead workers
            @test length(pool) == count_ready_workers(pool) == 1

            rmprocs(take!(pool))
            reset!(pool)
            # no more workers in pool
            @test !isready(pool)
            @test length(pool) == count_ready_workers(pool) == 0
        finally
            rmprocs(ws)
        end
    end
end
