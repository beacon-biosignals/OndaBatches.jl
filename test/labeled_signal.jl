@testset "labeled signal" begin
    @testset "Label preprocessing" begin
        using OndaBatches: all_contiguous

        @testset "is_contiguous and all_contiguous synth data" begin
            spans = [TimeSpan(Second(i), Second(i + 1)) for i in 1:10]
            spans_discont = spans[[1, 2, 4, 5, 7]]
            @test all_contiguous(spans)
            @test !all_contiguous(reverse(spans))
            @test !all_contiguous(spans_discont)
        end

        @testset "epoch divisibility" begin
            using OndaBatches: is_epoch_divisible, check_epoch_divisible
            epoch = Second(30)
            @test is_epoch_divisible(TimeSpan(0, Minute(10)), epoch)
            @test !is_epoch_divisible(TimeSpan(0, Second(31)), epoch)

            # no rounding by default
            @test !is_epoch_divisible(TimeSpan(0, Millisecond(30001)), epoch)
            # snapping to nearest second
            @test is_epoch_divisible(TimeSpan(0, Millisecond(30001)), epoch;
                                     roundto=Second)
            @test is_epoch_divisible(TimeSpan(0, Millisecond(30499)), epoch;
                                     roundto=Second)
            @test !is_epoch_divisible(TimeSpan(0, Millisecond(30500)), epoch;
                                      roundto=Second)
            @test !is_epoch_divisible(TimeSpan(0, Millisecond(30499)), epoch;
                                      roundto=Millisecond)
            # absurd rounding
            @test is_epoch_divisible(TimeSpan(0, Millisecond(30499)), epoch;
                                     roundto=Minute)

            good = TimeSpan(0, Second(30))
            bad = TimeSpan(0, Second(31))

            @test_throws ArgumentError check_epoch_divisible([bad], epoch)
            @test_throws ArgumentError check_epoch_divisible([bad, good], epoch)
            @test nothing === check_epoch_divisible([good], epoch)
            @test nothing === check_epoch_divisible([bad, good], Second(1))
        end

        @testset "snap containing" begin
            using OndaBatches: floor_containing
            epoch = Second(30)
            spans = [TimeSpan(epoch * i, epoch * (i + 1)) for i in 1:20]
            @test floor_containing(spans; epoch) == shortest_timespan_containing(spans)
            @test floor_containing(spans; epoch=Second(31)) ==
                  TimeSpan(Second(30), Second(30) + Second(31) * (length(spans) - 1))
            @test duration(floor_containing(spans; epoch=Second(31))) <
                  duration(floor_containing(spans; epoch))

            # non-congtiguous
            @test floor_containing(reverse(spans); epoch) == floor_containing(spans; epoch)
            @test floor_containing(spans[1:2:20]; epoch) == floor_containing(spans[1:19]; epoch)
        end

        @testset "int encode labels" begin
            using OndaBatches: int_encode_labels
            encoding = SLEEP_STAGE_INDEX
            stage_names = VALID_STAGES[1:5]
            stages = reduce(append!, (repeat([s], i) for (i, s) in enumerate(stage_names)))
            spans = [TimeSpan(Second(30 * i), Second(30 * (i + 1)))
                     for i in 1:length(stages)]

            labels = int_encode_labels(stages, spans;
                                       epoch=Second(30),
                                       encoding)
            @test labels == reduce(append!,
                                   (repeat(UInt8[i], i) for i in 1:length(stage_names)))

            labels = int_encode_labels(stages, spans; epoch=Second(10), encoding)
            @test labels == reduce(append!,
                                   (repeat(UInt8[i], i * 3) for i in 1:length(stage_names)))

            @test_throws ArgumentError int_encode_labels(stages, reverse(spans);
                                                         epoch=Second(30),
                                                         encoding)
            @test_throws ArgumentError int_encode_labels(stages[1:2:end], spans[1:2:end];
                                                         epoch=Second(30),
                                                         encoding)
            @test_throws ArgumentError int_encode_labels(stages[1:2], spans;
                                                         epoch=Second(30),
                                                         encoding)
            @test_throws ArgumentError int_encode_labels(stages, spans;
                                                         epoch=Second(31),
                                                         encoding)
            spans_short_first = [TimeSpan(Second(31), Second(60)); spans[2:end]]
            @test_throws ArgumentError int_encode_labels(stages, spans_short_first;
                                                         epoch=Second(30),
                                                         encoding)
            # ragged (last span is long) are truncated:
            spans_ragged = [spans[1:(end - 1)];
                            TimeSpan(start(spans[end]),
                                     stop(spans[end]) + Nanosecond(Second(2)))]
            @test int_encode_labels(stages, spans;
                                    epoch=Second(30),
                                    encoding) ==
                  int_encode_labels(stages, spans_ragged;
                                    epoch=Second(30),
                                    encoding)

            @test int_encode_labels(stages, spans; epoch=Second(10), encoding) ==
                  int_encode_labels(; epoch=Second(10), encoding)(stages, spans)
        end

        @testset "labels_to_samples_table" begin
            using OndaBatches: labels_to_samples_table
            # not sorted yet:
            @test_throws(ArgumentError("spans must be sorted"),
                         labels_to_samples_table(stages;
                                                 labels_column=:stage,
                                                 epoch=Second(30),
                                                 encoding=SLEEP_STAGE_INDEX))
            mystages = sort(stages, [:recording, order(:span; by=start)])

            labels = labels_to_samples_table(mystages;
                                             labels_column=:stage,
                                             epoch=Second(30),
                                             encoding=SLEEP_STAGE_INDEX)
            @test eltype(labels.labels) <: Samples
            @test eltype(labels.label_span) == TimeSpan


            labels_10s = labels_to_samples_table(mystages; epoch=Second(10),
                                                 labels_column=:stage,
                                                 encoding=SLEEP_STAGE_INDEX)
            foreach(labels.labels, labels_10s.labels) do labels, labels10
                # durations may be different if last span is long
                @test duration(labels) <= duration(labels10)
                @test duration(labels10) - duration(labels) < Second(30)
                # long final span can result in extra 10s span at the end
                labels_span = TimeSpan(0, duration(labels))
                @test all(labels.data .== labels10[:, labels_span].data[:, 1:3:end])
            end

            @test all(TimeSpans.contains.(labels_10s.label_span, labels.label_span))
        end
    end

    @testset "labeled signal" begin
        using OndaBatches: label_signals, get_labels
        sorted_stages = sort(stages, [:recording, order(:span; by=start)])
        labeled = label_signals(signals, sorted_stages;
                                labels_column=:stage,
                                encoding=SLEEP_STAGE_INDEX,
                                epoch=Second(30))

        @test isvalid(labeled, "labeled.signal@1")

        durations = combine(groupby(sorted_stages, :recording),
                            :span => (s -> sum(duration, s)) => :duration)
        leftjoin!(durations,
                  select(labeled,
                         :recording,
                         :label_span => ByRow(duration) => :label_duration),
                  on=:recording)
        @test all(floor.(durations.duration, Second(30)) .==
            round.(durations.label_duration, Second))

        for (r, (lab, stag)) in Legolas.gather(:recording, labeled, sorted_stages)
            lab = only(lab)
            (; labels, label_span) = lab
            (; stage, span) = stag
            foreach(stage, span) do s, sp
                # shift span to "label space"
                lab_sp = translate(sp, -start(label_span))
                unique_lab = only(unique(get_labels(labels, label_span)[1, lab_sp].data))
                @test VALID_STAGES[unique_lab] == s

                # sub_label_span takes a "new labeled span" (relative to the
                # recording start, so same as annotation spans)
                sub_lab = sub_label_span(lab, floor_containing([sp]; epoch=Second(30)))
                # these in general won't be exactly equal because the final
                # input span can be too long sometimes, and is truncated
                @test start(sp) == start(sub_lab.label_span)
                @test stop(sp) - stop(sub_lab.label_span) < Second(30)
                @test only(unique(sub_lab.labels.data)) == unique_lab
            end
        end

        # missing labels
        stages_missing = filter(:recording => !=(first(signals.recording)),
                                sorted_stages)
        labeled_missing = @test_logs (:warn, ) label_signals(signals,
                                                             stages_missing;
                                                             labels_column=:stage,
                                                             encoding=SLEEP_STAGE_INDEX,
                                                             epoch=Second(30))


        @test all(!=(first(signals.recording)), labeled_missing.recording)
        @test nrow(labeled_missing) == nrow(signals) - 1

        @testset "non-overlapping label spans are an error" begin
            onesignal = signals[1:1, :]
            badstage = [(; recording=onesignal.recording[1],
                         stage="wake",
                         id=uuid4(),
                         span=translate(onesignal.span[1], -Second(1)))]
            @test_throws ArgumentError label_signals(onesignal,
                                                     DataFrame(badstage);
                                                     labels_column=:stage,
                                                     encoding=SLEEP_STAGE_INDEX,
                                                     epoch=Second(30))
        end
    end

    @testset "`store_labels`" begin
        using OndaBatches: label_signals, store_labels
        sorted_stages = sort(stages, [:recording, order(:span; by=start)])
        new_labeled_signals = OndaBatches.store_labels(labeled_signals, TEST_ROOT)
        @test isvalid(new_labeled_signals, Legolas.Schema("labeled.signal@1"))
        @test all(isfile.(new_labeled_signals.file_path))
    end

    @testset "labeled signal (lazy)" begin
        using OndaBatches: label_signals, store_labels, get_labels
        sorted_stages = sort(stages, [:recording, order(:span; by=start)])
        labeled = label_signals(signals, sorted_stages;
                                labels_column=:stage,
                                encoding=SLEEP_STAGE_INDEX,
                                epoch=Second(30))

        labeled = store_labels(labeled, TEST_ROOT)


        @test isvalid(labeled, "labeled.signal@1")

        durations = combine(groupby(sorted_stages, :recording),
                            :span => (s -> sum(duration, s)) => :duration)
        leftjoin!(durations,
                  select(labeled,
                         :recording,
                         :label_span => ByRow(duration) => :label_duration),
                  on=:recording)
        @test all(floor.(durations.duration, Second(30)) .==
            round.(durations.label_duration, Second))

        for (r, (lab, stag)) in Legolas.gather(:recording, labeled, sorted_stages)
            lab = only(lab)
            (; labels, label_span) = lab
            (; stage, span) = stag
            foreach(stage, span) do s, sp
                # shift span to "label space"
                lab_sp = translate(sp, -start(label_span))
                unique_lab = only(unique(get_labels(labels, label_span)[1, lab_sp].data))
                @test VALID_STAGES[unique_lab] == s
                #
                # # sub_label_span takes a "new labeled span" (relative to the
                # # recording start, so same as annotation spans)
                sub_lab = sub_label_span(lab, floor_containing([sp]; epoch=Second(30)))
                # # these in general won't be exactly equal because the final
                # # input span can be too long sometimes, and is truncated
                @test start(sp) == start(sub_lab.label_span)
                @test stop(sp) - stop(sub_lab.label_span) < Second(30)
                @test only(unique(get_labels(sub_lab.labels, sub_lab.label_span).data)) == unique_lab
            end
        end

        # missing labels
        stages_missing = filter(:recording => !=(first(signals.recording)),
                                sorted_stages)
        labeled_missing = @test_logs (:warn, ) label_signals(signals,
                                                             stages_missing;
                                                             labels_column=:stage,
                                                             encoding=SLEEP_STAGE_INDEX,
                                                             epoch=Second(30))


        @test all(!=(first(signals.recording)), labeled_missing.recording)
        @test nrow(labeled_missing) == nrow(signals) - 1

        @testset "non-overlapping label spans are an error" begin
            onesignal = signals[1:1, :]
            badstage = [(; recording=onesignal.recording[1],
                         stage="wake",
                         id=uuid4(),
                         span=translate(onesignal.span[1], -Second(1)))]
            @test_throws ArgumentError label_signals(onesignal,
                                                     DataFrame(badstage);
                                                     labels_column=:stage,
                                                     encoding=SLEEP_STAGE_INDEX,
                                                     epoch=Second(30))
        end
    end

    @testset "`get_labels`" begin
        using OndaBatches: label_signals, store_labels, get_labels
        new_label_paths = joinpath.(TEST_ROOT,
                                    "labels_" .* first.(splitext.(basename.(signals.file_path))))
        new_labeled_signals = OndaBatches.store_labels(labeled_signals, TEST_ROOT)
        for (eager, lazy) in zip(eachrow(labeled_signals), eachrow(new_labeled_signals))
            labels = get_labels(lazy.labels, lazy.label_span)
            @test lazy.labels.sample_rate == labels.info.sample_rate
            @test lazy.labels.kind == labels.info.kind
            @test lazy.labels.channels == labels.info.channels
            @test lazy.labels.sample_unit == labels.info.sample_unit
            @test lazy.labels.sample_resolution_in_unit == labels.info.sample_resolution_in_unit
            @test lazy.labels.sample_offset_in_unit == labels.info.sample_offset_in_unit
            @test lazy.labels.sample_type == labels.info.sample_type

            @test labels.data == eager.labels.data
        end
    end

    @testset "load labeled signal" begin
        recording = uuid4()

        # generate a synthetic dataset of samples + annotations with one second
        # of 0, then 1, etc.; annotations are every 1s, samples are 128Hz.
        data = repeat(0:100; inner=128)
        samples = Samples(reshape(data, 1, :),
                          SamplesInfo(; kind="synthetic",
                                      channels=["synthetic"],
                                      sample_unit="none",
                                      sample_resolution_in_unit=1,
                                      sample_offset_in_unit=0,
                                      sample_type=Int,
                                      sample_rate=128),
                          false)
        signal_path = joinpath(TEST_ROOT, "$(recording).lpcm")
        signal = Onda.store(signal_path, "lpcm", samples, recording, 0)

        labels = [(; recording,
                   span=TimeSpan(Second(i), Second(i+1)),
                   value=i)
                  for i in 0:100]

        encoding = Dict(i => UInt8(i) for i in 0:128)

        labeled_signal = only(label_signals([signal], DataFrame(labels);
                                            groups=:recording,
                                            labels_column=:value,
                                            epoch=Second(1),
                                            encoding))

        # shfited versions of signal/labels that start at 00:00:10 of the
        # recording
        shifted_signal_path = joinpath(TEST_ROOT, "$(recording)-shifted.lpcm")
        shifted_signal = Onda.store(shifted_signal_path, "lpcm", samples,
                                    recording, Second(10))
        shifted_labels = [Tables.rowmerge(label;
                                          span=translate(label.span,
                                                         Second(10)))
                          for label in labels]

        @testset "aligned to start of recording" begin
            @test labeled_signal.span == labeled_signal.label_span
            samples_rt, _ = load_labeled_signal(labeled_signal)
            @test samples_rt.data == samples.data

            sub_lab_sig = sub_label_span(labeled_signal, TimeSpan(Second(10), Second(11)))
            sub_samples, _ = load_labeled_signal(sub_lab_sig)
            @test all(==(10), sub_samples.data)
        end

        @testset "all shifted by 10s" begin
            shifted_lab_sig = only(label_signals([shifted_signal], DataFrame(shifted_labels);
                                                 groups=:recording,
                                                 labels_column=:value,
                                                 epoch=Second(1),
                                                 encoding))

            shifted_samples, shifted_labs_rt = load_labeled_signal(shifted_lab_sig)
            @test shifted_labs_rt.data == shifted_lab_sig.labels.data

            shifted_sub_lab_sig = sub_label_span(shifted_lab_sig,
                                                 # translate 10-11
                                                 TimeSpan(Second(20), Second(21)))
            shifted_sub_samples, shifted_sub_labels = load_labeled_signal(shifted_sub_lab_sig)
            @test all(==(10), shifted_sub_samples.data)
            @test size(shifted_sub_samples.data) == (1, 128)
            @test all(==(10), shifted_sub_labels.data)
            @test size(shifted_sub_labels.data) == (1, 1)
        end

        @testset "only signals shifted by 10s" begin
            # errors because labels are starting at 00:00:00 but signal starts
            # at 00:00:10, so trying to load negative times from the signal
            # errors
            @test_throws(ArgumentError,
                         label_signals([shifted_signal], DataFrame(labels);
                                       groups=:recording,
                                       labels_column=:value,
                                       epoch=Second(1),
                                       encoding))

            # to test error path for load_labeled_signal, need to work around
            # the check in label_signals by manipulating the labeled signal
            # table directly
            shifted_sig_err = rowmerge(labeled_signal;
                                       span=translate(labeled_signal.span,
                                                      Second(10)))
            @test_throws ArgumentError load_labeled_signal(shifted_sig_err)

            # lop off first 10s of labels since there's no samples data for
            # them after shifting the signal span up by 10s
            labels_drop10 = filter(:span => >=(Second(10)) ∘ start,
                                   DataFrame(labels))
            shifted_sig = only(label_signals([shifted_signal],
                                             labels_drop10;
                                             groups=:recording,
                                             labels_column=:value,
                                             epoch=Second(1),
                                             encoding))
            shifted_samples, labs_rt = load_labeled_signal(shifted_sig)
            # we've lopped off 10s from the labels, so load 10s fewer samples
            @test duration(shifted_samples) ==
                  duration(labs_rt) ==
                  duration(samples) - Second(10)
            # labels start at 10 (they're not shifted)
            @test first(labs_rt.data) == 10
            # samples start at 0 since we've shifted them
            @test first(shifted_samples.data) == 0

            shifted_sub_sig = sub_label_span(shifted_sig,
                                             # translate 10-11
                                             TimeSpan(Second(20), Second(21)))
            shifted_sub_samples, shifted_sub_labels = load_labeled_signal(shifted_sub_sig)
            #
            @test all(==(10), shifted_sub_samples.data)
            @test size(shifted_sub_samples.data) == (1, 128)
            @test all(==(20), shifted_sub_labels.data)
            @test size(shifted_sub_labels.data) == (1, 1)
        end

        @testset "only labels shifted by 10s" begin
            # throws an error since there's 10s of extra labeled time after
            # shifting labels but not signals
            @test_throws(ArgumentError,
                         label_signals([signal], DataFrame(shifted_labels);
                                       groups=:recording,
                                       labels_column=:value,
                                       epoch=Second(1),
                                       encoding))
            # to test error path for load_labeled_signal, need to work around
            # the check in label_signals by manipulating the labeled signal
            # table directly
            shifted_lab_err = rowmerge(labeled_signal,
                                       label_span=translate(labeled_signal.label_span,
                                                            Second(10)))
            @test_throws ArgumentError load_labeled_signal(shifted_lab_err)

            # last label in original set is 100:101
            labels_drop10 = filter(:span => <=(Second(101)) ∘ stop,
                                   DataFrame(shifted_labels))
            shifted_lab = only(label_signals([signal], labels_drop10;
                                             groups=:recording,
                                             labels_column=:value,
                                             epoch=Second(1),
                                             encoding))
            samples_rt, shifted_labs_rt = load_labeled_signal(shifted_lab)
            @test duration(samples_rt) ==
                  duration(shifted_labs_rt) ==
                  duration(samples) - Nanosecond(Second(10))

            @test first(samples_rt.data) == 10
            @test first(shifted_labs_rt.data) == 0

            shifted_sub_sig = sub_label_span(shifted_lab,
                                             # translate 10-11
                                             TimeSpan(Second(20), Second(21)))
            shifted_sub_samples, shifted_sub_labels = load_labeled_signal(shifted_sub_sig)
            # signal samples still start at 0
            @test all(==(20), shifted_sub_samples.data)
            @test size(shifted_sub_samples.data) == (1, 128)
            # labels are shifted
            @test all(==(10), shifted_sub_labels.data)
            @test size(shifted_sub_labels.data) == (1, 1)
        end
    end
end
