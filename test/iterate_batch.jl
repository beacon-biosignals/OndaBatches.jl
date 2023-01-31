lab_sigs = ("eager" => labeled_signals, "lazy" => OndaBatches.store_labels(labeled_signals, TEST_ROOT))
@testset "iterate batches: $type" for (type, labeled_signals) in lab_sigs
    using OndaBatches: get_labels, _sample_rate, _label_sample_count
    batch_spec = RandomBatches(; labeled_signals,
                               signal_weights=nothing, # uniform
                               label_weights=nothing,  # uniform
                               n_channels=1,
                               batch_size=10,
                               batch_duration=Minute(1))

    rng = StableRNG(1337)
    batch_row, rng = iterate_batch_item(batch_spec, rng)
    @test rng isa StableRNG
    @test all(in(propertynames(batch_row)),
              (:label_span, :batch_channels, :labels))
    @test isnothing(Legolas.validate([batch_row],
                                     Legolas.Schema("onda.signal@1")))
    @test isnothing(Legolas.validate([batch_row],
                                     Legolas.Schema("labeled.signal@1")))
    @test duration(batch_row.label_span) ==
          duration(get_labels(batch_row.labels, batch_row.label_span)) ==
          batch_spec.batch_duration

    batch_list, rng = iterate_batch(batch_spec, StableRNG(1337))
    @test rng isa StableRNG
    @test BatchItem(first(Tables.rows(batch_list))) == batch_row
    @test isnothing(Legolas.validate(batch_list,
                                     Legolas.Schema("onda.signal@1")))
    @test isnothing(Legolas.validate(batch_list,
                                     Legolas.Schema("labeled.signal@1")))
    @test length(Tables.rows(batch_list)) == batch_spec.batch_size
    # this depends on the RNG but is meant to check that different
    # recordings are being sampled, in contrast to below...
    @test length(unique(batch_list.recording)) > 1
    # we actually subsample channels
    @test all(batch_list.channels .!= batch_list.batch_channels)
    @test all(length.(batch_list.batch_channels) .== batch_spec.n_channels)

    @testset "channels sampled without replacement" begin
        batch_spec = RandomBatches(; labeled_signals,
                                   signal_weights=nothing, # uniform
                                   label_weights=nothing,  # uniform
                                   n_channels=2,
                                   batch_size=10,
                                   batch_duration=Minute(1))
        rng = StableRNG(1337)
        for _ in 1:100
            batch_list, rng = iterate_batch(batch_spec, rng)
            @test all(allunique.(batch_list.batch_channels))
        end
    end

    @testset "weights are used" begin
        # make sure weights are actually being used by zeroing out everything
        # except one:
        for i in 1:nrow(labeled_signals)
            allone_weights = Weights((1:nrow(labeled_signals)) .== i)
            batch_spec_one = RandomBatches(; labeled_signals,
                                           signal_weights=allone_weights,
                                           label_weights=nothing,  # uniform
                                           n_channels=1,
                                           batch_size=10,
                                           batch_duration=Minute(1))
            
            allone_batch_list, rng = iterate_batch(batch_spec_one,
                                                   StableRNG(1337))
            @test all(==(labeled_signals.recording[i]),
                      allone_batch_list.recording)
        end

        # make sure label weights are actually being used
        label_weights_allone = map(enumerate(eachrow(labeled_signals))) do (i, row)
            (; labels, label_span) = row
            labels = get_labels(labels, label_span)
            n_labels = size(labels.data, 2)
            just_i = Weights((1:n_labels) .== i)
            return just_i
        end
        
        for i in 1:nrow(labeled_signals)
            allone_weights = Weights((1:nrow(labeled_signals)) .== i)
            batch_spec_one = RandomBatches(; labeled_signals,
                                           signal_weights=allone_weights,
                                           label_weights=label_weights_allone,
                                           n_channels=1,
                                           batch_size=10,
                                           batch_duration=Second(30))
            batch_item, _ = iterate_batch_item(batch_spec_one, StableRNG(1337))
            @test duration(batch_item.label_span) ==
                duration(get_labels(batch_item.labels, batch_item.label_span)) == 
                batch_spec_one.batch_duration
            t = time_from_index(_sample_rate(labeled_signals.labels[i]), i)
            t += start(labeled_signals.label_span[i])
            span = TimeSpan(t, t + Nanosecond(Second(30)))
            @test TimeSpans.contains(batch_item.label_span, t)
            @test batch_item.label_span == span
        end
    end

    @testset "n_channels == nothing means use all channels" begin
        batchspec_all_chans = RandomBatches(; labeled_signals,
                                            signal_weights=nothing, # uniform
                                            label_weights=nothing,  # uniform
                                            n_channels=nothing,
                                            batch_size=10,
                                            batch_duration=Minute(1))
        rng = StableRNG(1337)
        for _ in 1:100
            batch, rng = iterate_batch_item(batchspec_all_chans, rng)
            @test batch.batch_channels == batch.channels
        end
    end

    @testset "sample_label_span edge cases" begin
        using OndaBatches: sample_label_span

        (; label_span, labels) = first(labeled_signals)
        labels_weights = uweights(_label_sample_count(labels, label_span))

        rng = StableRNG(1337)

        whole_span = sample_label_span(rng, labels, label_span, labels_weights,
                                       duration(label_span))
        @test whole_span == label_span

        shifted = sample_label_span(rng,
                                    labels,
                                    translate(label_span, Minute(1)),
                                    labels_weights,
                                    duration(label_span))
        @test shifted == translate(label_span, Minute(1))

        # choose weights so only first span is valid:
        label_sample_idxs = 1:_label_sample_count(labels, label_span)
        first_only_ws = Weights(label_sample_idxs .== 1)
        first_span = sample_label_span(rng, labels, label_span, first_only_ws,
                                       Minute(1))
        @test first_span == translate(TimeSpan(0, Minute(1)), start(label_span))

        # choose weights so only last span is valid:
        last_only_ws = Weights(reverse(first_only_ws))
        last_span = sample_label_span(rng, labels, label_span, last_only_ws,
                                      Minute(1))
        @test stop(last_span) == stop(label_span)
        @test duration(last_span) == Minute(1)

        # sample a span that is a single label
        first_one = sample_label_span(rng, labels, label_span, first_only_ws,
                                      Second(30))
        @test duration(first_one) == Second(30)
        @test start(first_one) == start(label_span)

        last_one = sample_label_span(rng, labels, label_span, last_only_ws,
                                      Second(30))
        @test duration(last_one) == Second(30)
        @test stop(last_one) == stop(label_span)

        # span too short
        @test_throws(ArgumentError("batch segments must be an integer, got 0.03333333333333333 with batch duration of 1 second and sampling rate of 0.03333333333333333"),
                     sample_label_span(rng, labels, label_span, labels_weights,
                                       Second(1)))

        # span not even multiple of epoch
        @test_throws(ArgumentError("batch segments must be an integer, got 1.0333333333333332 with batch duration of 31 seconds and sampling rate of 0.03333333333333333"),
                     sample_label_span(rng, labels, label_span, labels_weights,
                                       Second(31)))

        # span too long
        @test_throws(ArgumentError,
                     sample_label_span(rng, labels, label_span, labels_weights,
                                       duration(label_span) + Nanosecond(Second(30))))

        # empty span
        @test_throws(ArgumentError,
                     sample_label_span(rng, labels, label_span, labels_weights,
                                       Second(0)))

    end
    
end
