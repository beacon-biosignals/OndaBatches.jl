@testset "materialize batches" for labed_sigs in (labeled_signals, OndaBatches.store_labels(labeled_signals, TEST_ROOT))
    # create batches by pulling one minute from each labeled signal
    batches = map(enumerate(Tables.rows(labed_sigs))) do (i, labeled_signal)
        (; label_span, channels) = labeled_signal
        one_minute = TimeSpan(Minute(i), Minute(i + 1))
        batch_span = translate(one_minute, start(label_span))
        batch_item = sub_label_span(labeled_signal, batch_span)
        batch_channels = mod1.(i:i + 1, length(batch_item.channels))
        return Tables.rowmerge(batch_item; batch_channels)
    end

    xys = materialize_batch_item.(batches)
    x, y = materialize_batch(batches)

    # consistency with the single item form
    @test x == cat(first.(xys)...; dims=3)
    @test y == cat(last.(xys)...; dims=3)

    @test size(x) == (2,                                    # channel
                      Dates.value(Second(Minute(1))) * 128, # time
                      nrow(labed_sigs))                     # batch

    @test size(y) == (1,                     # channel (just one for labels)
                      2,                     # 30s epochs in 1 minute
                      nrow(labed_sigs))      # batch

    # check consistency with manually pulled batches
    for i in 1:size(x, 3)
        labeled_signal = labed_sigs[i, :]
        (; span, label_span, sample_rate, channels, labels) = labeled_signal
        labels = OndaBatches.get_labels(labels, label_span)
        # span relative to start of labels
        batch_label_span = TimeSpan(Minute(i), Minute(i + 1))
        # span relative to start of signals
        batch_span = translate(batch_label_span, start(label_span) - start(span))
        batch_span = AlignedSpan(sample_rate, batch_span,
                                 ConstantSamplesRoundingMode(RoundDown))
        samples = Onda.load(labeled_signal, batch_span)
        chans = channels[mod1.(i:i + 1, length(channels))]
        @test samples[chans, :].data == x[:, :, i]
        @test labels[:, batch_label_span].data == y[:, :, i]
    end

    @testset "batch_channels missing" begin
        batches_all_chans = map(enumerate(Tables.rows(labed_sigs))) do (i, labeled_signal)
            (; label_span, channels) = labeled_signal
            one_minute = TimeSpan(Minute(i), Minute(i + 1))
            batch_span = translate(one_minute, start(label_span))
            batch_item = sub_label_span(labeled_signal, batch_span)
            return Tables.rowmerge(batch_item;
                                   batch_channels=missing)
        end

        xys_all_chans = materialize_batch_item.(batches_all_chans)
        xs_all = first.(xys_all_chans)
        @test size.(xs_all, 1) == length.(labed_sigs.channels)
    end

    @testset "get_channel_data" begin
        @testset "even-odd compartments" begin
            # we need to filter out the ones with an odd number of channels...
            for batch_item in eachrow(filter(:channels => iseven ∘ length,
                                             labed_sigs))
                label_span = translate(TimeSpan(0, Second(1)),
                                       start(batch_item.label_span))
                batch_eo = Tables.rowmerge(batch_item;
                                           batch_channels=EvenOdds(),
                                           label_span)
                x, y = materialize_batch_item(batch_eo)
                @test size(x) == (length(batch_item.channels) ÷ 2, 128, 2)
                chans = batch_item.channels
                chans_e = chans[2:2:end]
                chans_o = chans[1:2:end]
                samples, labels = load_labeled_signal(batch_eo)
                @test x[:, :, 1] == samples[chans_e, :].data
                @test x[:, :, 2] == samples[chans_o, :].data

                # throws method error because by default Onda tries to iterate
                # channels
                @test_throws MethodError samples[EvenOdds(), :]

                # manually construct even channel/odd channel batches and merge
                batch_e = Tables.rowmerge(batch_eo; batch_channels = chans_e)
                x_e, y_e = materialize_batch_item(batch_e)
                batch_o = Tables.rowmerge(batch_eo; batch_channels = chans_o)
                x_o, y_o = materialize_batch_item(batch_o)

                @test x == cat(x_e, x_o; dims=3)
                @test y == y_e == y_o
            end
        end

        @testset "zeros for missing channels loader" begin
            psg6 = ["c3-m2", "c4-m1", "f3-m2", "f4-m1", "o1-m2", "o2-m1"]
            zmc = ZeroMissingChannels(psg6)

            for batch_item in eachrow(labed_sigs)
                label_span = translate(TimeSpan(0, Second(1)),
                                       start(batch_item.label_span))
                batch_zmc = Tables.rowmerge(batch_item;
                                            batch_channels=zmc,
                                            label_span)
                samples, _ = load_labeled_signal(batch_zmc)
                x, y = materialize_batch_item(batch_zmc)
                @test size(x) == (length(zmc.channels), 128)
                for (i, c) in enumerate(zmc.channels)
                    if c ∈ batch_zmc.channels
                        @test x[i:i, :] == samples[c, :].data
                    else
                        @test all(iszero, x[i, :])
                    end
                end
            end
        end
    end
end
