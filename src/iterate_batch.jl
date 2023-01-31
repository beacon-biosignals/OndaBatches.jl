"""
    RandomBatches

An iterator of pseudo-randomly sampled batches derived from a table of densely labeled signals (a [`labeled.signal@2`](@ref LabeledSignalV2) table).
Batches consist of `batch_size` "batch items".
A single batch item consists of `batch_duration * label_sample_rate` labels, and `batch_duration * signal_sample_rate` samples of multichannel EEG data.

Batch items are sampled according
to the following procedure:
1. A single labeled signal is sampled (optionally with weights)
2. A single label from that signal is sampled (optionally with weights)
3. One or more channels is selected, optionally randomly.

Each batch item is sampled independently, and in particular different batch items in a given batch can have different channels included (although the same number of them, `n_channels`).

The functions [`iterate_batch_item`](@ref) and [`iterate_batch`](@ref) sample a
single batch item and a full batch, respectively.

## Fields

- `labeled_signals::DataFrame`: the table of labeled signals that batches are
  sampled from.
- `signal_weights::AbstractWeights`: weights for individual signals (unweighted
  by default).  May be `nothing` duration construction, in which case unit 
  weights are created.
- `label_weights::Vector{AbstractWeights}`: weights for individual labels of
  each labeled signal (unweighted by default).  May be `nothing` during
  construction, in which case unit weights will be created for each labeled
  signal.
- `n_channels::Union{Nothing,Int}`: the number of channels each batch item
  should have; this many channels are sampled without replacement, unless 
  `n_channels === nothing` in which case all channels are included.
- `batch_size::Int`: the number of items that make one complete batch
- `batch_duration::TimePeriod`: the duration of the window for a single batch.
"""
Base.@kwdef struct RandomBatches
    labeled_signals::DataFrame
    signal_weights::AbstractWeights
    label_weights::Vector{AbstractWeights}
    n_channels::Union{Nothing,Int}
    batch_size::Int
    batch_duration::TimePeriod
    function RandomBatches(labeled_signals,
                           signal_weights,
                           label_weights,
                           n_channels,
                           batch_size,
                           batch_duration)
        Legolas.validate(Tables.schema(Tables.columns(labeled_signals)),
                         LabeledSignalV2SchemaVersion())

        signal_weights = something(signal_weights,
                                   uweights(nrow(labeled_signals)))
        length(signal_weights) == nrow(labeled_signals) ||
            throw(ArgumentError("mismatch between number of signals ($nrow(labeled_signals)) and weights ($(length(signal_weights)))"))

        label_lengths = _label_sample_count.(eachrow(labeled_signals))
        label_weights = @something(label_weights, uweights.(label_lengths))
        all(length.(label_weights) .== label_lengths) ||
            throw(ArgumentError("mismatch between length of label weights and labels"))

        return new(labeled_signals,
                   signal_weights,
                   label_weights,
                   n_channels,
                   batch_size,
                   batch_duration)
    end
end

"""
    iterate_batch(batches::Batches, rng)

Return a "batch listing" that can be materialized into model training/evaluation
input.

A batch is a table that has one row per batch item, and follows the
[`"batch-item@2"`](@ref BatchItemV2) schema.

This is consumed by a [`materialize_batch`](@ref) function that can be run on a
remote worker, so this sends just the minimum of information necessary to load
the batch signal data, the stage labels, and the spans that say how they line
up.
"""
function iterate_batch(batches::RandomBatches, rng)
    (; batch_size) = batches
    batch = DataFrame()
    for i in 1:batch_size
        row, rng = iterate_batch_item(batches, rng)
        push!(batch, NamedTuple(row); cols=:union)
    end
    return batch, rng
end

"""
    iterate_batch_item(batches::RandomBatches, rng)

Yields a single "batch item".  See documentation for [`RandomBatches`](@ref) for
the details on the sampling scheme.

Individual batch items are rows of a batch table with schema
[`"batch-item@2"`](@ref BatchItemV2), and are consumed by
[`materialize_batch_item`](@ref).
"""
function iterate_batch_item(batches::RandomBatches, rng)
    (; labeled_signals,
     batch_duration,
     label_weights,
     signal_weights,
     n_channels) = batches

    row_idx = sample(rng, 1:nrow(labeled_signals), signal_weights)
    signal_label_row = labeled_signals[row_idx, :]
    label_weights = label_weights[row_idx]

    (; labels, label_span, channels) = signal_label_row
    batch_label_span = sample_label_span(rng, labels, label_span,
                                         label_weights, batch_duration)

    # TODO: #5
    batch_channels = if n_channels === nothing
        channels
    else
        # sample n_channels without replacement
        sample(rng, channels, n_channels; replace=false)
    end

    batch_item = Tables.rowmerge(sub_label_span(signal_label_row,
                                                batch_label_span);
                                 batch_channels)
    return BatchItemV2(batch_item), rng
end

"""
    sample_label_span(rng, labels, label_span, labels_weight, batch_duration)

Return a TimeSpan sampled from labels.  First, an epoch is sampled according to
`labels_weight`.  Next, the position of this epoch in a window of
`batch_duration` is sampled with uniform probability, with the constraint that
the window must lie completely within `labels`.

The returned TimeSpan will have duration equal to `batch_duration` and will be
relative to the start of the _recording_.  The earliest possible return span
starts at `start(label_span)`, and the latest possible span stops at
`stop(label_span)`.
"""
function sample_label_span(rng, labels, label_span, labels_weight, batch_duration)
    Nanosecond(batch_duration) <= duration(label_span) ||
        throw(ArgumentError("requested span of $(batch_duration) is too long " *
                            "given labeled span $(label_span) " * 
                            "($(duration(label_span)))"))
    batch_seconds = Dates.value(Nanosecond(batch_duration)) / Dates.value(Nanosecond(Second(1)))
    sample_rate = _sample_rate(labels)
    batch_segments = batch_seconds * sample_rate
    isinteger(batch_segments) ||
        throw(ArgumentError("batch segments must be an integer, got " *
                            "$(batch_segments) with batch duration of " * 
                            "$(batch_duration) and sampling rate of " *
                            "$(sample_rate)"))
    batch_segments = round(Int, batch_segments)
    available_epochs = 1:_label_sample_count(labels, label_span)
    epoch = sample(rng, available_epochs, labels_weight)
    # now sample position of epoch within a window of length batch_segments
    # window can start anywhere from epoch 1 to end-batch_segments
    earliest_start = first(available_epochs)
    latest_start = last(available_epochs) - batch_segments + 1
    available_starts = earliest_start:latest_start
    # possible starts that include the sampled epoch
    epoch_starts = (epoch + 1 - batch_segments):epoch
    # sample from the intersection of these two to ensure we get something valid
    # reasonable
    epoch_start = sample(rng, intersect(available_starts, epoch_starts))
    # TimeSpans are right-open, so we need an _epoch_ range of batch_segments+1.
    # By using [epoch_start, epoch_start + batch_segments) as the epoch index
    # interval and calling `time_from_index` on the start/stop manually we make
    # sure that we get correct behavior even when `batch_segments` is 1.
    #
    # works around https://github.com/beacon-biosignals/TimeSpans.jl/issues/45
    epoch_stop = epoch_start + batch_segments
    # this is relative to `label_span`
    new_span = TimeSpan(time_from_index(sample_rate, epoch_start),
                        time_from_index(sample_rate, epoch_stop))
    # shift return span to be relative to _recording_, like `label_span`
    return translate(new_span, start(label_span))
end
