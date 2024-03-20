@schema "batch-item" BatchItem

"""
    @version BatchItemV2{T} > LabeledSignalV2 begin
        batch_channels::T
    end

Legolas record type representing a single batch item.  Fields are inherited
from [`LabeledSignalV2 > SignalV2`](@ref LabeledSignalV2), and an additional `batch_channels`
field gives a channel selector for this batch.  A "channel selector" is anything
that can be used as a channel index for `Onda.Samples`, or `missing` (in which
case, all channels will be used in the order they occur in the `Samples`).

Columns include:

- columns from `Onda.SignalV2` (everything required to `Onda.load` the segment)
- `labels` and `label_span` from `LabeledSignalV2`
- `batch_channels`
"""
BatchItemV2

@version BatchItemV2 > LabeledSignalV2 begin
    batch_channels::(<:Any)
end

"""
    materialize_batch_item(batch_item, samples_eltype::Type=Float64)

Load the signal data for a single [`BatchItemV2`](@ref), selecting only the
channels specified in the `batch_channels` field (using all channels if the
field is `missing`).

Returns a `signal_data, label_data` tuple, which is the contents of the `data`
field of the signals and labels `Samples` objects returned by
`[load_labeled_signal`](@ref), after the signals data by `batch_channels`.

The eltype of `signal_data` will be `samples_eltype`; the eltype of `label_data`
is whatever is returned by [`get_labels`](@ref).
"""
function materialize_batch_item(batch_item, samples_eltype::Type=Float64)
    samples, labels = load_labeled_signal(batch_item, samples_eltype)
    batch_channels = coalesce(batch_item.batch_channels, samples.info.channels)
    signal_data = get_channel_data(samples, batch_channels)
    label_data = labels.data
    return signal_data, label_data
end

"""
    get_channel_data(samples, channels)

Get the data associated with the specified channels.  Default fallback simply
calls `samples[channels, :].data`.  But custom channel selectors can be used to
implement more exotic featurization schemes, (see tests for examples).
"""
get_channel_data(samples::Samples, channels) = samples[channels, :].data

"""
    materialize_batch(batch, samples_eltype::Type=Float64)

Materialize an entire batch, which is a table of [`BatchItemV2`](@ref) rows.  Each
row is materialized concurrently by [`materialize_batch_item`](@ref), and the
resulting signals and labels arrays are concatenated on dimension `ndims(x) + 1`
respectively.

Returns a `signal_data, label_data` tuple.  The dimensionality of these arrays
depends on the dimensionality of the results of
[`materialize_batch_item`](@ref), but will in general be `ndims(x) + 1`.

The eltype of `signal_data` will be `samples_eltype`; the eltype of `label_data`
is whatever is returned by [`get_labels`](@ref).
"""
function materialize_batch(batch, samples_eltype::Type=Float64)
    # TODO: check integrity of labeled_signals table for batch construction (are
    # the spans the same duratin/number of samples? etc.)
    signals_labels = asyncmap(item -> materialize_batch_item(item, samples_eltype),
                              Tables.rows(batch))
    signals, labels = first.(signals_labels), last.(signals_labels)

    x = _glue(signals)
    y = _glue(labels)

    return x, y
end
