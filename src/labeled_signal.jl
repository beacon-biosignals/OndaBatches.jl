"""
    LabeledSignal = Legolas.@row("labeled.signal@1" > "onda.signal@1",
                                 label_span::TimeSpan,
                                 labels::Union{Samples,Signal})

Type alias for a Legolas Row that represents one Onda signal with associated
labels.  Labels must be dense and contiguous, and are represented as
Onda.Samples or an Onda.Signal that referes to Onda.Samples serialized as LPCM.
`label_span` corresponds to the time span (relative to the recording) spanned by
the `labels`.

Note that the signal `span` and labels' `label_span` are both relative to the
start of the _recording_.
"""
const LabeledSignal = Legolas.@row("labeled.signal@1" > "onda.signal@1",
                                   label_span::TimeSpan,
                                   labels::Union{Samples,Signal})

# get the number of samples for a labeled signal row
_label_sample_count(row) = _label_sample_count(row.labels, row.label_span)
_label_sample_count(labels::Samples, _) = size(labels.data, 2)
_label_sample_count(labels::Signal, span) = sample_count(labels, duration(span))

_sample_rate(labels::Samples) = labels.info.sample_rate
_sample_rate(labels::Signal) = labels.sample_rate

"""
    store_labels(labeled_signals, root; format="lpcm")

Store labels to `root`, replacing the `Onda.Samples` in the `labels` column of
`labeled_signals` with `Onda.Signal`s.
"""
function store_labels(labeled_signals, root; format="lpcm")
    _mat = Tables.materializer(labeled_signals)
    rows = map(Tables.namedtupleiterator(labeled_signals)) do row
        return store_labels(LabeledSignal(row), root; format)
    end
    return _mat(rows)
end

_base_noversion(x::Any) = first(splitext(basename(x)))
_base_noversion(x::S3Path) = _base_noversion(string(S3Path(x.bucket, x.key)))

"""
    store_labels(labeled_signal::LabeledSignal, root; format="lpcm")

Store a single set of labels to `root`, replacing the `Onda.Samples` in the
`labels` column of `labeled_signals` with `Onda.Signal`s.  A single updated
`LabeledSignal` row is returned.

The filepath of the stored labels' Signal is the basename of
`labeled_signal.file_path` with `"labels_"` prepended.
"""
function store_labels(labeled_signal::LabeledSignal, root; format="lpcm")
    (; recording, labels, label_span, file_path, recording) = labeled_signal
    out_path = joinpath(root, "labels_" * _base_noversion(file_path))
    labels_signal = Onda.store(out_path, format, labels, recording, start(label_span))
    return rowmerge(labeled_signal; labels=Onda.Signal(labels_signal))
end

"""
    function load_labeled_signal(labeled_signal)

Load signal data as Onda.Samples from a labeled segment of an Onda.Signal (i.e.,
a [`LabeledSignal`](@ref) or row with schema `"labled.signal@1"`), and
returns the portion of the samples data corresponding to `labeled_signal.label_span`,
along with the corresponding labels (as another `Onda.Samples` object).

If possible, this will only retrieve the bytes corresponding to
`labeled_signal.label_span`.

Returns a `samples, labels` tuple.
"""
function load_labeled_signal(labeled_signal)
    # this fails on an individual row
    Legolas.validate((labeled_signal, ), Legolas.Schema("labeled.signal@1"))
    (; labels, label_span, span) = labeled_signal
    # we need to convert the label_span from relative to start of recording to
    # be relative to the loaded samples
    #
    # (---------------recording----------------------------------------)
    #         (--------span---------------------------)
    #                (----label_span-----)
    # -------> start(span)
    # --------------> start(label_span)
    #         ------> start(translate(label_span, -start(span)))
    # TODO: should we check that the label span is inside the signal span here
    # or on construction of the labeled span?  right now it's an error if the
    # labels start before or end after the actual signal span.
    label_span_relative_to_samples = translate(label_span, -start(span))
    samples = Onda.load(labeled_signal, label_span_relative_to_samples)

    # return labels as-is if they are Samples, and load/index appropriately if
    # they are a Lazy Signal
    labels = get_labels(labels, label_span)

    # XXX: #4 want to make sure that we're always getting the "right" number of
    # samples, so should use AlignedSpans here too probably
    return samples, labels
end

"""
    get_labels(labels::Samples, span)
    get_labels(labels::Signal, span)

Return labels as Samples, deserializing with `Onda.load` if necessary.  `span`
is the span _relative to the start of the recording_ that should be loaded.

This function is meant for internal use only; users should instead use
`load_labeled_signal` and `sub_label_span`.
"""
get_labels(labels::Samples, span) = labels
function get_labels(labels::Signal, span_relative_to_recording)
    # when labels are stored on disk, we can't eagerly sub-select them during
    # `sub_label_span`.  so we have to do the same juggling to translate the
    # label_span (here, span_relative_to_recording) to be relative to the
    # labels' Samples, and then load.
    span_relative_to_labels = translate(span_relative_to_recording,
                                        -start(labels.span))
    return Onda.load(labels, span_relative_to_labels)
end

"""
    label_signals(signals, annotations; groups=:recording, kwargs...)

Create a "labeled signals" table from a signals table and a table of annotations
containing labels.

Annotations will be passed to [`labels_to_samples_table`](@ref) and additional
kwargs are forwarded there as well.  Default values there are
- `labels_column` the column in the annotations table with labels,
- `epoch` the sampling period of the labels

Annotations must be
- contiguous and non-overlapping (withing `groups`)
- regularly sampled, with spans an even integer multiple of the `epoch` kwarg.

Returns a [`LabeledSignal`](@ref) table (e.g., with schema
`"labeled.signal@1"`), with labels in `:labels` and the signal spans occupied by
these labels in `:label_span`.  Like the signal `:span`, the `:label_span` is
relative to the start of the _recording_, not necessarily to the start of the
data represented by the _signal_.

If any label span is not entirely contained within the corresponding signal
span, this will throw an ArgumentError.
"""
function label_signals(signals, annotations; groups=:recording, kwargs...)
    labels_table = labels_to_samples_table(annotations; groups, kwargs...)
    joined = leftjoin(DataFrame(signals), labels_table; on=groups)
    if any(ismissing, joined.labels)
        missings = select(filter(:labels => ismissing, joined), groups)
        @warn "Dropping $(nrow(missings)) rows with no labels\n\n$(missings)"
        filter!(:labels => !ismissing, joined)
    end
    for (; recording, span, label_span) in eachrow(joined)
        if !TimeSpans.contains(span, label_span)
            e = "label span not contained in signal span for $(recording):\n" *
                "label span: $(label_span), signal span: $(span)"
            throw(ArgumentError(e))
        end
    end
    disallowmissing!(joined, [:labels, :label_span])
    return joined
end

"""
    sub_label_span(labeled_signal, new_label_span)

Select a sub-span of labeled signals `labeled_signal` (with schema
`"labeled.signal@1"`), returning a new labeled signal with updated `labels` and
`label_span`.

The `new_label_span` should be relative to the start of the recording (like the
signal's `span` and the current `label_span`).

"""
function sub_label_span(labeled_signal, new_label_span)
    (; labels, label_span) = labeled_signal
    if !TimeSpans.contains(label_span, new_label_span)
        throw(ArgumentError("""
            new labeled span is not contained within labeled span!
            input: $(new_label_span)
            currently labeled: $(label_span)
            """))
    end
    # new_label_span is relative to start of recording; align to start of
    # label_span
    #
    # (---------------recording----------------------------------------)
    #         (--------label-span---------------------------)
    #                (----new-label-span-----)
    # -------> start(label-span)
    # --------------> start(new-label-span)
    #         ------> start(translate(new_label_span, -start(label_span)))
    span = translate(new_label_span, -start(label_span))

    # This does not check that `span` aligns exactly with labels due to sample rounding and could
    # give bad results if they are misaligned.
    # TODO #4
    labels = _get_span(labels, span)
    label_span = new_label_span
    return Tables.rowmerge(labeled_signal; labels, label_span)
end

_get_span(samples::Samples, span) = samples[:, span]
# handle labels stored on disk/s3
_get_span(signal::Signal, span) = signal

#####
##### convert labels in spans to samples
#####

"""
    all_contiguous(spans)

Returns `true` if all `spans` are contiguous.  Assumes spans are sorted by start
time.
"""
function all_contiguous(spans)
    cur, rest = Iterators.peel(spans)
    for next in rest
        stop(cur) == start(next) || return false
        cur = next
    end
    return true
end

"""
    is_epoch_divisible(span::TimeSpan, epoch; roundto=nothing)

Tests whether `span` is evenly divisible into contiguous sub-spans of length
`epoch`, after optionally rounding to `roundto` (by default, no rounding is
performed).
"""
function is_epoch_divisible(span::TimeSpan, epoch; roundto=nothing)
    roundto = something(roundto, Nanosecond)
    dur = round(duration(span), roundto)
    return dur == floor(dur, epoch)
end

"""
    check_epoch_divisible(spans, epoch; roundto=Second)

Throw an `ArgumentError` if any of `spans` are not evenly divisible into
contiguous sub-spans of length `epoch`, according to
[`is_epoch_divisible`](@ref).
"""
function check_epoch_divisible(spans, epoch; roundto=nothing)
    all(is_epoch_divisible(span, epoch; roundto) for span in spans) ||
        throw(ArgumentError("spans are not evenly divisible into epochs!"))
    return nothing
end

function int_encode_labels(; epoch,
                           encoding::Dict,
                           roundto=nothing)
    return (stages, spans) -> int_encode_labels(stages, spans;
                                                epoch, encoding, roundto)
end

"""
    int_encode_labels(stages, spans; epoch, encoding::Dict,
                      roundto=nothing)
    int_encode_labels(; epoch, encoding, roundto)

Return a `Vector{UInt8}` of stage labels, using `encoding` to look up each stage
label in `stages`, sampled evenly at intervals of `epoch`.  `spans` are expanded
into non-overlapping, contiguous spans of duration `epoch`; `spans` must be
contiguous and with durations evenly divisible by `epoch`, except for the final
span which will be truncated.  `spans` durations will be rounded to the nearest
`roundto` (can be a `TimePeriod` subtype or instance, such as
`Millisecond(100)`, or `nothing`) before division into epochs to accommodate
minor errors in stage label durations; if `roundto=nothing` (the default) no
rounding will be performed.

The `Vector{UInt8}` of labels that is returned will have length
`floor(duration(shortest_timespan_containing(spans)), epoch)`

The `encoding` is used to map the values in `stages` to `UInt8`s, and should be
provided in the form of a `Dict{eltype(stages), UInt8}`.

`int_encode_labels(; epoch, encoding, roundto)` will return a closure which
captures the configuration options.
"""
function int_encode_labels(stages, spans;
                           epoch,
                           encoding,
                           roundto=nothing)
    issorted(spans; by=start) || throw(ArgumentError("spans must be sorted"))
    length(spans) == length(stages) ||
        throw(ArgumentError("mismatching lengths of spans ($(length(spans))) " *
                            "and stages ($(length(stages)))"))
    all_contiguous(spans) ||
        throw(ArgumentError("can only int encode contiguous label spans"))
    check_epoch_divisible(@view(spans[1:(end - 1)]), epoch; roundto)

    roundto = something(roundto, Nanosecond)
    # iterate through the spans/stages and undo the RLE
    labels = UInt8[]
    for (span, stage) in zip(spans, stages)
        # XXX: this may be necessary to "snap" some spans that all start/end at like 995ms.  it
        # may cause some very slight misalignment between the processed label
        # spans and the source, but by no more than 500 ms (and in practice,
        # more like 5ms) out of 30s (so ~1% max).
        #
        # note that we now DEFAULT to no rounding; this is still included to
        # preserve backwards compatibility with older versions
        dur = round(duration(span), roundto)
        n = Nanosecond(dur) ÷ Nanosecond(epoch)
        i = encoding[stage]
        for _ in 1:n
            push!(labels, i)
        end
    end
    return labels
end

floor_containing(; epoch) = spans -> floor_containing(spans; epoch)

"""
    floor_containing(spans; epoch)
    floor_containing(; epoch)

Compute the shortest timespan containing contiguous `spans`, rounded down to
the nearest multiple of `epoch`.

Note that this function will not check whether spans are contiguous.

The kwarg-only method returns a closure which captures the epoch.
"""
function floor_containing(spans; epoch)
    span = shortest_timespan_containing(spans)
    dur = floor(duration(span), epoch)
    return TimeSpan(start(span), start(span) + Nanosecond(dur))
end

"""
    labels_to_samples(labels::AbstractVector{UInt8}; epoch)
    labels_to_samples(; epoch)

Convert a vector of UInt8 stage labels sampled evenly at intervals of `epoch`
into `Onda.Samples` with samples rate of `1/epoch`.

The kwarg only form returns a closure that captures the `epoch`.

The returned samples have samples info:
```
SamplesInfo(; kind="label",
            channels=["label"],
            sample_unit="label",
            sample_resolution_in_unit=1,
            sample_offset_in_unit=0,
            sample_type=UInt8,
            sample_rate=Second(1) / epoch,
            sample_window=Nanosecond(epoch))
```
"""
labels_to_samples(; epoch) = x -> labels_to_samples(x; epoch)
function labels_to_samples(labels::AbstractVector{UInt8}; epoch)
    # XXX: include label levels (ref array) and other useful metadata (possibly
    # using a schema extension.
    info = SamplesInfo(; kind="label",
                       channels=["label"],
                       sample_unit="label",
                       sample_resolution_in_unit=1,
                       sample_offset_in_unit=0,
                       sample_type=UInt8,
                       sample_rate=Second(1) / epoch,
                       sample_window=Nanosecond(epoch))
    samples = Samples(reshape(labels, 1, :), info, false)
    return samples
end

"""
    labels_to_samples_table(labels::AbstractDataFrame; labels_column,
                            groups=:recording, epoch, kwargs...)

Convert annotations table into a table of labels as Samples.  This groups by
`groups` (defaults to `:recording`), and then applies
[`int_encode_labels`](@ref) to the `labels_column` and `:span` columns from each
group, and converts the resulting `UInt8` labels to `Onda.Samples` via
[`labels_to_samples`](@ref).  The sampling rate for the resulting labels is `1 /
epoch`.  The samples are returned in the `:labels` column.

Along with `epoch`, additional kwargs are forwarded to
[`int_encode_labels`](@ref):
- `encoding::Dict` the label -> `UInt8` mapping to use for encoding
- `roundto` controls rounding of "shaggy spans" (defaults to `nothing` for no
  rounding)

The `span` corresponding to these labels is determined by
[`floor_containing`](@ref) and returned in the `:label_span` column.

A `DataFrame` is returned with the `:labels` and `:label_span` per group, as
well as the `groups` variables.
"""
function labels_to_samples_table(stages::AbstractDataFrame; labels_column,
                                 groups=:recording, epoch, kwargs...)
    grouped = groupby(stages, groups)
    make_samples = labels_to_samples(; epoch) ∘ int_encode_labels(; epoch, kwargs...)
    return combine(grouped,
                   [labels_column, :span] => make_samples => :labels,
                   :span => floor_containing(; epoch) => :label_span)
end
