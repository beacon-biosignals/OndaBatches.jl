module OndaBatches

using AlignedSpans
using AWSS3
using DataFrames
using Dates
using Distributed
using Legolas: Legolas, @schema, @version
using Onda: Onda, Samples, SignalV2, SamplesInfoV2
using StatsBase
using Tables
using TimeSpans

using Tables: rowmerge

include("utils.jl")

include("labeled_signal.jl")
export LabeledSignalV2, sub_label_span, label_signals, load_labeled_signal,
       store_labels, sort_and_trim_spans

include("materialize_batch.jl")
export BatchItemV2, materialize_batch_item, materialize_batch

include("iterate_batch.jl")
export RandomBatches, iterate_batch_item, iterate_batch

include("batch_services.jl")
export Batcher, start!, stop!, get_status

end
