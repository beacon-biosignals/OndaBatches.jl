module OndaBatches

using AWSS3
using DataFrames
using Dates
using Distributed
using Legolas
using Onda
using StatsBase
using Tables
using TimeSpans

using Legolas: @row
using Tables: rowmerge

include("utils.jl")

include("labeled_signal.jl")
export LabeledSignal, sub_label_span, label_signals, load_labeled_signal,
       store_labels

include("materialize_batch.jl")
export BatchItem, materialize_batch_item, materialize_batch

include("iterate_batch.jl")
export RandomBatches, iterate_batch_item, iterate_batch

include("batch_services.jl")
export Batcher, start!, stop!, get_status

end
