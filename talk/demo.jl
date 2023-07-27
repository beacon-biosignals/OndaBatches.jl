using DataFrames
using Dates
using Legolas
using Onda
using OndaBatches
using StableRNGs

#####
##### setup
##### 

include("local_data.jl")
const VALID_STAGES = ("wake", "nrem1", "nrem2", "nrem3", "rem", "no_stage")
const SLEEP_STAGE_INDEX = Dict(s => UInt8(i)
                               for (i, s)
                               in enumerate(VALID_STAGES))

#####
##### basic functionality
#####

signals, labels = load_tables(; strip_refs=true)
signals

samples = Onda.load(first(signals))

describe(signals, :eltype, :first)
describe(labels, :eltype, :first)

labeled_signals = label_signals(signals, labels, 
                                labels_column=:stage, 
                                encoding=SLEEP_STAGE_INDEX, 
                                epoch=Second(30))
describe(labeled_signals, :eltype)
labeled_signals.labels[1]

batches = RandomBatches(; labeled_signals,
                        # UNIFORM WEIGHTING OF SIGNALS + LABELS
                        SIGNAL_WEIGHTS=NOTHING,
                        LABEL_WEIGHTS=NOTHING,
                        N_CHANNELS=2,
                        BATCH_SIZE=3,
                        BATCH_DURATION=MINUTE(5))

state0 = StableRNG(1)

batch, state = iterate_batch(batches, deepcopy(state0))
describe(batch, :eltype, :first)

x, y = materialize_batch(batch);

# signal tensor:
x

# labels tensor:
y

#####
##### perist label sets
#####

labeled_signals_stored = store_labels(labeled_signals,
                                      joinpath(@__DIR__, "data", "labels"))

describe(labeled_signals_stored, :eltype, :first)
first(labeled_signals_stored.labels)

batches = RandomBatches(; labeled_signals=labeled_signals_stored,
                        # uniform weighting of signals + labels
                        signal_weights=nothing,
                        label_weights=nothing,
                        n_channels=2,
                        batch_size=3,
                        batch_duration=Minute(5))

state0 = StableRNG(1)

batch, state = iterate_batch(batches, deepcopy(state0))
describe(batch, :eltype, :first)

x, y = materialize_batch(batch);

x
y

#####
##### zero missing channels
#####

struct ZeroMissingChannels
    channels::Vector{String}
end
function OndaBatches.get_channel_data(samples::Samples, channels::ZeroMissingChannels)
    out = zeros(eltype(samples.data), length(channels.channels), size(samples.data, 2))
    for (i, c) in enumerate(channels.channels)
        if c ∈ samples.info.channels
            @views out[i:i, :] .= samples[c, :].data
        end
    end
    return out
end

channels = ZeroMissingChannels(["c3", "c4", "o1", "o2"])

OndaBatches.get_channel_data(samples, channels)

# normally we'd make our batch iterator set this field for us but for demo
# purposes we'll do it manually
batch.batch_channels .= Ref(channels);
x, y = materialize_batch(batch);

batch
x

#####
##### compartments
#####

struct EvenOdds
    n_channels::Int
end
function OndaBatches.get_channel_data(samples::Samples, channels::EvenOdds)
    n = channels.n_channels
    chans = samples.info.channels
    odds = @view(samples[chans[1:2:n], :]).data
    evens = @view(samples[chans[2:2:n], :]).data
    return cat(evens, odds; dims=3)
end

channels = EvenOdds(4)

labeled_signals_four_channels = filter(:channels => >=(4) ∘ length,
                                       labeled_signals)
batches_eo = RandomBatches(; labeled_signals=labeled_signals_four_channels,
                           # uniform weighting of signals + labels
                           signal_weights=nothing,
                           label_weights=nothing,
                           n_channels=2,
                           batch_size=3,
                           batch_duration=Minute(5))
                           
batch, _ = iterate_batch(batches_eo, copy(state0))
batch.batch_channels .= Ref(channels)
x, y = materialize_batch(batch)

x

batch_flat = deepcopy(batch)
batch_flat.batch_channels .= Ref(1:4)
x_flat, _ = materialize_batch(batch_flat)
x_flat
