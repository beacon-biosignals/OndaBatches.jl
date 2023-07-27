# motivation

build batches for ML

multi-channel time series data, "image segmentation"; target is regularly
sampled labels (i.e., every 30s span gets a label)

onda-formatted data `Samples`

transform into tensors that get fed into whichever ML model you want

"batches" (stack single sample tensors)

---

# why should you care? / who is this for?

you're a ML engineer looking to model large time-series datasets and want to
acutally _use_ OndaBatches to build your batches.

you're developing similar tools and are interested in how we build re-usable
tools like this at Beacon.

???

gonna be honest, mostly focusing on the second group here!

this is pretty specific to beacon's tooling and needs!  and there's a fair
amount of path dependence in how we got to this state...

---

# outline

Part 1: design, philosophy, and basic functionality

Part 2: making a distributed batch loading system that doesn't require expertise
in distributed systems to use

---

# design

separate the _cheap_ parts where _order matters_ ("batch specification") from
_expensive parts_ which can be done _asynchronously_ ("batch materialization")

--

use [Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl) to define
interfaces via schemas (which extend
[Onda.jl](https://github.com/beacon-biosignals/Onda.jl) schema for Signal)

--

use _iterator patterns_ to generate the sequences of batch specs.

--

use function calls we control to provide hooks for users to customize certain
behaviors via multiple dispatch (e.g., how to materialize `Samples` data into
batch tensor)

---

# implementation/examples

```julia
signals, labels = load_tables()
labeled_signals = label_signals(signals, labels, 
                                labels_column=:stage, 
                                encoding=SLEEP_STAGE_INDEX, 
                                epoch=Second(30))

batches = RandomBatches(; labeled_signals,
                        # uniform weighting of signals + labels
                        signal_weights=nothing,
                        label_weights=nothing,
                        n_channels=1,
                        batch_size=2,
                        batch_duration=Minute(5))

state0 = StableRNG(1)

batch, state = iterate_batch(batches, deepcopy(state0))
x, y = materialize_batch(batch)
```

???

live demo here...

---

# extensibility

make sure we always have data from teh same channels, even if they're not
present in the data (use zeros instead):

```julia
struct ZeroMissingChannels
    channels::Vector{String}
end

function OndaBatches.get_channel_data(samples::Samples, channels::ZeroMissingChannels)
    out = zeros(eltype(samples.data), length(channels.channels), size(samples.data, 2))
    for (i, c) in enumerate(channels.channels)
        if c ∈ samples.info.channels
            @views out[i:i, :] .= samples[c, :]
        end
    end
    return out
end
```

---

# extensibility

separate even and odd channels into separate "compartments" (so they're
processed independently in the model)

```julia
struct EvenOdds end

function OndaBatches.get_channel_data(samples::Samples, channels::EvenOdds)
    chans = samples.info.channels
    odds = @view(samples[chans[1:2:end], :]).data
    evens = @view(samples[chans[2:2:end], :]).data
    return cat(evens, odds; dims=3)
end
```

---

# distributed batch loading

why?  different models have different demands on batch loading (data size,
amount of preprocessing required, etc.)

batch loading should _never_ be the bottleneck in our pipeline (GPU time is
expensive)

distributing batch loading means we can always "throw more compute" at it

???

(case study in lifting a serial workload into a distributed/async workload)

another thing: working around flakiness of multithreading and unacceptably low
throughput for S3 reads.  worker-to-worker communication has good enough
throughput

# batching service


