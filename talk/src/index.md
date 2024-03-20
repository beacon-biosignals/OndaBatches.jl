class: middle

.slide-title[

# OndaBatches.jl: Continuous, repeatable, and distributed batching

## Dave Kleinschmidt — Beacon Biosignals

### JuliaCon 2023 — [slide source + demo](https://github.com/beacon-biosignals/OndaBatches.jl/tree/main/talk/)
]

---

# Who am I?

Research Software Engineer at Beacon Biosignals

Our team builds tools for internal users at Beacon doing machine learning and
other quantitative/computational work

---

# Who are we?

Beacon Biosignals

> From its founding in 2019, Beacon Biosignals has engineered a machine learning
> platform designed to interrogate large EEG datasets at unprecedented speed and
> scale.

---

# Why did we make this?

Support common need to _build batches from annotated time series data_ across
multiple ML efforts at Beacon:

--

Multi-channel, regularly sampled time series data (i.e., EEG recordings)

Task is "image segmentation": output dense, regularly sampled labels (i.e.,
every 30s span gets a label)

--

Input data is Onda-formatted `Samples` + annotations (time span + label)

Models requires _numerical tensors_ for training/evaluation/inference

---

# Who is this for?

This might be interesting to you if you are

1. a ML engineer looking to model large time-series datasets and want to
   acutally _use_ OndaBatches to build your batches.
2. developing similar tools and are interested in how we build re-usable
   tools like this at Beacon.

--

## Why might you care?

1. We actually use this at Beacon!
2. It's a potentially useful example (cautionary tale?) for how to wrangle
   inconveniently large data and the nuances of distributed computing in a
   restricted domain

???

gonna be honest, mostly focusing on the second group here!

this is pretty specific to beacon's tooling and needs!  and there's a fair
amount of path dependence in how we got to this state...

---

# Outline

Part 1: Design, philosophy, and basic functionality

Part 2: Making a distributed batch loading system that doesn't require expertise
in distributed systems to use

---

# Design: Goals

Distributed (integrate with our distributed ML pipelines, throw more resources
at it to make sure data movement is not the bottleneck)

Scalable (handle out-of-core datasets, both for signal data and labels)

Deterministic + reproducible (pseudo-random)

Resumable

Flexible and extensible via normal Julia mechanisms of multiple dispatch

---

# Design: Philosophy

Separate the _cheap_ parts where _order matters_ ("batch specification") from
_expensive parts_ which can be done _asynchronously_ ("batch materialization")

--

Build on standard tooling (at Beacon), using
[Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl) to define
interface schemas which extend
[Onda.jl](https://github.com/beacon-biosignals/Onda.jl) schemas.

--

Use _iterator patterns_ to generate pseudorandom sequence of batch specs.

--

Be flexible enough that it can be broadly useful across different ML efforts at
Beaacon (and beyond??)

--

Use function calls we control to provide hooks for users to customize certain
behaviors via multiple dispatch (e.g., how to materialize `Samples` data into
batch tensor)

---

# How does it work?

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

# Extensibility

Some models require a specific set of channels to function (a "montage"), but
recordings don't always have all the required channels.

Here's a "channel selector" to fill in the missing channels with zeros:

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

# Extensibility

A very silly kind of "featurization": separate even and odd channels into
separate "compartments" (so they're processed independently in the model)

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

# Distributed batch loading: Why

different models have different demands on batch loading (data size,
amount of preprocessing required, etc.)

batch loading should _never_ be the bottleneck in our pipeline (GPU time is
expensive)

distributing batch loading means we can always "throw more compute" at it

???

(case study in lifting a serial workload into a distributed/async workload)

another thing: working around flakiness of multithreading and unacceptably low
throughput for S3 reads.  worker-to-worker communication has good enough
throughput

---

# Distributed batch loading: How

step 1: `return` → `RemoteChannel`

```julia
start_batching(channel, batches, state)
    try
        while true
            batch, state = iterate_batch(batches, state)
            xy = materialize_batch(batch)
            put!(channel, (xy, copy(state)))
        end
    catch e
        if is_channel_closed(e)
            @info "channel closed, stopping batcher..."
            return :closed
        else
            rethrow()
        end
    end
end

init_state = StableRNG(1)
# need a buffered channel in order for producers to stay ahead
channel = RemoteChannel(() -> Channel{Any}(10))
batch_worker = addprocs(1)
future = remotecall(start_batching!, batch_worker, batches, channel, init_state)
# now consumer can `take!(channel)` to retrieve batches when they're ready
```

???

the basic idea is that instead of calling a function `materialize_batch ∘
iterate_batch`, we will instead make a _service_ that feeds materialized batches
and the corresponding batcher states onto a `Distributed.RemoteChannel` where a
consumer can retrieve them.

of course, this still loads batches in serial, one at a time.  if we didn't care
about the order of the batches or reproducibility, we could just start multiple
independent feeder processes to feed the channel.

---

# Distributed batch loading: How

Step 2: Load multiple batches at the same time

Need to be careful to make sure the _order of batches_ is the same regardless of
the number of workers etc.

This is where the separation between batch _specification_ and batch
_materialization_ pays off: the specifications are small and cheap to
produce/serialize, so we can do them sequentially on the "manager" process.

```julia
function pmap_batches!(channel::RemoteChannel, spec, state, workers)
    futures_states = map(workers) do worker
        batch, state = iterate_batch(spec, state)
        batch_future = remotecall(materialize_batch, worker, batch)
        return batch_future, copy(state)
    end

    for (future, s) in futures_states
        xy = fetch(future)
        put!(channel, (xy, s))
    end

    return state
end
```

(Note this doesn't quite work when you have _finite_ series of batches)

???

cycle through the workers one at a time, feeding them a batch spec.

---

# Distributed batch loading: How

Step 2: Load multiple batches at the same time

```julia
function pmap_batches!(channel::RemoteChannel, spec, state, workers)
    # ...
end

function start_batching(channel::RemoteChannel, spec, state, workers)
    try
        while true
            state = pmap_batches!(channel, spec, state, workers)
        end
    catch e
        if is_closed_ex(e)
            @info "batch channel closed, batching stopped"
            return :closed
        else
            rethrow(e)
        end
    end
end
```

---

# Batching service

Lots of bookkeeping requried for this!
- `Future` returned by `remotecall(start_batching, ...)`
- `RemoteChannel` for serving the batches
- batch iterator itself

What happens when things go wrong??  it's very tricky to get errors to surface
properly and avoid bad states like slient deadlocks

We provide a `Batcher` struct that
- does the bookkeeping
- provides a limited API surface to reduce complexity for users...
- ...and manage complexity for developers/maintainers

---

# thanks!


