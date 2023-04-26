# In this code tour, we will outline some of the main functions of OndaBatches.jl with
# examples of how they might be used in machine learning project in a distributed environment.
# Before proceeding, we recommend completing the code tours of Legolas.jl and Onda.jl:
# https://github.com/beacon-biosignals/Legolas.jl/blob/main/examples/tour.jl
# https://github.com/beacon-biosignals/Onda.jl/blob/main/examples/tour.jl
#
# Why OndaBatches?
# We've seen how Onda.jl defines a set of interrelated, but distinct, Legolas.jl schemas for
# working with LPCM-encoded data serialised (usually) in Arrow format, namely: onda.signal
# and onda.annotation.
# In practice, however, these schema are usually extended further to encode information
# relevant to a particular use-case, e.g. defining a label component for an annotation.
#
# To effectively use these data in a machine learning context, where models are trained to,
# e.g., infer the annotation label for a given section of a signal, we need to have some
# tooling in place first:
# - We need to be able to associate existing labels to the corresponding signals to create a
#   hollistic dataset.
# - We need to be able to systematically construct batches of training / evaluation data
#   from this dataset while being flexible enough in our sampling mechanism so that we can
#   tailor the properties of the outputs.
# - Finally, we need to have this batching mechanism be carefully controlled by a scheduler
#   when working in a distributed environment.
#
# OndaBatches.jl aims to serve these needs, which we will explore in detail below.

# For this walkthrough we will use CleanSleepUtils.jl as a source of data:
# https://github.com/beacon-biosignals/data-map/tree/main/datasets/clean-sleep/CleanSleepUtils.jl
# However, you can use whichever source is convenient and accessible to you.
using CleanSleepUtils
using Dates
using Distributed
using Onda
using OndaBatches
using Random
using Test
using TimeSpans

# Encodings taken from SleepLab
const STAGE_ENCODING = Dict(s => UInt8(i) for (i, s) in enumerate(CleanSleepUtils.VALID_STAGES))
const RNG = MersenneTwister(1)

# Load the necessary signals and annotations tables
# Note: you'll need to use a role that has access to the clean-sleep S3 buckets.
signals = CleanSleepUtils.load_signals()
annotations = CleanSleepUtils.load_stages()

###
### LabeledSignals
###

# LabeledSignalV2 extends the SignalV2 schema to incorporate the labels of a given signal,
# which are represented as either another onda.signal or onda.sample, by marrying the
# underling signals and annotations across overlapping time spans.
# Note that the labels must be dense, contiguous and span the entire Signal.
# This constraint is a legacy of the origns of this code and may not be applicable to all
# use-cases.

# We will run into an OOM if we try to work with the full dataset.
# So let's work with a subset so that this can run locally
const N_TARGETS = 3
targets = rand(signals.recording, N_TARGETS)
signals = view(signals, findall(in(targets), signals.recording), :)
annotations = view(annotations, findall(in(targets), annotations.recording), :)

# This discards unnecessary fields such as :author_X, :source_X, and :display_icon_shortcode.
# Also removes :stage in favour of adding :labels, which get encoded as above and stored as
# onda.samples, and :label_span.
labeled_signals = label_signals(
    signals,
    annotations;
    labels_column=:stage,
    epoch=Second(1),
    encoding=STAGE_ENCODING,
    roundto=Second
)

@test size(labeled_signals, 1) == N_TARGETS  # one for each target recording
@test eltype(labeled_signals.labels) <: Onda.Samples
@test eltype(labeled_signals.label_span) <: Vector{TimeSpan}

# We can now load and inspect the underlying Samples data for one of our labeled signals.
# This is given as a tuple of Samples: one for the signal and the other the labels.
# See Onda.jl for working with Samples objects.
s1, l1 = load_labeled_signal(labeled_signals[1, :])

@test s1 isa Samples
@test l1 isa Samples

###
### RandomBatches
###

# Now that we are able to construct a holistic dataset, in which the labels for each signal
# have been assigned, we want to be able to prescribe and load batches of these data for
# training and evaluating a machine learning learning model.
#
# The RandomBatches type specifies how to iteratively sample from a collection of labeled
# signals to generate these batches in pseudo-random, iterable fashion.
#
# Specifically, RandomBatches constructs batches via the following mechanism:
# - randomly sample over the signals
# - for a given signal, randomly select a label
# - for a given label, randomly select a segment of the signal with that label
# - for a given segment, randomly select the number of required channels
#
# Optionally, we can also weight the sampling of the signals and labels.
#
# In this example we are specifying:
# - 1 channel per sample
# - 3 samples per batch
# - Each sample taken over a 2 second window
# - All signals and labels uniformly weighted
batches = RandomBatches(labeled_signals, nothing, nothing, 1, 3, Second(2))

# We can now draw down batches of samples from the RandomBatches
# Calling iterate_batch returns a batch item and the new state of the RNG after iterating once.
item, new_state = iterate_batch_item(batches, RNG)
@test item isa BatchItemV2
@test new_state isa AbstractRNG

# Here, we return and load a single batch item
# Note that the labels are not necessarily sampled at the same resolution as the signal.
# This is because XXX
x, y = materialize_batch_item(item)
@test size(x) == (1, 256)  # 1 channel, 2 seconds @ 128Hz resolution
@test size(y) == (1, 2)    # 1 label, 2 seconds @ 1Hz

# Additionally, we can draw down the entire batch at once
batch, new_state = iterate_batch(batches, RNG)
X, Y = materialize_batch(batch)
@test size(X) == (1, 256, 3)  # we now draw 3 samples and concatenate along the 3rd dimension
@test size(Y) == (1, 2, 3)

# Note that we can continue to draw as many batches as we like by repeatedly passing the
# new_state back to iterate_batch, much like Julia's Iterator interface
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration
# In this way, we can dynamically allocate and load batches of data depending on the needs
# of the model and infrastructure resources that are available.
batch, new_state = iterate_batch(batches, RNG)
X2, Y2 = materialize_batch(batch)
@test X2 != X
@test Y2 != Y

###
### Batcher
###

# Now that we have the mechanism for generating pseudo-random batches, we need to have these
# dynamically allocated and loaded in parallel across julia processes. This will enable us
# to scale our training process across a cluster.

# Let's start by adding a few processes and loading the necessary packages
addprocs(4)
@everywhere begin
    using Pkg
    Pkg.add("OndaBatches")
    using OndaBatches
end

# A Batcher governs the allocation of batch processing on a distributed environment.
# We'll provide the RandomBatcher defined above.
# Note that the "Batch Manager" cannot be assigned to the primary "Manager" node (1) because XXX
batcher = Batcher(2, [3, 4, 5], batches; start=false, state=RNG)

# First let's check the initialised batcher hasn't started
@test get_status(batcher) == :stopped
@test !isready(batcher.channel)

# Now let's start the batcher - but given we already used the RandomBatcher previously, we
# need to provide the correct new_state
# XXX: why does it work if we use state=RNG above? why do we need to do this twice?
start!(batcher, RNG)

# It should now be running and ready to allocated batches across nodes
@test get_status(batcher) == :running
@test isready(batcher.channel)

# X, Y are the collection of training / evaluation data above
(X3, Y3), new_state = take!(batcher, RNG)
@test size(X3) == size(X)
@test size(Y3) == size(Y)


# TODO: example of running this distributed

stop!(batcher)
@test get_status(batcher) == :stopped
