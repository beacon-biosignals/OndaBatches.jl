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

# For this walkthrough we will use testdataset.jl script as a source of data.
# You may not have access to this data so use whichever source is convenient if you want to
# work through this code interactively.
using DataFrames
using Dates
using Distributed
using Legolas
using OndaBatches
using Onda
using Random
using Test
using TimeSpans

const VALID_STAGES = ("wake", "nrem1", "nrem2", "nrem3", "rem", "no_stage")
const SLEEP_STAGE_INDEX = Dict(s => UInt8(i)
                               for (i, s)
                               in
                               enumerate(VALID_STAGES))

# Load the necessary signals and annotations tables - you may have to edit this file path
# depending where your instantiated project directory is.
include("test/testdataset.jl")

signals = DataFrame(Legolas.read(uncompressed_signals_path))
annotations = DataFrame(Legolas.read(stages_path))

###
### LabeledSignals
###

# LabeledSignalV2 extends the SignalV2 schema to incorporate the labels of a given signal,
# which are represented as either another onda.signal or onda.sample, by marrying the
# underling signals and annotations across overlapping time spans.
# Note that the labels must be dense and contiguous.
# This constraint is a legacy of the origns of this code and may not be applicable to all
# use-cases.

# Preprocess the annotations table
annotations = sort_and_trim_spans(annotations, :recording; epoch=Second(30))

# This discards unnecessary fields such as :author_X, :source_X, and :display_icon_shortcode.
# Also removes :stage in favour of adding :labels, which get encoded as above and stored as
# onda.samples, and :label_span.
labeled_signals = label_signals(signals,
    annotations,
    labels_column=:stage,
    encoding=SLEEP_STAGE_INDEX,
    epoch=Second(30))

@test eltype(labeled_signals.labels) <: Onda.Samples
@test eltype(labeled_signals.label_span) <: TimeSpan

# We can now load and inspect the underlying Samples data for one of our labeled signals.
# This is given as a tuple of Samples: one for the signal and the other the labels.
# See Onda.jl for working with Samples objects.
ls1 = LabeledSignalV2(labeled_signals[1, :])
s1, l1 = load_labeled_signal(ls1)

@test s1 isa Samples
@test l1 isa Samples

# Of note is that a LabeledSignal describes two time spans, both of which are described
# relative to the _start of the recording_:
# - `span`: describing the time span of the signal
# - `label_span`: describing the time span of the labels
#
# Note that these spans are not necessarily equal however the signal `span` must entirely
# encapsulate the `label_span`.

@test overlaps(ls1.span, ls1.label_span)

# It is also possible to select a sub-span of the LabeledSignal.
# Here we extract a sub-span that starts at 2hrs into the recording and ends at 3hrs.
sub_span = TimeSpan(Minute(120), Minute(180))
ls2 = LabeledSignalV2(sub_label_span(ls1, sub_span))
s2, l2 = load_labeled_signal(ls2)

# XXX: is this a bug? shouldn't signal span also be == sub_span?
# the size of the data below seems to suggest it should?
@test ls2.span == ls1.span
@test ls2.label_span == sub_span

@test size(s1.data, 2) > size(s2.data, 2)
@test size(l1.data, 2) > size(l2.data, 2)


###
### RandomBatches
###

# Now that we are able to construct a holistic dataset, in which the labels for each signal
# have been assigned, we want to be able to prescribe and load batches of these data for
# training and evaluating a machine learning learning model.
#
# The RandomBatches type specifies one possible iteration scheme for sampling from a
# collection of labeled signals to generate these batches in pseudo-random fashion.
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
# - Each sample taken over a 60 second window
# - All signals and labels uniformly weighted
batches = RandomBatches(labeled_signals, nothing, nothing, 1, 3, Second(60))

# We can now draw down batches of samples from the RandomBatches
# Calling iterate_batch returns a batch item and the new state after iterating once.
init_state = MersenneTwister(1)
item, new_state = iterate_batch_item(batches, init_state)
@test item isa BatchItemV2
@test new_state isa AbstractRNG

# Here, we return and load a single batch item
# Note that the labels are not necessarily sampled at the same resolution as the signal.
# This is because the labels are sampled at 0.033 Hz (1 every 30 seconds) while the signal
# is sampled at 128 Hz.
x, y = materialize_batch_item(item)
@test size(x) == (1, 7680)  # 1 channel, 60 seconds @ 128Hz resolution
@test size(y) == (1, 2)    # 2 labels, one for each 30 second segment

# Additionally, we can draw down the entire batch at once
init_state = MersenneTwister(1)
batch, new_state = iterate_batch(batches, init_state)
X, Y = materialize_batch(batch)
@test size(X) == (1, 7680, 3)  # we now draw 3 samples and concatenate along the 3rd dimension
@test size(Y) == (1, 2, 3)

# Since we provided the same initial state - the first items in X and Y are x and y above.
@test X[:, :, 1] == x
@test Y[:, :, 1] == y

# Note that we can continue to draw as many batches as we like by repeatedly passing the
# new_state back to iterate_batch, much like Julia's Iterator interface
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration
# In this way, we can dynamically allocate and load batches of data depending on the needs
# of the model and infrastructure resources that are available.
batch, new_state = iterate_batch(batches, new_state)
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
batch_workers = workers()
batch_manager = popfirst!(batch_workers) # https://github.com/beacon-biosignals/OndaBatches.jl/issues/25
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    using OndaBatches
end

# A Batcher governs the allocation of batch processing on a distributed environment.
# We'll provide the RandomBatcher defined above.
batcher = Batcher(batch_manager, batch_workers, batches; start=false)

# First let's check the initialised batcher hasn't started
@test get_status(batcher) == :stopped

# Now let's start the batcher with a fresh initial state
init_state = MersenneTwister(1)
start!(batcher, init_state)

# It should now be running and ready to allocated batches across nodes
@test get_status(batcher) == :running

# X3, Y3 are the same batches we sampled above.
# Similarly, we can keep sampling from this by repeatedly passing in the new_state
(X3, Y3), new_state = take!(batcher, init_state)
@test X3 == X
@test Y3 == Y

stop!(batcher)
@test get_status(batcher) == :closed

###
### A Realistic Example
###

# In practice, a batcher is launched from a process performing some computationally heavy
# task, such as training a neural network. Meanwhile, the batch_manager is run on a
# lightweight process that simply constructs the batches and allocates them to high-RAM
# batch_workers who materialize and store the associated data.
# The training worker then `takes!` these batches in sequence and updates the model based on
# the data + labels.
# In summary: a typical architecture for using this package might involve:
# - 1 root process
# - 1 train_worker GPU
# - N batch_worker CPUs
# - 1 batch_manager CPU

# spawn the training worker and allocate any resources it needs to train the model
train_worker = only(addprocs(1))
@everywhere train_worker begin
    using OndaBatches
    using Random
end

# Restart the batcher...
init_state = MersenneTwister(1)
start!(batcher, init_state)

# Issue a remotecall to `train_worker` which will serialize the code inside the do-block.
# In practice, this would be training a model but as a MWE we'll instead just compute a
# statistic on the batch results.
model_fut = remotecall(train_worker, batcher) do batcher
    results = Float64[]  # placeholder for some initialized model
    batch_state = init_state
    for batch_i in 1:10
        (x, y), batch_state = take!(batcher, batch_state)
        # Here we would implement a function that trains the model based on data + labels.
        # Instead we just push to the results vector to simulate it.
        push!(results, sum(x))
    end
    return results
end

# We can now fetch the result of the Future returned by the remotecall
model = fetch(model_fut)
@test model isa Vector{Float64}

stop!(batcher)
