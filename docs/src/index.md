```@meta
CurrentModule = OndaBatches
```

# API

## Batching service

```@docs
Batcher
Batcher(::Int, ::AbstractWorkerPool, ::Any; start::Any, state::Any, buffer::Any)
Base.take!(::Batcher, state)
start!
stop!
get_status
```

## Batch sampling

```@docs
BatchItem
RandomBatches
iterate_batch_item
iterate_batch
```

## Batch materialization

```@docs
materialize_batch_item
materialize_batch
get_channel_data
```

## Labeled signals

```@docs
LabeledSignal
sub_label_span
label_signals
load_labeled_signal
store_labels
```

## Internal utilities

!!! warning
    None of the following are meant to be called by users, are not part of the
    API for semantic versioning purposes, and can change at any time.

```@docs
labels_to_samples_table
labels_to_samples
get_labels
int_encode_labels
floor_containing
is_epoch_divisible
check_epoch_divisible
all_contiguous
sample_label_span
start_batching
_feed_jobs!
reset!
with_channel
```
