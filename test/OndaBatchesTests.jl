module OndaBatchesTests

using AlignedSpans
using Aqua
using AWS
using AWSS3
using DataFrames
using Dates
using Distributed
using Legolas: Legolas, SchemaVersion, @schema, @version
using Onda
using OndaBatches
using Pkg
using StableRNGs
using StatsBase
using ReTest
using TimeSpans
using UUIDs

using Onda: SignalV2SchemaVersion
using OndaBatches: LabeledSignalV2SchemaVersion

using Tables: rowmerge

function isvalid(tbl, schema::SchemaVersion)
    tbl_schema = Tables.schema(Tables.columns(tbl))
    return Legolas.complies_with(tbl_schema, schema)
end

const VALID_STAGES = ("wake", "nrem1", "nrem2", "nrem3", "rem", "no_stage")
const SLEEP_STAGE_INDEX = Dict(s => UInt8(i)
                               for (i, s)
                               in enumerate(VALID_STAGES))

const TEST_ROOT = joinpath(S3Path("s3://beacon-public-oss/ondabatches-ci/tmp"),
                           string(uuid4()))

atexit() do
    (; bucket, key) = TEST_ROOT
    @sync for key in s3_list_keys(bucket, key * "/")
        for obj in s3_list_versions(bucket, key)
            version = obj["VersionId"]
            @async begin
                try
                    s3_delete(bucket, key; version)
                catch e
                    path = string(S3Path(bucket, key; version))
                    @error("Error deleting $(path)",
                           exception=(e, catch_backtrace()))
                end
            end
        end
    end
end

include("testdataset.jl")

const signals = DataFrame(Legolas.read(signals_path); copycols=true)
const uncompressed_signals = DataFrame(Legolas.read(uncompressed_signals_path); copycols=true)
const stages = DataFrame(Legolas.read(stages_path); copycols=true)

# this gets used all over the place so we'll just do it once here and avoid
# repetition
const labeled_signals = label_signals(uncompressed_signals,
                                      sort_and_trim_spans(stages, :recording; epoch=Second(30)),
                                      labels_column=:stage,
                                      encoding=SLEEP_STAGE_INDEX,
                                      epoch=Second(30))

const N_WORKERS = 3

# for testing get_channel_data
struct EvenOdds end
function OndaBatches.get_channel_data(samples::Samples, channels::EvenOdds)
    chans = samples.info.channels
    odds = @view(samples[chans[1:2:end], :]).data
    evens = @view(samples[chans[2:2:end], :]).data
    return cat(evens, odds; dims=3)
end

struct ZeroMissingChannels
    channels::Vector{String}
end
function OndaBatches.get_channel_data(samples::Samples, channels::ZeroMissingChannels)
    out = zeros(eltype(samples.data),
                length(channels.channels),
                size(samples.data, 2))
    for (i, c) in enumerate(channels.channels)
        if c ∈ samples.info.channels
            # XXX: this is extraordinarly inefficient and makes lots of copies,
            # it's just here for demonstration
            out[i, :] .= samples[c, :].data[1, :]
        end
    end
    return out
end

# utilities to get the number of workers in a pool that are actually available
function count_ready_workers(pool::AbstractWorkerPool)
    n = 0
    workers = []
    while isready(pool)
        # need to work around bugged behavior of `isread` vs `take!` blocking
        # when presented with a non-existent worker id
        id = @async take!(pool)
        timedwait(() -> istaskdone(id), 1) == :ok || continue
        push!(workers, fetch(id))
        n += 1
    end

    # replace the workers
    foreach(w -> put!(pool, w), workers)
    return n
end

# activate project + load dependencies on workers
function provision_worker(worker_ids)
    project = Pkg.project().path
    Distributed.remotecall_eval(Main, worker_ids,
                                :(using Pkg; Pkg.activate($(project))))
    if isdefined(Main, :Revise)
        Distributed.remotecall_eval(Main, worker_ids, :(using Revise))
    end
    Distributed.remotecall_eval(Main, worker_ids,
                                :(using OndaBatches, StableRNGs, ReTest))
    # not needed in CI but useful when running tests locally and doens't hurt
    Distributed.remotecall_eval(Main, worker_ids,
                                :(using AWS;
                                  global_aws_config($(global_aws_config()))))
    return nothing
end

@testset "aqua" begin
    Aqua.test_all(OndaBatches; ambiguities=false)
end
include("utils.jl")
include("labeled_signal.jl")
include("iterate_batch.jl")
include("materialize_batch.jl")
include("batch_services.jl")

end #module
