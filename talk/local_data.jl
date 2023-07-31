using AWSS3
using Legolas: @schema, @version
using Onda
using OndaBatches

include(joinpath(@__DIR__, "../test/testdataset.jl"))
local_root = joinpath(@__DIR__, "data")

local_signals_path = joinpath(local_root, "signals.arrow")
if !isfile(local_signals_path)
    signals = DataFrame(Legolas.read(uncompressed_signals_path); copycols=true)

    local_signals = transform(signals,
                              :file_path => ByRow() do path
                                  local_path = joinpath(local_root, "samples",
                                                        basename(path))
                                  cp(path, Path(local_path))
                                  @info string(path, '→', local_path)
                                  return local_path
                              end => :file_path)

    Onda.load(first(local_signals))

    Legolas.write(local_signals_path, local_signals, SignalV2SchemaVersion())
end

local_stages_path = joinpath(local_root, "stages.arrow")
if !isfile(local_stages_path)
    cp(stages_path, Path(local_stages_path))
    stages = DataFrame(Legolas.read(local_stages_path); copycols=true)
    stages = OndaBatches.sort_and_trim_spans(stages, :recording; epoch=Second(30))
    Legolas.write(local_stages_path, stages, SleepStageV1SchemaVersion())
end

function load_tables(; strip_refs=true)
    signals = DataFrame(Legolas.read(local_signals_path); copycols=true)
    if strip_refs
        transform!(signals,
                   :channels => ByRow() do channels
                       [string(first(split(c, "-"; limit=2))) for c in channels]
                   end => :channels)
    end
        
    stages = DataFrame(Legolas.read(local_stages_path); copycols=true)
    return signals, stages
end

