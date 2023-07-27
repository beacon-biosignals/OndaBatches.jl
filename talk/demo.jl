using OndaBatches
using Onda

include("local_copy.jl")

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
