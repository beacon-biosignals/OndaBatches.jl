using Remark, FileWatching

while true
    Remark.slideshow(@__DIR__;
                     options = Dict("ratio" => "16:9"),
                     title = "OndaBatches.jl")
    @info "Rebuilt"
    FileWatching.watch_file(joinpath(@__DIR__, "src", "index.md"))
end
