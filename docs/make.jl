using OndaBatches
using Documenter

makedocs(; modules=[OndaBatches],
         sitename="OndaBatches",
         authors="Beacon Biosignals and other contributors",
         pages=["Documentation" => "index.md"],
         strict=Documenter.except(:missing_docs),
         format=Documenter.HTML(; prettyurls=true, ansicolor=true))

deploydocs(; repo="github.com/beacon-biosignals/OndaBatches.jl.git",
           push_preview=true,
           devbranch="main")
