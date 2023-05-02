# OndaBatches.jl

[![Build Status](https://github.com/beacon-biosignals/OndaBatches.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/beacon-biosignals/OndaBatches.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/beacon-biosignals/OndaBatches.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/beacon-biosignals/OndaBatches.jl)
[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://beacon-biosignals.github.io/OndaBatches.jl/stable)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://beacon-biosignals.github.io/OndaBatches.jl/dev)

[Take the tour!](https://github.com/beacon-biosignals/OndaBatches.jl/tree/master/examples/tour.jl)

OndaBatches.jl provides tooling to enable local and distributed batch loading of Onda-formatted datasets.
In particular, it defines utilites to:
- ...associate existing labels to the corresponding signals to create a hollistic dataset.
- ...systematically construct batches of training / evaluation data from this dataset while being flexible enough in our sampling mechanism so that we can tailor the properties of the outputs.
- ...initiate a batching mechanism, carefully controlled by a scheduler, when working in a distributed environment.
