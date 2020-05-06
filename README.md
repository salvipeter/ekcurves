# ϵ-κ-curves

This is a testbed for a generalization of κ-curves, called extended- or ϵ-κ-curves.

## Installation

First download and install [Julia](https://julialang.org/).
Then in the Julia shell:

```shell
julia> using Pkg;
julia> Pkg.add("Polynomials");
julia> Pkg.add("Gtk");
julia> Pkg.add("Graphics");
```

The above steps may take some time to finish.

Also download and extract this repository.

## Usage

In the Julia shell:

```shell
julia> cd("c:/path/to/ekcurves");
julia> include("ekcurves.jl");
julia> EKCurves.run()
```
(You can skip the first step if you start Julia from the `ekcurves` directory.)

The GUI is quite self-explanatory.

(Note that since Julia uses Just-In-Time compilation,
the program may be slow when you start to use it, but then it will speed up..)
