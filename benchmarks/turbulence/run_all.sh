#!/usr/bin/env bash
# Reproduces the weakly turbulent flow comparison of time-stepping methods reported in the
# page "Time stepping for weakly turbulent flows". Run from this folder:  bash run_all.sh
# Takes several hours of CPU time (about 1.5 h of wall time on 4 cores). Each run is
# single-threaded; independent runs are executed 4 at a time.
set -euo pipefail
JL="julia --project=../.."
NU=4e-3; G=1.0; T0=1200; USTAR=20; T=1000

# 1. Warm-up with Tsit5 from (1/ν)A⁻¹g up to t = 3000, saving a checkpoint every 100 time units
$JL warmup.jl $NU $G 3000 | tee warmup.log

# 2. All methods from ω(t₀), for T time units each (method and parameter: σ or reltol)
cat > jobs.txt <<JOBS
RK4 0.5
ETDRK4 0.5
SSPRK33 0.5
CNAB2hand 0.25
Tsit5 1e-4
RK4 1.0
ETDRK4 1.0
SSPRK33 1.0
CNAB2hand 0.5
Tsit5 1e-3
RK4 2.0
ETDRK4 2.0
SSPRK33 2.0
CNAB2hand 0.75
Tsit5 1e-2
RK4 3.0
ETDRK4 3.0
SSPRK33 3.0
CNAB2hand 1.0
RK4 4.0
ETDRK4 4.0
SSPRK33 4.0
CNAB2hand 2.0
RK4 6.0
ETDRK4 6.0
SSPRK33 6.0
RK4 8.0
ETDRK4 8.0
JOBS
xargs -P 4 -L 1 sh -c "$JL run_method.jl $NU $G $T0 $USTAR \$0 \$1 $T 2>&1 | tail -1" < jobs.txt | tee scan.log

# 3. Cost per step of each method, run alone
$JL timing.jl $NU $G $T0 | tee timing.log

# 4. Tables and figures, written to figs/
GKSwstype=100 $JL analyze.jl figs
