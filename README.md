# Least cores in energy community games — reproduction

## Reproduction: "Least cores in energy community games" (arXiv 2511.05291), Section 7

**Claim tested:** the paper's computational result that **six approaches — row
generation (RG), the iterative scans IT / ITI / ITD, the compact program CQC, and
a cheap leave-one-out heuristic LCH — all compute the same least core value ε\***
of an Energy Sharing Game**, and that the leave-one-out "heuristic" matches the
exact ε\*.

**What was done.** Re-implemented the Energy Sharing Game (paper Eqs. 4–5, no
admission fees → balanced) and all six algorithms in Python + HiGHS (`reproduce/`),
on synthetic instances following the paper's Fioriti-et-al-2021 base configuration
(3 consumers + 7 prosumers), replicated to |U| = 10, 20, 50, 100 users.

**Assessment: level C — partial reproduction.** On our synthetic instances all six
methods return **ε\* = 0.4009218** (γ = 0.10) at every size, the leave-one-out
heuristic **LCH matches it exactly**, and at |U|=10 this agrees with an exhaustive
brute force over all 2¹⁰ coalitions (`0.40092176`); the result also holds under a
second reward regime (γ = 0.20 → **ε\* = 0.9312573**). Because the games are
monotone, ε\* is attained at the largest coalition (k=|U|−1), which is why LCH is
exact here. **However** these numbers are algorithmic self-consistency on *our* data,
not the paper's headline Table-1 time ranking. The paper's core conclusion that
**CQC is the fastest exact method is not reproduced** — under our substitution
(HiGHS, CQC via its min-over-`k` form) CQC is among the *slowest*, reversing the
paper's method ordering. Experimental conditions differ materially (dataset, solver
Gurobi→HiGHS, framework Julia→Python, hardware), so this is not a full or scaled
reproduction.

| Quantity | Paper result | Observed |
|---|---|---|
| ε\* (all methods agree) | identical ≥6 decimals | **identical to 8 decimals** at |U|=10–100 |
| LCH vs exact ε\* | "leave-one-out likely yield the least core value" | **LCH = ε\* exactly** (γ=0.10 & γ=0.20) |
| ε\*, γ=0.10 / γ=0.20 | — (illustrative) | 0.40092176 / 0.93125734 |
| Brute-force ground truth (|U|=10, γ=0.10) | — | **0.40092176** (exact match) |

**Condition differences vs the paper (why this is level C, not A/B):**
- **Data:** paper = real Fioriti et al. profiles (EnergyCommunity.jl); ours = synthetic 3-consumer/7-prosumer base, replicated. Changes which coalitions bind and the absolute times.
- **Solver:** Gurobi 13, 8 threads (4×18-core Xeon) → HiGHS (free, single-process default).
- **Framework:** Julia/JuMP/EnergyCommunity.jl+TheoryOfGames.jl → Python.
- **CQC:** paper solves the single non-convex QCP (34); we used its provably-equivalent min-over-`k` form because HiGHS did not solve the non-convex QCP optimally — this **reverses CQC's relative ordering**.
- **Sweep:** paper to |U|=200 (IT/ITI to 100); ours to 100, IT/ITI/CQC not run at 100.

**Compute.** All runs executed on the agreed **Vast.ai RTX 4090** instance
(`ssh1.vast.ai:18642`, SSH host alias `lcec-4090`) via `orx exp run --backend ssh --host lcec-4090`.

**Artifacts.**
- **Report** (visual write-up with figures): [`reports/least-core-reproduction/report.md`](reports/least-core-reproduction/report.md)
- **Notebook** (tutorial, opens with the measured evidence): [`leastcore_reproduction.py`](leastcore_reproduction.py) — run locally with `marimo edit leastcore_reproduction.py` or `marimo run leastcore_reproduction.py`.

### Experiment log

| Branch / experiment | Purpose / change | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| `orx/baseline-reproduce-least-core-value-methods-n-10` (exp `fdfcdafa-…`, runs `6d5e611e`, `586f8446`) | Baseline: reproduce ε\*, all 6 methods, γ=0.10, |U|=10–100 | `python3 -c "import numpy,highspy" 2>/dev/null \|\| pip3 install -q highspy numpy; cd reproduce && python3 reproduce_all.py 1` | **Supported** — ε\*=0.40092176 all methods + LCH + brute force (self-consistency on synthetic data) | Vast.ai RTX 4090 (SSH) |
| `orx/reward-sensitivity-higher-community-reward-esg-g` (exp `cad72778-…`, run `129b5ae8`) | Vary community reward γ 0.10→0.20 in `config.json`; same claim | `python3 -c "import numpy,highspy" 2>/dev/null \|\| pip3 install -q highspy numpy; cd reproduce && python3 reproduce_all.py 1` | **Supported** — ε\*=0.93125734 all methods + LCH + brute force | Vast.ai RTX 4090 (SSH) |

`main` is the publication surface (profile + README + report + notebook); it was
**not run as an experiment**. Experiment branches above carry the committed
reproduction source (`reproduce/`) and the run results live in `orx runs`.

---

## Project: Least cores in energy community games

Research reproduction project for the paper *Least cores in energy community games*
(Bigi, Fioriti, Frangioni, Passacantando, Poli; arXiv 2511.05291).
