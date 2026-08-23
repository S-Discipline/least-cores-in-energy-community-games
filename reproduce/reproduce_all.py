"""Full reproduction driver: computes eps* and wall-times of the 6 approaches
(RG, IT, ITI, ITD, CQC, LCH) across the paper's sizes and reports a table.
Also runs an exhaustive (brute force) ground truth for the smallest size."""
import sys, json, time
import numpy as np
import esg as E

def run_size(n_reps, methods, brute=True):
    g = E.make_instance(n_reps)
    n = g.n
    out = {"n": n, "methods": {}}
    V = E.vN(g)
    out["v(N)"] = float(V)
    for m in methods:
        t0 = time.time()
        if m == "IT":   eps, dt, det = E.run_it(g)
        elif m == "ITI": eps, dt, det = E.run_iti(g)
        elif m == "ITD": eps, dt, det = E.run_itd(g)
        elif m == "RG":  eps, dt, det = E.run_rg(g)
        elif m == "CQC": eps, dt, det = E.run_cqc(g)
        elif m == "LCH": eps, dt, det = E.run_lch(g)
        else: raise ValueError(m)
        out["methods"][m] = {"eps": float(eps), "time_s": round(dt, 3),
                             "detail": {k: str(v) for k, v in det.items()}}
    if brute:
        best = float("inf"); bestS = None
        import itertools
        for k in range(2, n):
            for S in itertools.combinations(range(n), k):
                e = (V - E.vS_at(g, list(S))) / (n + 1 - k)
                if e < best: best = e; bestS = S
        out["brute_eps"] = float(best)
        out["brute_S_size"] = len(bestS)
    return out

def main():
    sizes_methods = {
        1: ["IT", "ITI", "ITD", "RG", "CQC", "LCH"],
        2: ["IT", "ITI", "ITD", "RG", "CQC", "LCH"],
        5: ["IT", "ITI", "ITD", "RG", "CQC", "LCH"],
        10: ["ITD", "RG", "LCH"],   # full IT/ITI/all-k CQC prohibitive at 100 users (paper: not run)
    }
    only = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    results = []
    for reps, methods in sizes_methods.items():
        if only and reps != only: continue
        r = run_size(reps, methods, brute=(reps == 1))
        results.append(r)
        print(f"## n={r['n']}  v(N)={r['v(N)']:.4f}" + (f"  brute_eps={r['brute_eps']:.8f}" if 'brute_eps' in r else ""), flush=True)
        for m, d in r["methods"].items():
            print(f"   {m:4s}: eps={d['eps']:.8f} time={d['time_s']:9.3f}s {d['detail']}")
        sys.stdout.flush()
    if only:
        json.dump(results, open(f"results_n{results[0]['n']}.json", "w"), indent=2, default=str)
    else:
        json.dump(results, open("results.json", "w"), indent=2, default=str)
    print("saved", flush=True)

if __name__ == "__main__":
    main()
