#!/usr/bin/env python
"""Tables for the referee response (comments 4 and 6) from profile_pe.py outputs.

    python summarise_profile.py profile_gpu.json profile_cpu_1.json profile_cpu_4.json ... [--batch 100]

* per-evaluation cost of every likelihood stage on each device / thread count, at the batch size
  the sampler uses (num_delete = 100 chains advance together), and JIT compile times;
* GW170817: split of the production run's wall time (from its timing.json numbers) into waveform,
  relative-binning overlaps, tc/phi_c marginalisation and sampler overhead.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("profiles", nargs="+")
ap.add_argument("--batch", type=int, default=100)
# production GW170817 run 20261114_162230 (timing.json) and the CPU references
ap.add_argument("--gw170817-wall", type=float, default=1236.41)
ap.add_argument("--gw170817-calls", type=float, default=7460504)
ap.add_argument("--out", default=None)
a = ap.parse_args()

lab = lambda p: p["info"]["label"] or p["info"]["device"]
# files with the same label (e.g. profile_gpu.json + profile_gpu_hm.json from a second GPU process)
# are merged into one column; later files add the analyses the earlier ones lack
P = []
for f in a.profiles:
    p = json.load(open(f))
    same = next((q for q in P if lab(q) == lab(p)), None)
    if same is None:
        P.append(p)
    else:
        for w, r in p["results"].items():
            same["results"].setdefault(w, r)


def per_eval(stage, b):
    if b in stage:
        return stage[b]["per_eval_s"]
    if str(b) in stage:
        return stage[str(b)]["per_eval_s"]
    ks = sorted(int(k) for k in stage if str(k) != "errors")            # largest measured batch <= b
    k = max([x for x in ks if x <= b] or ks[:1])
    return stage[str(k)]["per_eval_s"] if str(k) in stage else stage[k]["per_eval_s"]


def compile_s(stage):
    v = stage.get(str(1), stage.get(1)) or next(v for k, v in stage.items() if k != "errors")
    return v["compile_s"]


out = {}
for what in ("gw170817", "gw150914_22", "gw150914_hm"):
    rows = [(lab(p), p["results"][what]) for p in P if what in p["results"]]
    gpu_labels = {lab(p) for p in P if p["info"]["platform"] == "gpu"}
    if not rows:
        continue
    print(f"\n=== {what}: per-evaluation cost at batch {a.batch} [ms]  (JIT compile [s])")
    stages = list(rows[0][1]["stages"])
    print(f"{'stage':38s}" + "".join(f"{l[:22]:>24s}" for l, _ in rows))
    out[what] = {}
    for s in stages:
        cells = []
        for l, r in rows:
            st = r["stages"].get(s)
            if st is None:
                cells.append(f"{'-':>24s}"); continue
            v = per_eval(st, a.batch); c = compile_s(st)
            out[what].setdefault(s, {})[l] = dict(per_eval_ms=1e3 * v, compile_s=c)
            cells.append(f"{1e3 * v:14.4f} ({c:5.1f} s)")
        print(f"{s:38s}" + "".join(cells))

    if what == "gw170817":
        for l, r in rows:
            S = r["stages"]
            if l not in gpu_labels:          # the production run was on a GPU
                continue
            if not all(k in S for k in ("waveform_rb_bins", "rb_fixed_tc_phi", "rb_tc_phi_marg")):
                continue
            n = a.gw170817_calls
            wf, rbp, prod = (per_eval(S[k], a.batch) for k in ("waveform_rb_bins", "rb_fixed_tc_phi", "rb_tc_phi_marg"))
            parts = dict(waveform=n * wf, rb_overlaps_projection=n * max(rbp - wf, 0), tc_phi_marginalisation=n * max(prod - rbp, 0))
            like = sum(parts.values())
            print(f"\nGW170817 production run on {l}: {n:.3g} likelihood calls, wall {a.gw170817_wall:.0f} s")
            for k, v in parts.items():
                print(f"  {k:28s} {v:8.1f} s  ({v / a.gw170817_wall:5.1%})")
            print(f"  {'sampler (proposals, prior, bookkeeping, JIT)':28s} {a.gw170817_wall - like:8.1f} s  ({1 - like / a.gw170817_wall:5.1%})")
            if "full_fixed_tc_phi" in S:
                print(f"  without RB (full band, fixed tc/phi, same calls): {n * per_eval(S['full_fixed_tc_phi'], a.batch) / 3600:.1f} h")
            if "full_tc_phi_marg_501tc" in S:
                t = per_eval(S["full_tc_phi_marg_501tc"], 1) * 2001 / 501
                print(f"  without RB, tc/phi marg. on 2001 tc points (scaled from 501): {n * t / 3600:.1f} h")
            out.setdefault("gw170817_breakdown", {})[l] = dict(parts_s=parts, sampler_overhead_s=a.gw170817_wall - like)

print("""
CPU references (bilby/dynesty, same data/priors): GW170817 TEOBResumSPA RB 31-39 ms/call (1 core), run 9.2 h x 32 cores
for 2.97e7 calls; GW150914 SEOBNRv5HM (2,2) 47 ms/call (laptop, 1 core); GW150914 SEOBNRv5HM 7 modes: 97.8 h x 16 cores
for 4.73e7 calls (= 0.12 core-s per call).""")
if a.out:
    json.dump(out, open(a.out, "w"), indent=2)
