"""Fitness-curve plots for hw17 within-species EA runs (evolve_hw17.py output).

  python -u icos_v2/analysis/plot_ea_hw17.py <run_dir> [<run_dir2> ...] [--out fig.png]

Per run: (1) best/mean/worst per generation from history.json, (2) every robot as a dot from
results.jsonl (cache hits hollow), (3) champion symmetry distance per generation.
Multiple runs (e.g. 3333 + 444) are overlaid in the fitness panel for the paper figure.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"3333": "#1f77b4", "444": "#d62728"}


def load(run):
    hist = json.load(open(os.path.join(run, "history.json")))
    rows = [json.loads(l) for l in open(os.path.join(run, "results.jsonl")) if l.strip()]
    sp = list(hist[0]["species"].keys())[0]
    label = "".join(c for c in sp if c.isdigit())
    gens = [h["gen"] for h in hist]
    s = [list(h["species"].values())[0] for h in hist]
    return dict(run=run, label=label, gens=gens,
                best=[x["best"] for x in s], mean=[x["mean"] for x in s], worst=[x.get("worst", np.nan) for x in s],
                champ_sym=[x.get("champ_sym", np.nan) for x in s], mean_sym=[x.get("mean_sym", np.nan) for x in s],
                rows=[r for r in rows if r.get("status") == "ok"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = [load(r) for r in a.runs]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for d in data:
        c = COLORS.get(d["label"])
        ax.plot(d["gens"], d["best"], "-", color=c, lw=2, label=f'{d["label"]} best')
        ax.plot(d["gens"], d["mean"], "--", color=c, lw=1.2, alpha=0.8, label=f'{d["label"]} mean')
        ax.fill_between(d["gens"], d["worst"], d["best"], color=c, alpha=0.10)
    ax.set_xlabel("generation"); ax.set_ylabel("displacement (m)")
    ax.set_title("EA progress (pop 16, 20M steps, S2R+DR)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for d in data:
        c = COLORS.get(d["label"])
        fresh = [r for r in d["rows"] if "cached_from" not in r]
        cach = [r for r in d["rows"] if "cached_from" in r]
        ax.scatter([r["gen"] for r in fresh], [r["displacement"] for r in fresh], s=12, color=c, alpha=0.55,
                   label=f'{d["label"]} trained (n={len(fresh)})')
        if cach:
            ax.scatter([r["gen"] for r in cach], [r["displacement"] for r in cach], s=14, facecolors="none",
                       edgecolors=c, alpha=0.55, label=f'{d["label"]} cache hit (n={len(cach)})')
    ax.set_xlabel("generation"); ax.set_ylabel("displacement (m)")
    ax.set_title("every robot"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for d in data:
        c = COLORS.get(d["label"])
        ax.plot(d["gens"], d["champ_sym"], "-", color=c, lw=2, label=f'{d["label"]} champion')
        ax.plot(d["gens"], d["mean_sym"], "--", color=c, lw=1.2, alpha=0.8, label=f'{d["label"]} pop mean')
    ax.set_xlabel("generation"); ax.set_ylabel("symmetry distance (motor)")
    ax.set_title("symmetry drift"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = a.out or os.path.join(a.runs[0], "ea_curves.png")
    fig.savefig(out, dpi=150)
    print("→", out)
    for d in data:
        n_f = len([r for r in d["rows"] if "cached_from" not in r]); n_c = len(d["rows"]) - n_f
        print(f'  {d["label"]}: gens={len(d["gens"])} best={max(d["best"]):.3f} trained={n_f} cache_hits={n_c}')


if __name__ == "__main__":
    main()
