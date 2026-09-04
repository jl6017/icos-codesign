"""Merged elite symmetry-drift figure (abstract mixed / abstract speciated / realistic tripod / realistic 3333) x (F, J),
clean labels: one shared legend, short row titles, slope & p as in-panel text.
  python -u icos_v2/analysis/plot_drift_merged.py --run444 <dir> --run3333 <dir> [--elite 8] [--out fig.png]
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "icos_v2", "analysis"))
from icos.presentation.plot_symmetry_combined import mean_sym as mean_sym_v, compute_drift_baselines, parse_v6_log, parse_v7_log  # noqa
import json

S2R_DRIFT = {"444": (4.85, 19.64), "3333": (5.99, 18.25)}   # plot_sym_drift_s2r.drift_baseline (pop 16, 50 gens, 5 seeds)


def s2r_robots(run, elite):
    """Top-`elite` robots by fitness per generation of an S2R run (cache-hit rows carry their fitness)."""
    by_gen = {}
    for line in open(os.path.join(run, "results.jsonl")):
        r = json.loads(line)
        if r.get("status") == "ok": by_gen.setdefault(r["gen"], []).append(r)
    xs, F, J = [], [], []
    for g, rows in sorted(by_gen.items()):
        seen, kept = set(), []
        for r in sorted(rows, key=lambda r: -r["displacement"]):
            k = tuple(r["config"])
            if k in seen: continue
            seen.add(k); kept.append(r)
            if len(kept) == elite: break
        for r in kept:
            f, j = mean_sym_v(np.asarray(r["config"])); xs.append(g); F.append(f); J.append(j)
    return np.array(xs, float), np.array(F), np.array(J)

BLUE, RED = "#2b8cff", "#ff4d3d"


def fit(ax, x, y, color, per_gen=4, seed=0):
    rng = np.random.default_rng(seed); keep = np.zeros(len(x), bool)
    for g in np.unique(x):
        idx = np.where(x == g)[0]; keep[rng.choice(idx, min(per_gen, len(idx)), replace=False)] = True
    ax.scatter(x[keep] + rng.uniform(-0.3, 0.3, keep.sum()), y[keep], s=11, color=color, alpha=0.45, edgecolors="none", zorder=3)
    r = stats.linregress(x, y); xx = np.linspace(x.min(), x.max(), 100); yy = r.intercept + r.slope * xx
    n = len(x); s = np.sqrt(np.sum((y - (r.intercept + r.slope * x)) ** 2) / (n - 2))
    se = s * np.sqrt(1 / n + (xx - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.plot(xx, yy, color=color, lw=2, zorder=4); ax.fill_between(xx, yy - 1.96 * se, yy + 1.96 * se, color=color, alpha=0.28, zorder=2)
    return r


def panel(ax, x, y, color, drift, drift_curve=None):
    """drift: constant baseline; drift_curve: (gens, values) composition-matched baseline drawn instead."""
    r = fit(ax, x, y, color)
    if drift_curve is not None:
        gx, gy = drift_curve; ax.step(gx, gy, where="mid", color="0.25", ls="--", lw=1.2, zorder=1); ds = list(gy)
    else:
        ax.axhline(drift, color="0.25", ls="--", lw=1.2, zorder=1); ds = [drift]
    line = r.intercept + r.slope * np.array([x.min(), x.max()]); sd = np.std(y - (r.intercept + r.slope * x))
    lo, hi = min(line.min(), *ds), max(line.max(), *ds); ax.set_ylim(lo - 0.9 * sd, hi + 0.9 * sd)
    ptxt = "p<0.001" if r.pvalue < 0.001 else (f"p={r.pvalue:.3f}" if r.pvalue < 0.01 else f"p={r.pvalue:.2f}")
    ax.text(0.02, 0.95, f"slope {r.slope*10:+.2f}/10 gen, {ptxt}", transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.grid(alpha=0.2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run444", required=True); ap.add_argument("--run3333", required=True)
    ap.add_argument("--elite", type=int, default=8); ap.add_argument("--out", default="icos_v2/paper/iros-workshop-icos/figures/drift_merged_6x2.png")
    a = ap.parse_args()
    base = compute_drift_baselines(n_gens=50, pop_per_species=20, n_seeds=5); d_all, d_444 = base["all"], base[(4, 4, 4)]
    gens6, elite6 = parse_v6_log("evolve_runs_v6/evolve-20260217-011249/evolve.log", elite_size=16)
    rows6 = [(g, c) for g in gens6 for c, _ in elite6[g]]
    by_gen = defaultdict(list)
    for g, c, d in parse_v7_log("evolve_runs_v7/evolve-20260218-012327/evolve.log"): by_gen[g].append((c, d))
    rows7 = [(g, c) for g in sorted(by_gen) for c, _ in sorted(by_gen[g], key=lambda e: -e[1])[:10]]
    sys.path.insert(0, os.path.join(ROOT, "icos_v2", "codesign")); from evo_operator import dof_partition
    rows6t = [(g, c) for g, c in rows6 if dof_partition(np.asarray(c)) == (4, 4, 4)]
    rows7t = [(g, c) for g in sorted(by_gen) for c, _ in by_gen[g] if dof_partition(np.asarray(c)) == (4, 4, 4)]
    def fj(rows):
        x = np.array([g for g, _ in rows], float)
        F = np.array([mean_sym_v(np.asarray(c))[0] for _, c in rows]); J = np.array([mean_sym_v(np.asarray(c))[1] for _, c in rows]); return x, F, J
    x6, F6, J6 = fj(rows6); x7, F7, J7 = fj(rows7); x6t, F6t, J6t = fj(rows6t); x7t, F7t, J7t = fj(rows7t)
    def matched(rows, k):
        """per-generation drift baseline averaged over the species present in that elite set"""
        gens = sorted({g for g, _ in rows}); out = []
        for g in gens:
            parts = [dof_partition(np.asarray(c)) for gg, c in rows if gg == g]
            out.append(np.mean([base.get(p, d_all)[k] for p in parts]))
        return np.array(gens, float), np.array(out)
    bl6 = (matched(rows6, 0), matched(rows6, 1)); bl7 = (matched(rows7, 0), matched(rows7, 1))
    x4, F4, J4 = s2r_robots(a.run444, a.elite); x3, F3, J3 = s2r_robots(a.run3333, a.elite)
    dr444 = S2R_DRIFT["444"]; dr3333 = S2R_DRIFT["3333"]
    GREEN = "#2ca25f"
    rows = [("Abstract, mixed competition: elites (top-16)", x6, F6, J6, RED, d_all, bl6),
            ("Abstract, mixed competition: tripod subpopulation", x6t, F6t, J6t, RED, d_444, None),
            ("Abstract, speciated: elites (top-10 across species)", x7, F7, J7, GREEN, d_all, bl7),
            ("Abstract, speciated: tripod species", x7t, F7t, J7t, GREEN, d_444, None),
            (f"Realistic, tripod (4,4,4): elites (top-{a.elite})", x4, F4, J4, BLUE, dr444, None),
            (f"Realistic, (3,3,3,3): elites (top-{a.elite})", x3, F3, J3, BLUE, dr3333, None)]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(6, 2, figsize=(7.2, 10.4), sharex="row")
    for i, (title, x, F, J, color, dr, bl) in enumerate(rows):
        panel(axes[i, 0], x, F, color, dr[0], None if bl is None else bl[0]); axes[i, 0].set_ylabel("F  (face placement)")
        panel(axes[i, 1], x, J, color, dr[1], None if bl is None else bl[1]); axes[i, 1].set_ylabel("J  (motor level)")
        axes[i, 0].set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=4)
        if i == 5:
            for ax in axes[i]: ax.set_xlabel("generation")
    handles = [Line2D([], [], marker="o", ls="", color="0.4", ms=4, label="individual candidates (thinned)"),
               Patch(fc="0.7", alpha=0.6, label="linear regression, 95% CI"),
               Line2D([], [], color="0.25", ls="--", label="drift baseline (species-matched)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.032, 1, 1)); fig.savefig(a.out, dpi=200); print("→", a.out)


if __name__ == "__main__":
    main()
