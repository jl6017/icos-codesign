"""Symmetry drift figure in the Science-Robotics scatter+regression style (icos_v2/paper/image.png):
every evaluated robot is a dot (x = generation, y = symmetry score), one linear regression per species with
its standard-error band, species overlaid in blue/red, drift baselines dashed in matching colours.

  python -u icos_v2/analysis/plot_sym_drift_style.py <run_444> <run_3333> [--out fig.png]
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sym_drift_s2r import mean_sym, drift_baseline, PART   # noqa: E402

BLUE, RED = "#2b8cff", "#ff4d3d"
STYLE = {"444": ("tripod (4,4,4)", RED), "3333": ("quadruped (3,3,3,3)", BLUE)}


def robots(run, elite=0):
    """All evaluated robots per generation, or (elite=K) the top-K by fitness of each generation's population
    (champion + offspring; cache-hit rows carry their fitness) — the survivors selection actually keeps."""
    by_gen = {}
    for line in open(os.path.join(run, "results.jsonl")):
        r = json.loads(line)
        if r.get("status") != "ok": continue
        by_gen.setdefault(r["gen"], []).append(r)
    xs, F, J = [], [], []
    for g, rows in sorted(by_gen.items()):
        if elite:
            seen, kept = set(), []
            for r in sorted(rows, key=lambda r: -r["displacement"]):
                k = tuple(r["config"])
                if k in seen: continue
                seen.add(k); kept.append(r)
                if len(kept) == elite: break
            rows = kept
        for r in rows:
            f, j = mean_sym(r["config"]); xs.append(g); F.append(f); J.append(j)
    return np.array(xs, float), np.array(F), np.array(J)


def regplot(ax, x, y, color, label, jitter=0.25, seed=0, per_gen=5):
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(x), bool)                     # sparser cloud: at most per_gen dots per generation
    for g in np.unique(x):
        idx = np.where(x == g)[0]; keep[rng.choice(idx, min(per_gen, len(idx)), replace=False)] = True
    ax.scatter(x[keep] + rng.uniform(-jitter, jitter, keep.sum()), y[keep] + rng.uniform(-0.03, 0.03, keep.sum()),
               s=18, color=color, alpha=0.5, edgecolors="none", label=label, zorder=3)
    res = stats.linregress(x, y)                      # regression on ALL robots
    xx = np.linspace(x.min(), x.max(), 100)
    yy = res.intercept + res.slope * xx
    n = len(x); s = np.sqrt(np.sum((y - (res.intercept + res.slope * x)) ** 2) / (n - 2))
    se = s * np.sqrt(1.0 / n + (xx - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))   # SE of the regression mean
    ax.plot(xx, yy, color=color, lw=2.2, zorder=4)
    ax.fill_between(xx, yy - se, yy + se, color=color, alpha=0.18, zorder=2)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_444"); ap.add_argument("run_3333"); ap.add_argument("--out", default=None)
    ap.add_argument("--elite", type=int, default=0, help="top-K per generation instead of all evaluated robots (0 = all)")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    runs = {"444": a.run_444, "3333": a.run_3333}
    drifts = {sp: drift_baseline(PART[sp]) for sp in runs}
    ylabels = ("Face-placement symmetry score  F", "Motor-level symmetry score  J")
    out = a.out or os.path.join(os.path.dirname(a.run_444), "sym_drift_style.png")
    for row, (sp, run) in enumerate(runs.items()):
        label, color = STYLE[sp]
        x, F, J = robots(run, a.elite)
        if a.elite: label = f"{label}, elites (top-{a.elite})"
        for k, name in enumerate(("F", "J")):
            ax = axes[row, k]; y = F if k == 0 else J
            res = regplot(ax, x, y, color, label, per_gen=min(5, a.elite) if a.elite else 5)
            ax.axhline(drifts[sp][k], color="0.3", ls="--", lw=1.3, alpha=0.8, zorder=1, label=f"drift baseline ({drifts[sp][k]:.2f})")
            # y-range around the regression line: line + drift, padded by the residual spread
            line = res.intercept + res.slope * np.array([x.min(), x.max()])
            resid_sd = np.std(y - (res.intercept + res.slope * x))
            lo, hi = min(line.min(), drifts[sp][k]), max(line.max(), drifts[sp][k])
            ax.set_ylim(lo - 0.8 * resid_sd, hi + 0.8 * resid_sd)
            ax.set_xlabel("Generation"); ax.set_ylabel(ylabels[k] + "  (lower = more symmetric)")
            ax.set_title(f"{label}: slope {res.slope*10:+.3f} / 10 gens, p = {res.pvalue:.3f}", fontsize=10)
            ax.legend(loc="upper right", frameon=True, framealpha=1.0, fontsize=9)
            print(f"{sp} {name}: slope {res.slope*10:+.3f}/10 gens  p={res.pvalue:.3f}  drift {drifts[sp][k]:.2f}")
            # each panel also as its own file
            ext = fig.dpi_scale_trans.inverted(); bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(ext)
            fig.savefig(out.replace(".png", f"_{sp}_{name}.png"), dpi=200, bbox_inches=bb.expanded(1.04, 1.06))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight"); print("→", out, "(+ four single-panel files)")


if __name__ == "__main__":
    main()
