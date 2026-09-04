"""Fig. 1: the tripod example against the 15 reflection planes — labels "P1  F=6  J=24", larger text, PDF + PNG.
  python -u icos_v2/analysis/plot_fig1_planes.py [--config tripod] [--scale 1.75]
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "icos", "codesign"), os.path.join(ROOT, "icos", "robot")]
import visualize as V                                    # noqa: E402 (frozen icos/ code, imported unchanged)
from robot_former import create_example_configs         # noqa: E402
from symmetry import get_reflection_planes, symmetry_scores_face, symmetry_scores_motor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="tripod"); ap.add_argument("--scale", type=float, default=1.75)
    ap.add_argument("--out", default=os.path.join(ROOT, "icos_v2", "paper", "iros-workshop-icos", "figures", "symmetry_metric_planes"))
    a = ap.parse_args()
    config = create_example_configs()[a.config]
    planes = get_reflection_planes(); F = symmetry_scores_face(config); J = symmetry_scores_motor(config)
    base = 11 * a.scale
    plt.rcParams.update({"font.size": base, "axes.titlesize": base, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 10))
    for idx in range(15):
        ax = fig.add_subplot(3, 5, idx + 1, projection="3d")
        _, perm = planes[idx]
        V._draw_icosahedron(ax, config, perm=perm, title=None)
        ax.set_title(f"P{idx+1}   F={F[idx]}   J={J[idx]}", fontsize=base, fontweight="bold" if F[idx] == 0 else "normal", pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.02, hspace=0.06)
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=150)
    print("→", a.out + ".pdf / .png   (F,J per plane:", list(zip(F, J)), ")")


if __name__ == "__main__":
    main()
