"""Plot the two completed development runs with an existing matplotlib runtime."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    root = parser.parse_args().analysis_dir.resolve()
    runs = [root / "luna_agent_test_v2", root / "luna_agent_test_v3"]
    data = [json.loads((p / "metrics.json").read_text()) for p in runs]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    refs = ["obturator", "posterior", "anterior_superior"]
    cols = refs + ["other_unknown"]
    for ax, d, title in zip(
        axes, data, ["Initial conservative prompt", "Clarified directional policy"]
    ):
        cm = d["reports"]["confusion"]
        for y, r in enumerate(refs):
            for x, c in enumerate(cols):
                count = cm[r][c]
                color = (
                    "#e5a845"
                    if count and c == "other_unknown"
                    else ("#1b647c" if count else "#eff3f6")
                )
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=3,
                    )
                )
                ax.text(
                    x,
                    y,
                    str(count),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=19,
                    color="white" if count and c != "other_unknown" else "#172b3a",
                )
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_xticks(
            range(4),
            ["Obturator", "Posterior", "Anterior-\nsuperior", "Other /\nunknown"],
        )
        ax.set_yticks(
            range(3),
            ["Obturator\n(n=22)", "Posterior\n(n=19)", "Anterior-superior\n(n=1)"],
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Model output", labelpad=10)
        ax.set_title(
            title
            + "\n"
            + f"{d['reports']['correct']}/42 labels matched ({d['reports']['agreement_all']:.1%})",
            fontweight="bold",
            pad=15,
        )

    fig.suptitle(
        "Live GPT-6 Luna via ClawAgents: ambiguity policy changes coverage",
        fontweight="bold",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.10,
        "Final linked-record result: 28/31 assigned and correct; IDs 40, 45, 76 remain for review.\nBoth prompts tested on the same selected examples. This is development performance, not held-out validation.",
        ha="center",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.75, bottom=0.29, wspace=0.42)
    fig.savefig(runs[1] / "Live_agent_performance.png", dpi=180, facecolor="white")
    fig.savefig(runs[1] / "Live_agent_performance.svg", facecolor="white")


if __name__ == "__main__":
    main()
