"""Simple per-pair summary charts - descriptive documentation of results
already computed elsewhere, controlled by config.SAVE_GRAPHS."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import helpers


def _line_chart(x, series, ylabel, title, out_path, ylim=None):
    fig, ax = plt.subplots(figsize=(max(6, len(x) * 1.2), 5))
    for label, values, marker in series:
        ax.plot(x, values, marker=marker, label=label)
    ax.set_xlabel("Pair")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_graphs(results):
    if not results:
        return []

    graphs_folder = helpers.ensure_dir(os.path.join(config.OUTPUT_FOLDER, "Graphs"))
    pair_ids = [r["pair_id"] for r in results]
    paths = []

    p = os.path.join(graphs_folder, "psnr_per_pair.png")
    _line_chart(pair_ids, [("PSNR (dB)", [r["psnr"] for r in results], "o")],
                "PSNR (dB)", "PSNR per Pair", p)
    paths.append(p)

    p = os.path.join(graphs_folder, "ssim_msssim_per_pair.png")
    _line_chart(pair_ids, [
        ("SSIM", [r["ssim"] for r in results], "o"),
        ("MS-SSIM", [r["ms_ssim"] for r in results], "s"),
    ], "Score", "SSIM / MS-SSIM per Pair", p, ylim=(0, 1.05))
    paths.append(p)

    p = os.path.join(graphs_folder, "error_metrics_per_pair.png")
    _line_chart(pair_ids, [
        ("MAE (%)", [r["mae_percent"] for r in results], "o"),
        ("RMSE (%)", [r["rmse_percent"] for r in results], "s"),
        ("Max Error (%)", [r["max_error_percent"] for r in results], "^"),
    ], "Normalized Error (%)", "Normalized Error Metrics per Pair", p)
    paths.append(p)

    return paths
