"""
analysis/ranking.py
=====================
Responsible ONLY for sorting by RMSE and selecting the Best / Worst
samples. Nothing else.
"""

import pandas as pd

import config


def rank_and_select_samples(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank all evaluated samples and select the Top-Best / Top-Worst subsets.

    Ranking policy:
        PRIMARY key   : lowest RMSE (best reconstruction fidelity)
        SECONDARY key : highest SSIM (best structural similarity)

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Full per-image metrics table (one row per evaluated sample).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (best_samples_df, worst_samples_df). Worst samples are ordered
        worst-first.
    """
    ranked = metrics_df.sort_values(by=["RMSE", "SSIM"], ascending=[True, False]).reset_index(drop=True)

    n_best = min(config.TOP_BEST_IMAGES, len(ranked))
    n_worst = min(config.TOP_WORST_IMAGES, len(ranked))

    best_samples = ranked.head(n_best).copy()
    worst_samples = ranked.tail(n_worst).copy().iloc[::-1].reset_index(drop=True)

    return best_samples, worst_samples
