"""
metrics/normalized_error.py
=============================
Contains only `calculate_normalized_error()`.
"""


def calculate_normalized_error(rmse: float, max_intensity: float = 255.0) -> float:
    """Compute the Normalized Error, expressed as a percentage of the maximum
    possible pixel-intensity range.

    Takes the already-computed RMSE as input (rather than recomputing it
    from the two images) to avoid duplicating the RMSE calculation that
    already lives in metrics/rmse.py.

    Parameters
    ----------
    rmse : float
        Root Mean Squared Error between ground-truth and prediction.
    max_intensity : float, optional
        Maximum possible pixel intensity (255.0 for 8-bit grayscale
        imagery, the default).

    Returns
    -------
    float
        Normalized error as a percentage.
    """
    return float((rmse / max_intensity) * 100.0)
