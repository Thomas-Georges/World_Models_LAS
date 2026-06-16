from __future__ import annotations


def set_default_plot_style() -> None:
    """Apply a small, report-friendly matplotlib style when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is not installed in this environment.") from exc

    plt.rcParams.update(
        {
            "figure.figsize": (7, 4),
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
