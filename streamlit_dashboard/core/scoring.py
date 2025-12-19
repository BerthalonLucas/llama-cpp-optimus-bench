from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def min_max_norm(value: float | None, *, min_v: float, max_v: float) -> float:
    if value is None:
        return 0.0
    if max_v == min_v:
        # All identical -> every present value gets 1.0
        return 1.0
    return (value - min_v) / (max_v - min_v)


@dataclass
class SweepScores:
    w_tg: float
    w_pp: float
    metric_tg: str | None = None  # Auto-detect if None
    metric_pp: str | None = None  # Auto-detect if None

    def _detect_metrics(self, run_records: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        """Detect tg and pp metric keys from run records."""
        tg_key = self.metric_tg
        pp_key = self.metric_pp
        
        for r in run_records:
            m = r.get("metrics") or {}
            if tg_key is None:
                tg_key = next((k for k in m.keys() if k.startswith("tg")), None)
            if pp_key is None:
                pp_key = next((k for k in m.keys() if k.startswith("pp")), None)
            if tg_key and pp_key:
                break
        
        return tg_key, pp_key

    def score_runs(self, run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute normalized scores in-place and return run_records.

        Normalization is computed across *this* list.
        Missing metrics -> 0.
        """
        
        # Auto-detect metric keys
        tg_key, pp_key = self._detect_metrics(run_records)

        tg_vals: list[float] = []
        pp_vals: list[float] = []
        for r in run_records:
            m = (r.get("metrics") or {})
            tv = m.get(tg_key) if tg_key else None
            pv = m.get(pp_key) if pp_key else None
            if isinstance(tv, (int, float)):
                tg_vals.append(float(tv))
            if isinstance(pv, (int, float)):
                pp_vals.append(float(pv))

        tg_min, tg_max = _min_max(tg_vals)
        pp_min, pp_max = _min_max(pp_vals)

        for r in run_records:
            m = (r.get("metrics") or {})
            tv = m.get(tg_key) if tg_key else None
            pv = m.get(pp_key) if pp_key else None
            tv_f = float(tv) if isinstance(tv, (int, float)) else None
            pv_f = float(pv) if isinstance(pv, (int, float)) else None

            n_tg = min_max_norm(tv_f, min_v=tg_min, max_v=tg_max)
            n_pp = min_max_norm(pv_f, min_v=pp_min, max_v=pp_max)

            score = self.w_tg * n_tg + self.w_pp * n_pp
            r["norm"] = {tg_key or "tg": n_tg, pp_key or "pp": n_pp}
            r["score"] = score

        return run_records


def compute_sweep_scores(run_records: list[dict[str, Any]], *, w_tg: float = 0.7, w_pp: float = 0.3) -> list[dict[str, Any]]:
    """Convenience function to compute sweep scores."""
    scorer = SweepScores(w_tg=w_tg, w_pp=w_pp)
    return scorer.score_runs(run_records)


def plot_sweep(df, x_col: str, y_col: str):
    """Create a matplotlib figure for sweep results.
    
    Returns a matplotlib figure or None if matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if df is None or df.empty:
            return None
        
        if x_col not in df.columns or y_col not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df[x_col], df[y_col], marker='o', linestyle='-', linewidth=2, markersize=6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.grid(True, alpha=0.3)
        
        # Highlight best point
        if not df[y_col].isna().all():
            best_idx = df[y_col].idxmax()
            best_x = df.loc[best_idx, x_col]
            best_y = df.loc[best_idx, y_col]
            ax.scatter([best_x], [best_y], color='red', s=100, zorder=5, label=f'Best: {best_y:.2f}')
            ax.legend()
        
        plt.tight_layout()
        return fig
    except ImportError:
        return None
    except Exception:
        return None
