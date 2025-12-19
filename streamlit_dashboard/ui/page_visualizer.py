from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import altair as alt

from ..core.gguf_meta import get_cached_meta
from ..core.paths import RepoPaths
from ..core.settings import AppSettings
from ..core.storage import (
    list_runs,
    list_sweeps,
    list_optimus_runs,
    list_optimizer_runs,
    load_run_json,
    load_json,
)
from .common import sidebar, get_paths, get_settings, st_dataframe


def _guess_quant(model_name: str) -> str:
    m = re.search(r"q\d+[a-z_]*", model_name.lower())
    return m.group(0).upper() if m else "?"


def _safe_float(x: Any) -> float | None:
    try:
        f = float(x)
        if pd.isna(f):
            return None
        return f
    except Exception:
        return None


def _model_meta(paths: RepoPaths, model_path_str: str) -> dict[str, Any]:
    try:
        p = Path(model_path_str)
        if not p.is_absolute():
            p = paths.models_dir / p
        meta = get_cached_meta(paths, p)
        return {
            "arch": meta.architecture,
            "is_moe": bool(meta.is_moe),
            "experts": meta.expert_count,
            "layer_count": meta.layer_count,
            "ctx_max": meta.extra.get("general.context_length")
            or meta.extra.get("llama.context_length")
            or meta.extra.get("gptneox.context_length"),
        }
    except Exception:
        return {}


def _add_common_fields(rec: dict[str, Any]) -> dict[str, Any]:
    rec["quant"] = _guess_quant(str(rec.get("model", "")))
    rec["tg"] = _safe_float(rec.get("tg"))
    rec["pp"] = _safe_float(rec.get("pp"))
    rec["mean"] = _safe_float(rec.get("mean"))
    rec["batch"] = _safe_float(rec.get("batch"))
    rec["ubatch"] = _safe_float(rec.get("ubatch"))
    rec["threads"] = _safe_float(rec.get("threads"))
    rec["ngl"] = _safe_float(rec.get("ngl"))
    rec["ncmoe"] = _safe_float(rec.get("ncmoe"))
    rec["ctx"] = _safe_float(rec.get("ctx"))
    return rec


def _load_bench(paths: RepoPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in list_runs(paths):
        data = load_run_json(d)
        if not data:
            continue
        cfg = data.get("bench_config") or {}
        m = data.get("metrics") or {}
        row = {
            "id": d.name,
            "source": "bench",
            "model": data.get("model_file", "?"),
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "tg": next((v for k, v in m.items() if str(k).startswith("tg")), None),
            "pp": next((v for k, v in m.items() if str(k).startswith("pp")), None),
            "batch": cfg.get("batch") or cfg.get("n_batch"),
            "ubatch": cfg.get("ubatch") or cfg.get("n_ubatch"),
            "threads": cfg.get("threads"),
            "ngl": cfg.get("gpu_layers"),
            "ncmoe": cfg.get("n_cpu_moe"),
            "ctx": data.get("ctx_size_tokens") or cfg.get("ctx_tokens"),
            "raw": data,
        }
        row.update(_model_meta(paths, row["model"]))
        rows.append(_add_common_fields(row))
    return rows


def _load_sweeps(paths: RepoPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sd in list_sweeps(paths):
        info_path = sd / "sweep.json"
        if not info_path.exists():
            info_path = sd / "sweep_info.json"
        info = load_json(info_path) if info_path.exists() else {}
        csv_path = sd / "sweep_results.csv"
        df = None
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path).convert_dtypes()
            except Exception:
                df = None
        if df is not None:
            for _, r in df.iterrows():
                row = {
                    "id": sd.name,
                    "source": "sweep",
                    "model": info.get("model_file") or r.get("model") or "?",
                    "status": r.get("status", info.get("status")),
                    "created_at": info.get("created_at"),
                    "tg": next((r[c] for c in df.columns if str(c).startswith("tg")), None),
                    "pp": next((r[c] for c in df.columns if str(c).startswith("pp")), None),
                    "batch": r.get("batch") or r.get("n_batch"),
                    "ubatch": r.get("ubatch") or r.get("n_ubatch"),
                    "threads": r.get("threads"),
                    "ngl": r.get("ngl") or r.get("gpu_layers"),
                    "ncmoe": r.get("n_cpu_moe"),
                    "ctx": r.get("ctx_size_tokens") or r.get("ctx_tokens"),
                    "raw": info,
                }
                row.update(_model_meta(paths, row["model"]))
                rows.append(_add_common_fields(row))
        else:
            # fallback minimal row
            row = {
                "id": sd.name,
                "source": "sweep",
                "model": info.get("model_file") or "?",
                "status": info.get("status"),
                "created_at": info.get("created_at"),
                "raw": info,
            }
            row.update(_model_meta(paths, row["model"]))
            rows.append(_add_common_fields(row))
    return rows


def _load_optimus(paths: RepoPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in list_optimus_runs(paths):
        data = load_run_json(d)
        if not data:
            continue
        cfg = data.get("final_config") or {}
        metrics = data.get("optimized_results") or {}
        row = {
            "id": d.name,
            "source": "optimus",
            "model": data.get("model_file", "?"),
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "tg": metrics.get("tg"),
            "pp": metrics.get("pp"),
            "mean": metrics.get("mean"),
            "batch": cfg.get("batch"),
            "ubatch": cfg.get("ubatch"),
            "threads": cfg.get("threads"),
            "ngl": cfg.get("ngl"),
            "ncmoe": cfg.get("ncmoe"),
            "ctx": data.get("ctx_size_tokens"),
            "raw": data,
        }
        row.update(_model_meta(paths, row["model"]))
        rows.append(_add_common_fields(row))
    return rows


def _load_optimizer(paths: RepoPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in list_optimizer_runs(paths):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            s = load_json(summary_path)
        except Exception:
            continue
        results = s.get("results") or []
        for r in results:
            bc = r.get("bench_config") or {}
            metrics = r.get("metrics") or {}
            row = {
                "id": d.name,
                "source": "optimizer",
                "model": s.get("config", {}).get("model_path_host", "?"),
                "status": r.get("status") or s.get("status"),
                "created_at": s.get("created_at"),
                "tg": next((v for k, v in metrics.items() if str(k).startswith("tg")), None),
                "pp": next((v for k, v in metrics.items() if str(k).startswith("pp")), None),
                "mean": r.get("score"),
                "batch": bc.get("batch"),
                "ubatch": bc.get("ubatch"),
                "threads": bc.get("threads"),
                "ngl": bc.get("gpu_layers"),
                "ncmoe": bc.get("n_cpu_moe"),
                "ctx": r.get("ctx_tokens"),
                "raw": r,
            }
            row.update(_model_meta(paths, row["model"]))
            rows.append(_add_common_fields(row))
    return rows


def _load_all(paths: RepoPaths) -> pd.DataFrame:
    rows = []
    rows += _load_bench(paths)
    rows += _load_sweeps(paths)
    rows += _load_optimus(paths)
    rows += _load_optimizer(paths)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Ensure numeric columns are numeric to avoid Arrow issues
    for col in ["tg", "pp", "mean", "batch", "ubatch", "threads", "ngl", "ncmoe", "ctx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "created_at" in df.columns:
        try:
            df["created_at_ts"] = pd.to_datetime(df["created_at"], errors="coerce")
        except Exception:
            df["created_at_ts"] = pd.NaT
    return df


def _filters_ui(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filtres")
    run_types = sorted(df["source"].dropna().unique())
    selected_types = st.sidebar.multiselect("Types", options=run_types, default=run_types)

    models = sorted(df["model"].dropna().unique())
    model_search = st.sidebar.text_input("Recherche modèle (contient)", value="")
    default_models = models
    selected_models = st.sidebar.multiselect("Modèles", options=models, default=default_models)
    if model_search:
        model_search_low = model_search.lower()
        selected_models = [m for m in models if model_search_low in str(m).lower()]
        if not selected_models:
            st.sidebar.info("Aucun modèle ne correspond à la recherche.")

    status_vals = sorted(df["status"].dropna().unique())
    selected_status = st.sidebar.multiselect("Statut", options=status_vals, default=status_vals)

    def _range_slider(label: str, series: pd.Series):
        s = series.dropna()
        if s.empty:
            return None
        mn, mx = float(s.min()), float(s.max())
        if mn == mx:
            st.sidebar.caption(f"{label}: {mn}")
            return None
        return st.sidebar.slider(label, mn, mx, (mn, mx))

    tg_range = _range_slider("tg (t/s)", df["tg"])
    pp_range = _range_slider("pp (t/s)", df["pp"])
    mean_range = _range_slider("mean (t/s)", df["mean"])
    ctx_range = _range_slider("ctx tokens", df["ctx"])

    filt = df.copy()
    if selected_types:
        filt = filt[filt["source"].isin(selected_types)]
    if selected_models:
        filt = filt[filt["model"].isin(selected_models)]
    if selected_status:
        filt = filt[filt["status"].isin(selected_status)]
    if tg_range:
        filt = filt[(filt["tg"].isna()) | ((filt["tg"] >= tg_range[0]) & (filt["tg"] <= tg_range[1]))]
    if pp_range:
        filt = filt[(filt["pp"].isna()) | ((filt["pp"] >= pp_range[0]) & (filt["pp"] <= pp_range[1]))]
    if mean_range:
        filt = filt[(filt["mean"].isna()) | ((filt["mean"] >= mean_range[0]) & (filt["mean"] <= mean_range[1]))]
    if ctx_range:
        filt = filt[(filt["ctx"].isna()) | ((filt["ctx"] >= ctx_range[0]) & (filt["ctx"] <= ctx_range[1]))]
    return filt


def _chart_bar(df: pd.DataFrame, metric: str, by: str = "model", title: str = ""):
    if df.empty or metric not in df.columns or by not in df.columns:
        return
    color_field = "source" if "source" in df.columns else None
    enc = {
        "x": alt.X(f"{metric}:Q", title=metric),
        "y": alt.Y(f"{by}:N", sort="-x", title=by),
        "tooltip": [c for c in ["source", "model", "tg", "pp", "mean", "batch", "ubatch", "ngl", "ncmoe"] if c in df.columns],
    }
    if color_field:
        enc["color"] = f"{color_field}:N"
    chart = alt.Chart(df).mark_bar().encode(**enc).properties(height=320, title=title or f"{metric} par {by}")
    st.altair_chart(chart, use_container_width=True)


def _chart_scatter(df: pd.DataFrame, x: str, y: str, color: str = "model", title: str = ""):
    if df.empty or x not in df.columns or y not in df.columns:
        return
    chart = (
        alt.Chart(df)
        .mark_circle(size=80, opacity=0.7)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            color=f"{color}:N",
            tooltip=["source", "model", "tg", "pp", "mean", "batch", "ubatch", "threads", "ngl", "ncmoe", "ctx"],
        )
        .properties(height=340, title=title or f"{y} vs {x}")
    )
    st.altair_chart(chart, use_container_width=True)


def _chart_pie_status(df: pd.DataFrame):
    if df.empty or "status" not in df.columns:
        return
    agg = df.groupby("status").size().reset_index(name="count")
    chart = (
        alt.Chart(agg)
        .mark_arc()
        .encode(theta="count:Q", color="status:N", tooltip=["status", "count"])
        .properties(title="Répartition des statuts")
    )
    st.altair_chart(chart, use_container_width=True)


def _aggregate_metric(df: pd.DataFrame, metric: str, by: str, agg: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    if agg == "max":
        grouped = df.groupby(by, dropna=True)[metric].max().reset_index()
    else:
        grouped = df.groupby(by, dropna=True)[metric].mean().reset_index()
    return grouped


def render() -> None:
    paths = get_paths()
    settings = get_settings()
    sidebar(paths, settings)

    st.title("Visualiser")

    df = _load_all(paths)
    if df.empty:
        st.warning("Aucune donnée trouvée (bench/sweep/optimus/optimizer).")
        return

    filt = _filters_ui(df)

    st.subheader(f"Données filtrées ({len(filt)})")
    display_cols = ["source", "model", "status", "tg", "pp", "mean", "batch", "ubatch", "threads", "ngl", "ncmoe", "ctx", "created_at", "id"]
    existing_cols = [c for c in display_cols if c in filt.columns]
    st_dataframe(filt[existing_cols], hide_index=True, use_container_width=True)
    st.download_button("Télécharger CSV (filtré)", filt.to_csv(index=False).encode("utf-8"), file_name="visualiser_filtered.csv")

    st.divider()
    st.subheader("Comparaisons rapides")

    met_options = [c for c in ["tg", "pp", "mean"] if c in filt.columns]
    met = st.selectbox("Métrique", options=met_options, index=0 if met_options else 0)
    agg_mode = st.selectbox("Agrégation", options=["max", "moyenne"], index=0)
    if met_options:
        agg_df = _aggregate_metric(filt, metric=met, by="model", agg="max" if agg_mode == "max" else "mean")
        if not agg_df.empty:
            _chart_bar(agg_df.rename(columns={"model": "model", met: met}), metric=met, by="model", title=f"{met} ({agg_mode}) par modèle")
        else:
            st.info("Aucune donnée pour ce graphique.")

    st.subheader("Scatter")
    _chart_scatter(filt, x="batch", y="tg", color="model", title="tg vs batch")
    _chart_scatter(filt, x="ctx", y="pp", color="model", title="pp vs ctx")

    st.subheader("Statuts")
    _chart_pie_status(filt)

    with st.expander("Données brutes JSON (échantillon)"):
        sample = filt.head(5).to_dict(orient="records")
        st.code(json.dumps(sample, indent=2, ensure_ascii=False, default=str), language="json")
