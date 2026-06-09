#!/usr/bin/env python3
"""
summarize_searchlight_reliability_ONEFILE.py

Creates ONE self-contained HTML report (tables + plots embedded) summarizing
post-RSA searchlight split-half reliability across approaches.

This version supports arbitrary approach folder names passed via --approaches,
while still restricting scans to ONLY those approach folders under --root.

Key behaviors:
- ALWAYS includes all rows regardless of passed_gate (no gating filter).
- Produces a single report file: <outdir>/SUMMARY.html
- Optionally also writes reliability_long.csv + dropped_duplicates.csv.

Approach labeling:
- Primarily assigns approach by detecting which --approaches folder name appears
  in the provenance JSON path (longest match wins).
- Special-case: files under /D_up3x_legacy_upsampled_slicetime_change/ are labeled
  as D_up3x_legacy (even if that alt folder is used on disk).
"""

import os
import re
import json
import argparse
import base64
from io import BytesIO
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


APP_DEFAULT = ["B_up3x_cov40", "B_up3x_subject_specific_ROIs", "B_up3x_min_neighbor", "D_up3x_legacy"]
DT_DEFAULT  = ["unsmoothed", "smoothed"]


def _normalize_sub(s: str) -> str:
    s = str(s).strip()
    if s.startswith("sub-"):
        return s
    if s.isdigit():
        return f"sub-{int(s):03d}"
    return f"sub-{s}"


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _safe_int(x):
    try:
        if x is None:
            return 0
        return int(x)
    except Exception:
        return 0


def _canonical_approach_from_path(prov_path: str,
                                 approach_names: List[str],
                                 alias_dirs: Dict[str, List[str]]) -> str:
    """
    Return the canonical approach label based on which approach folder is present in the path.
    - alias_dirs maps canonical label -> list of on-disk folder names that should map to it.
    - If multiple matches, longest match wins (more specific folder names first).
    """
    p = (prov_path or "").replace("\\", "/")

    # 1) Alias mapping first (e.g., alt D legacy folder -> canonical D_up3x_legacy)
    for canon_label, dirnames in (alias_dirs or {}).items():
        for dn in dirnames:
            if dn and (f"/{dn}/" in p):
                return canon_label

    # 2) Direct match against user-provided approach names
    matches = [a for a in approach_names if a and (f"/{a}/" in p)]
    if matches:
        return max(matches, key=len)

    return ""


def _infer_meta_from_path(prov_path: str,
                          approach_names: List[str],
                          alias_dirs: Dict[str, List[str]]) -> dict:
    """
    Infer subject / datatype / approach / radius / metric / min_neigh from the provenance path.
    Fallback when JSON keys are missing or renamed.
    """
    p = str(prov_path or "").replace("\\", "/")
    meta = {
        "subject": "",
        "datatype": "",
        "approach": "",
        "radius": 0,
        "metric": "",
        "min_neigh": 0,
    }

    # subject
    m = re.search(r"(sub-\d+)", p)
    if m:
        meta["subject"] = _normalize_sub(m.group(1))

    # datatype
    m = re.search(r"(?:/|_)(unsmoothed|smoothed)(?:/|_)", p)
    if m:
        meta["datatype"] = m.group(1)

    # approach: prefer canonical folder-based approach label
    canon = _canonical_approach_from_path(p, approach_names, alias_dirs)
    if canon:
        meta["approach"] = canon
    else:
        # then try "approachB_up3x" style
        m = re.search(r"(?:/|_)approach([A-Za-z0-9_]+?)(?:/|_)", p)
        if m:
            meta["approach"] = m.group(1)
        else:
            # fallback: some coarse folder tokens
            m = re.search(r"(?:/)(A_up3x|B_up3x|D_up3x_legacy|nilearn)(?:/)", p)
            if m:
                meta["approach"] = m.group(1)

    # radius (rad3 / radius3 / radius-3 etc.)
    m = re.search(r"(?:rad|radius)[-_]?(\d+)", p)
    if m:
        meta["radius"] = _safe_int(m.group(1))

    # min_neigh / k (k60 / K60 / knn_k60 etc.)
    m = re.search(r"(?:^|[/_])k(\d+)(?:$|[/_])", p, flags=re.IGNORECASE)
    if m:
        meta["min_neigh"] = _safe_int(m.group(1))
    else:
        m = re.search(r"(?:knn|neighbors|neigh)[-_]?k(\d+)", p, flags=re.IGNORECASE)
        if m:
            meta["min_neigh"] = _safe_int(m.group(1))

    # metric (look for common tokens in the path/filename)
    for tok in ["crossnobis", "pearson", "spearman", "correlation", "corr", "cosine", "euclidean"]:
        if re.search(rf"(?:^|[/_]){re.escape(tok)}(?:$|[/_])", p, flags=re.IGNORECASE) or re.search(rf"{re.escape(tok)}", p, flags=re.IGNORECASE):
            meta["metric"] = tok.lower()
            break

    if meta["metric"] == "correlation":
        meta["metric"] = "corr"

    return meta


def find_provenance_jsons(roots) -> List[str]:
    """
    Scan ONLY the provided root folder(s) (no other locations) for *_provenance.json.
    """
    if isinstance(roots, str):
        roots = [roots]
    out = []
    seen = set()
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith("_provenance.json"):
                    p = os.path.join(dirpath, fn)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
    return sorted(out)


def load_one_prov(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            d = json.load(f)
        d["_prov_path"] = path
        d["_mtime"] = os.path.getmtime(path)
        return d
    except Exception:
        return None


def prov_to_row(d: dict,
                approach_names: List[str],
                alias_dirs: Dict[str, List[str]]) -> dict:
    prov_path = d.get("_prov_path", "")
    inf = _infer_meta_from_path(prov_path, approach_names, alias_dirs)

    subj_raw = d.get("subject", "") or d.get("sub", "") or d.get("subj", "") or inf["subject"]
    subj = _normalize_sub(subj_raw) if subj_raw else _normalize_sub(inf["subject"])

    approach_raw = (
        str(d.get("approach", "")).strip()
        or str(d.get("ab_tag", "")).strip()
        or str(d.get("pipeline", "")).strip()
        or str(inf["approach"]).strip()
    )
    if approach_raw.startswith("approach"):
        approach_raw = approach_raw[len("approach"):]
    approach = approach_raw.strip()

    # Override approach with canonical folder-based label if available
    canon = _canonical_approach_from_path(prov_path, approach_names, alias_dirs)
    if canon:
        approach = canon

    datatype_raw = str(d.get("datatype", "")).strip() or str(d.get("data_type", "")).strip() or str(inf["datatype"]).strip()
    datatype = datatype_raw.strip()

    radius = _safe_int(
        d.get("radius",
              d.get("searchlight_radius",
                    d.get("radius_vox",
                          d.get("rad", None))))
    )
    if (not radius) and inf["radius"]:
        radius = int(inf["radius"])

    metric = (
        str(d.get("metric", "")).strip()
        or str(d.get("rsa_metric", "")).strip()
        or str(d.get("dist_metric", "")).strip()
        or str(d.get("distance_metric", "")).strip()
        or str(d.get("similarity_metric", "")).strip()
        or str(inf["metric"]).strip()
    ).strip()

    min_neigh = _safe_int(
        d.get("min_neigh",
              d.get("min_neigh_k",
                    d.get("knn_k",
                          d.get("k",
                                d.get("K", None)))))
    )
    if (not min_neigh) and inf["min_neigh"]:
        min_neigh = int(inf["min_neigh"])

    row = {
        "subject": subj,
        "approach": approach,
        "datatype": datatype,

        "radius": radius,
        "metric": metric,
        "min_neigh": min_neigh,

        "have_trials": bool(d.get("have_trials", False)),
        "passed_gate": bool(d.get("passed_gate", False)),

        "global_rel_spear": _safe_float(d.get("global_rel_spear")),
        "global_rel_pear": _safe_float(d.get("global_rel_pear")),
        "offdiag_mean": _safe_float(d.get("offdiag_mean")),
        "tSNR_median": _safe_float(d.get("tSNR_median")),

        "rsa_rel_median": _safe_float(d.get("rsa_rel_median")),
        "rsa_rel_mean": _safe_float(d.get("rsa_rel_mean")),
        "rsa_rel_p75": _safe_float(d.get("rsa_rel_p75")),
        "rsa_rel_p95": _safe_float(d.get("rsa_rel_p95")),
        "rsa_rel_nvox": _safe_int(d.get("rsa_rel_nvox")),

        "prov_path": prov_path,
        "mtime": d.get("_mtime", np.nan),
    }
    return row


def dedupe_newest(df: pd.DataFrame, key_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    df2 = df.sort_values("mtime", ascending=False).copy()
    keep_idx = df2.drop_duplicates(subset=key_cols, keep="first").index
    kept = df.loc[keep_idx].copy()
    dropped = df.drop(index=keep_idx).copy()
    return kept, dropped


def choose_winner(group: pd.DataFrame) -> pd.Series:
    g = group.copy()
    g = g[np.isfinite(g["rsa_rel_median"].to_numpy(dtype=float))]
    if g.empty:
        return pd.Series({"winner": np.nan, "winner_value": np.nan})

    g = g.sort_values(
        by=["rsa_rel_median", "rsa_rel_p75", "rsa_rel_mean", "rsa_rel_nvox"],
        ascending=[False, False, False, False],
        kind="mergesort"
    )
    best = g.iloc[0]
    return pd.Series({"winner": best["approach"], "winner_value": best["rsa_rel_median"]})


def wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def fig_to_base64(fig) -> str:
    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("ascii")


def make_heatmap(df: pd.DataFrame, title: str) -> Optional[str]:
    if df.empty:
        return None
    subj_order = sorted(df["subject"].unique().tolist())
    app_order  = sorted(df["approach"].unique().tolist())

    M = np.full((len(subj_order), len(app_order)), np.nan, float)
    for i, s in enumerate(subj_order):
        for j, a in enumerate(app_order):
            v = df.loc[(df["subject"] == s) & (df["approach"] == a), "rsa_rel_median"]
            if len(v):
                M[i, j] = float(v.iloc[0])

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(app_order)), max(7, 0.28 * len(subj_order))))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(app_order)))
    ax.set_xticklabels(app_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(subj_order)))
    ax.set_yticklabels(subj_order)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("rsa_rel_median")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_boxplot(df: pd.DataFrame, title: str) -> Optional[str]:
    if df.empty:
        return None
    apps = sorted(df["approach"].unique().tolist())
    data = [df.loc[df["approach"] == a, "rsa_rel_median"].to_numpy(dtype=float) for a in apps]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(apps)), 5))
    ax.boxplot(data, labels=apps, showfliers=False)
    ax.set_ylabel("rsa_rel_median")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_scatter_rel_vs_nvox(df: pd.DataFrame, title: str) -> Optional[str]:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for a in sorted(df["approach"].unique()):
        sub = df[df["approach"] == a]
        ax.scatter(sub["rsa_rel_nvox"].to_numpy(), sub["rsa_rel_median"].to_numpy(), label=a, alpha=0.7)
    ax.set_xlabel("rsa_rel_nvox (eligible voxels)")
    ax.set_ylabel("rsa_rel_median")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_win_count_bar(winners: pd.DataFrame, title: str) -> Optional[str]:
    if winners.empty:
        return None
    vc = winners["winner"].value_counts(dropna=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(vc.index)), 4))
    ax.bar(vc.index.astype(str), vc.values.astype(int))
    ax.set_ylabel("#subjects won")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig_to_base64(fig)


def pairwise_wilcoxon(df: pd.DataFrame, value_col: str = "rsa_rel_median") -> pd.DataFrame:
    apps = sorted(df["approach"].unique().tolist())

    piv = df.pivot_table(index="subject", columns="approach", values=value_col, aggfunc="first")
    rows = []
    for i in range(len(apps)):
        for j in range(i + 1, len(apps)):
            a, b = apps[i], apps[j]
            if a not in piv.columns or b not in piv.columns:
                continue
            x = piv[a]
            y = piv[b]
            m = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
            if m.sum() < 5:
                continue
            dx = (x[m] - y[m]).to_numpy(dtype=float)
            mean_dx = float(np.mean(dx))
            med_dx  = float(np.median(dx))
            n = int(m.sum())

            p = np.nan
            stat = np.nan
            if HAVE_SCIPY:
                try:
                    stat, p = wilcoxon(dx, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
                    stat = float(stat)
                    p = float(p)
                except Exception:
                    pass

            rows.append({
                "A_minus_B": f"{a} - {b}",
                "n_pairs": n,
                "mean_delta": mean_dx,
                "median_delta": med_dx,
                "wilcoxon_stat": stat,
                "wilcoxon_p": p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["wilcoxon_p", "mean_delta"], ascending=[True, False])
    return out


def df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>(empty)</em></p>"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows).copy()
        note = f"<p><em>Showing first {max_rows} rows (table truncated).</em></p>"
    else:
        note = ""
    return note + d.to_html(index=False, escape=True, border=0)


def _get_scan_roots(root_abs: str, approaches: List[str]) -> Tuple[List[str], List[str]]:
    """
    Build scan roots as <root>/<approach>, but only include directories that exist.
    Returns (scan_roots, missing_approach_dirs).
    """
    scan_roots = []
    missing = []

    for a in approaches:
        if not a:
            continue

        # Special case: allow alt on-disk folder for D legacy
        if a == "D_up3x_legacy":
            cands = [
                os.path.join(root_abs, "D_up3x_legacy"),
                os.path.join(root_abs, "D_up3x_legacy_upsampled_slicetime_change"),
            ]
        else:
            cands = [os.path.join(root_abs, a)]

        chosen = None
        for c in cands:
            if os.path.isdir(c):
                chosen = c
                break

        if chosen is None:
            missing.append(a)
        else:
            scan_roots.append(chosen)

    return scan_roots, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder. Only scans <root>/<approach> for the requested --approaches.")
    ap.add_argument("--outdir", required=True, help="Where to write the single HTML report (+ optional CSVs)")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subject list; if omitted, inferred from found provenance.")
    ap.add_argument("--approaches", nargs="+", default=APP_DEFAULT)
    ap.add_argument("--datatypes", nargs="+", default=DT_DEFAULT)
    ap.add_argument("--radius", type=int, default=0, help="If >0, keep only this radius.")
    ap.add_argument("--metric", default="", help="If non-empty, keep only this metric (e.g., crossnobis).")
    ap.add_argument("--min-neigh", type=int, default=0, help="If >0, keep only this min_neigh.")
    ap.add_argument("--prefer-newest", action="store_true",
                    help="If multiple reruns exist per key, keep newest and write dropped_duplicates.csv.")
    ap.add_argument("--write-long-csv", action="store_true",
                    help="Also write reliability_long.csv (recommended for provenance).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    root_abs = os.path.abspath(args.root)

    # Map canonical approach label -> list of on-disk folder names to treat as that label
    alias_dirs = {
        "D_up3x_legacy": ["D_up3x_legacy_upsampled_slicetime_change"],
    }

    scan_roots, missing = _get_scan_roots(root_abs, args.approaches)

    print("[INFO] Requested approaches:")
    for a in args.approaches:
        print("  -", a)

    print("\n[INFO] Scan roots actually used:")
    for r in scan_roots:
        print("  -", r)

    if missing:
        print("\n[WARN] These approach folders were requested but not found under --root:")
        for a in missing:
            print("  -", a)
        print("[WARN] Continuing with the ones that exist.\n")

    prov_paths = find_provenance_jsons(scan_roots)
    if not prov_paths:
        raise SystemExit(f"No *_provenance.json found under the requested approach roots: {scan_roots}")

    prov_dicts = [load_one_prov(p) for p in prov_paths]
    prov_dicts = [d for d in prov_dicts if d is not None]

    rows = [prov_to_row(d, args.approaches, alias_dirs) for d in prov_dicts]
    df = pd.DataFrame(rows)

    # filters
    df = df[df["approach"].isin(args.approaches)].copy()
    df = df[df["datatype"].isin(args.datatypes)].copy()
    if args.radius and args.radius > 0:
        df = df[df["radius"] == int(args.radius)].copy()
    if args.metric:
        df = df[df["metric"] == args.metric].copy()
    if args.min_neigh and args.min_neigh > 0:
        df = df[df["min_neigh"] == int(args.min_neigh)].copy()

    if args.subjects:
        wanted = set(_normalize_sub(s) for s in args.subjects)
        df = df[df["subject"].isin(wanted)].copy()

    if df.empty:
        raise SystemExit("After filtering, no provenance rows remain. Check filters and root path.")

    dropped = pd.DataFrame()
    if args.prefer_newest:
        key_cols = ["subject", "datatype", "approach", "radius", "metric", "min_neigh"]
        df, dropped = dedupe_newest(df, key_cols=key_cols)
        if not dropped.empty:
            dropped.to_csv(os.path.join(args.outdir, "dropped_duplicates.csv"), index=False)

    if args.write_long_csv:
        df.to_csv(os.path.join(args.outdir, "reliability_long.csv"), index=False)

    subjects = sorted(df["subject"].unique().tolist())
    if args.subjects:
        subjects = sorted(set(_normalize_sub(s) for s in args.subjects))

    # expected combos / missing
    expected = []
    for s in subjects:
        for dt in args.datatypes:
            for a in args.approaches:
                expected.append((s, dt, a))
    exp_df = pd.DataFrame(expected, columns=["subject", "datatype", "approach"])
    got_df = df[["subject", "datatype", "approach"]].drop_duplicates()
    missing_df = exp_df.merge(got_df, on=["subject", "datatype", "approach"], how="left", indicator=True)
    missing_df = missing_df[missing_df["_merge"] == "left_only"].drop(columns="_merge")

    # IMPORTANT: no gating filter — use all rows
    df_sum = df.copy()

    winners = (df_sum.groupby(["subject", "datatype"], dropna=False)
                    .apply(choose_winner)
                    .reset_index())

    win_counts = (winners.dropna(subset=["winner"])
                        .groupby(["datatype", "winner"])
                        .size()
                        .reset_index(name="n_wins")
                        .rename(columns={"winner": "approach"}))

    df_rank = df_sum.copy()
    df_rank["rank"] = (df_rank.groupby(["subject", "datatype"])["rsa_rel_median"]
                              .rank(ascending=False, method="average"))

    agg = (df_rank.groupby(["datatype", "approach"], dropna=False)
                 .agg(
                     n_subjects=("subject", "nunique"),
                     n_rows=("rsa_rel_median", "size"),
                     mean_median=("rsa_rel_median", "mean"),
                     median_median=("rsa_rel_median", "median"),
                     p25_median=("rsa_rel_median", lambda v: float(np.nanpercentile(v, 25)) if np.isfinite(v).any() else np.nan),
                     p75_median=("rsa_rel_median", lambda v: float(np.nanpercentile(v, 75)) if np.isfinite(v).any() else np.nan),
                     mean_rank=("rank", "mean"),
                     mean_nvox=("rsa_rel_nvox", "mean"),
                     passed_gate_rate=("passed_gate", lambda v: float(np.mean(np.asarray(v, bool))) if len(v) else np.nan),
                 )
                 .reset_index())

    wrows = []
    for (dt, a), g in df_sum.groupby(["datatype", "approach"], dropna=False):
        wrows.append({
            "datatype": dt,
            "approach": a,
            "wmean_median_by_nvox": wmean(g["rsa_rel_median"], g["rsa_rel_nvox"]),
            "wmean_mean_by_nvox": wmean(g["rsa_rel_mean"], g["rsa_rel_nvox"]),
        })
    wdf = pd.DataFrame(wrows)
    agg = agg.merge(wdf, on=["datatype", "approach"], how="left")
    agg = agg.merge(win_counts, on=["datatype", "approach"], how="left")
    agg["n_wins"] = agg["n_wins"].fillna(0).astype(int)

    pw_tables = []
    for dt in sorted(df_sum["datatype"].unique().tolist()):
        subdf = df_sum[df_sum["datatype"] == dt].copy()
        pw = pairwise_wilcoxon(subdf, value_col="rsa_rel_median")
        pw.insert(0, "datatype", dt)
        pw_tables.append(pw)
    pw_all = pd.concat(pw_tables, axis=0, ignore_index=True) if pw_tables else pd.DataFrame()

    rec_lines = []
    for dt in sorted(df_sum["datatype"].unique().tolist()):
        a = agg[agg["datatype"] == dt].copy()
        if a.empty:
            continue
        a = a.sort_values(
            ["mean_median", "wmean_median_by_nvox", "n_wins"],
            ascending=[False, False, False]
        )
        best = a.iloc[0]
        rec_lines.append(
            f"{dt}: {best['approach']}  "
            f"(mean_median={best['mean_median']:.4f}, "
            f"wmean_median_by_nvox={best['wmean_median_by_nvox']:.4f}, "
            f"wins={int(best['n_wins'])}/{int(best['n_subjects'])})"
        )

    plot_blocks = []
    for dt in sorted(df_sum["datatype"].unique().tolist()):
        subdf = df_sum[df_sum["datatype"] == dt].copy()

        hm = make_heatmap(subdf[["subject", "approach", "rsa_rel_median"]], title=f"rsa_rel_median heatmap — {dt}")
        bp = make_boxplot(subdf, title=f"rsa_rel_median by approach — {dt}")
        sc = make_scatter_rel_vs_nvox(subdf, title=f"rsa_rel_median vs eligible voxels — {dt}")
        wc = make_win_count_bar(winners[winners["datatype"] == dt], title=f"Winner counts — {dt}")

        def img_html(b64: Optional[str]) -> str:
            if not b64:
                return "<p><em>(plot unavailable)</em></p>"
            return f'<img src="data:image/png;base64,{b64}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;" />'

        plot_blocks.append(f"""
        <h2>Plots — {dt}</h2>
        <h3>Subject × approach heatmap</h3>
        {img_html(hm)}
        <h3>Boxplot</h3>
        {img_html(bp)}
        <h3>Reliability vs coverage</h3>
        {img_html(sc)}
        <h3>Win counts</h3>
        {img_html(wc)}
        """)

    piv_med = df_sum.pivot_table(index=["subject", "datatype"], columns="approach", values="rsa_rel_median", aggfunc="first").reset_index()

    css = """
    body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px; color: #111; }
    h1 { margin-top: 0; }
    .meta { background: #f6f8fa; padding: 12px 14px; border-radius: 10px; border: 1px solid #e5e7eb; }
    .callout { background: #fff7ed; padding: 12px 14px; border-radius: 10px; border: 1px solid #fed7aa; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0 18px 0; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 13px; }
    th { background: #f3f4f6; text-align: left; }
    .small { font-size: 12px; color: #374151; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
    """

    report_path = os.path.join(args.outdir, "SUMMARY.html")
    with open(report_path, "w") as f:
        f.write(f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Searchlight RSA Reliability Summary</title>
<style>{css}</style>
</head>
<body>
<h1>Searchlight RSA split-half reliability summary</h1>

<div class="meta">
<div><strong>Root scanned:</strong> <code>{args.root}</code></div>
<div><strong>Outdir:</strong> <code>{args.outdir}</code></div>
<div><strong>Rows included:</strong> {len(df_sum)} (includes passed_gate=False; no gating filter)</div>
<div><strong>Unique subjects:</strong> {len(subjects)}</div>
<div class="small">
Filter: datatype in {args.datatypes}, approach in {args.approaches},
radius={args.radius or "ANY"}, metric={args.metric or "ANY"}, min_neigh={args.min_neigh or "ANY"}.
prefer_newest={bool(args.prefer_newest)}.
SciPy Wilcoxon available={HAVE_SCIPY}.
</div>
</div>

<h2>Recommendation (per datatype)</h2>
<div class="callout">
<p><strong>Top approach by mean of per-subject medians</strong> (tie-break: voxel-weighted median, wins):</p>
<ul>
{''.join(f'<li><code>{line}</code></li>' for line in rec_lines) if rec_lines else '<li><em>(no recommendation available)</em></li>'}
</ul>
<p class="small">
Interpretation: mean_median is the average across subjects of each subject’s <code>rsa_rel_median</code>.
wmean_median_by_nvox penalizes approaches that look good only in a tiny number of voxels.
</p>
</div>

<h2>Approach summary by datatype</h2>
{df_to_html_table(agg.sort_values(["datatype","mean_median"], ascending=[True, False]), max_rows=200)}

<h2>Pairwise within-subject Wilcoxon tests (rsa_rel_median)</h2>
<p class="small">{'P-values are NaN because SciPy is unavailable.' if not HAVE_SCIPY else ''}</p>
{df_to_html_table(pw_all, max_rows=400)}

<h2>Win counts by datatype</h2>
{df_to_html_table(win_counts.sort_values(["datatype","n_wins"], ascending=[True, False]), max_rows=200)}

<h2>Per-subject winners</h2>
{df_to_html_table(winners.sort_values(["datatype","subject"]), max_rows=300)}

<h2>Per-subject medians pivot (rsa_rel_median)</h2>
{df_to_html_table(piv_med.sort_values(["datatype","subject"]), max_rows=300)}

<h2>Missing expected combos</h2>
{df_to_html_table(missing_df.sort_values(["datatype","subject","approach"]), max_rows=400)}

{''.join(plot_blocks)}

<hr />
<p class="small">
Generated by summarize_searchlight_reliability_ONEFILE.py.
</p>
</body>
</html>
""")

    print(f"[OK] Wrote single report:\n  {report_path}")
    if args.prefer_newest and (not dropped.empty):
        print(f"[OK] Wrote dropped duplicates CSV:\n  {os.path.join(args.outdir, 'dropped_duplicates.csv')}")
    if args.write_long_csv:
        print(f"[OK] Wrote long table CSV:\n  {os.path.join(args.outdir, 'reliability_long.csv')}")


if __name__ == "__main__":
    main()
