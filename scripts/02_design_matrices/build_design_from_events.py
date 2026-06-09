#!/usr/bin/env python3
"""
Build condition-level design matrices from events.

- Time grid: dt = TR / upsample (upsample>=1). Each event is placed in the bin
  [k*dt, (k+1)*dt) that contains its onset (i.e., floor(onset/dt)).
- Modeled window: T_seconds = (n_volumes_original - n_trim) * TR
- Columns: fixed canonical order (see CANON).
- No "near-end" culling. Only events with onset ∈ [0, T_seconds) are kept.
- Multiple events landing in the same bin accumulate amplitude (2, 3, ...).
- Writes per-run .npy and .json; also writes a run-level CSV summary and a
  global mapping JSON (identity under CANON).

Usage:
python build_design_from_events.py \
  --events-root  /Volumes/T7/BIDS_data/derivatives/events_original \
  --bold-root    /Volumes/T7/preprocessed \
  --bold-pattern '{subject}/func/{subject}_task-PLDs_run-{run02}_space-MNI152NLin2009cAsym_res-1.7_desc-preproc_bold.nii.gz' \
  --out-root     /Volumes/T7/preprocessed/design_matrices_new \
  --tr 1.9 --upsample 1 --n-trim 5 --strict
"""

import os, json, argparse, math
import numpy as np
import pandas as pd
import nibabel as nb

CANON = ['happy_females','happy_males','angry_females','angry_males','neutral_females','scrambled']
RUNS  = ['01','02']

def norm_subject_id(s): return s if str(s).startswith('sub-') else f"sub-{s}"

def events_path(root, subj, run02):
    return os.path.join(root, subj, 'func', f'{subj}_task-PLDs_run-{run02}_events.tsv')

def bold_path(root, patt, subj, run02):
    return os.path.join(root, patt.format(subject=subj, run02=run02))

def load_subjects(kept_csv, events_root):
    if kept_csv and os.path.isfile(kept_csv):
        df = pd.read_csv(kept_csv)
        subs = sorted(norm_subject_id(s) for s in df['subject'].astype(str))
    else:
        subs = sorted(d for d in os.listdir(events_root) if d.startswith('sub-') and os.path.isdir(os.path.join(events_root, d)))
    return subs

def build_design(ev_df: pd.DataFrame, T_seconds: float, dt: float, labels: list[str]):
    """
    Return X (n_bins × K), counts, and collision stats.
    Events outside [0, T_seconds) are ignored. Missing labels allowed; unknown labels error.
    """
    n_bins = int(math.floor((T_seconds + 1e-9) / dt))
    K = len(labels)
    X = np.zeros((n_bins, K), dtype=np.float32)

    uniq = sorted(map(str, ev_df['trial_type'].dropna().unique()))
    extra = [c for c in uniq if c not in labels]
    if extra:
        raise ValueError(f"Unexpected labels in events: {extra} (expected subset of {labels})")

    counts_ev = {c: 0 for c in labels}

    # Bin events by floor (bin that *contains* the onset)
    for _, r in ev_df.iterrows():
        lab = str(r['trial_type'])
        if lab not in labels:
            continue  # already guarded by 'extra', but be safe
        t = float(r['onset'])
        if not (0.0 <= t < T_seconds):
            continue
        b = int((t + 1e-9) // dt)
        if 0 <= b < n_bins:
            X[b, labels.index(lab)] += 1.0
            counts_ev[lab] += 1

    # Totals from the design
    impulses_per_col  = X.sum(axis=0).astype(int).tolist()
    nonzero_bins_per_col = (X > 0).sum(axis=0).astype(int).tolist()
    events_vec = [int(counts_ev[c]) for c in labels]
    # Sanity: number of impulses equals number of events per label
    if events_vec != impulses_per_col:
        raise AssertionError(
            "events vs design mismatch in impulse totals.\n"
            f"labels={labels}\n"
            f"events (counts)={events_vec}\n"
            f"design (impulses sum)={impulses_per_col}"
        )

    # Collisions (multiple events in same bin for a label) are allowed
    collisions = [impulses_per_col[i] - nonzero_bins_per_col[i] for i in range(K)]

    return X, counts_ev, impulses_per_col, nonzero_bins_per_col, collisions, n_bins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events-root', required=True)
    ap.add_argument('--bold-root',   required=True)
    ap.add_argument('--bold-pattern', required=True)
    ap.add_argument('--out-root',    required=True)
    ap.add_argument('--kept-csv',    default=None)
    ap.add_argument('--tr',          type=float, required=True)
    ap.add_argument('--upsample',    type=int, default=1, help='Temporal upsample factor (1 = TR grid).')
    ap.add_argument('--n-trim',      type=int, default=5, help='Dummy TRs removed from the *start* of BOLD.')
    ap.add_argument('--tol-tr',      type=float, default=0.02, help='Allowed |TR_hdr - TR| in seconds before warning.')
    ap.add_argument('--strict',      action='store_true', help='Fail if any run is missing or any design couldn’t be built.')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    dt = args.tr / max(1, args.upsample)

    subs = load_subjects(args.kept_csv, args.events_root)
    print(f"[info] subjects={len(subs)} | TR={args.tr:.6f}s | upsample={args.upsample} (dt={dt:.6f}s) | n_trim={args.n_trim}")

    # Summary rows and error bookkeeping
    summary_rows = []
    n_missing = 0
    n_failed  = 0

    # Global mapping json (identity under CANON)
    map_json_path = os.path.join(args.out_root, "design_order_map_enforced.json")
    global_map = {
        "columns_assumed": CANON,
        "label_to_col": {lab: i for i, lab in enumerate(CANON)},
        "upsample": args.upsample,
        "tr": args.tr,
        "dt": dt,
        "n_trim": args.n_trim,
        "notes": "Columns are enforced to canonical order. Mapping is identity."
    }

    for subj in subs:
        for run02 in RUNS:
            ev_p = events_path(args.events_root, subj, run02)
            b_p  = bold_path(args.bold_root, args.bold_pattern, subj, run02)

            if not os.path.isfile(ev_p) or not os.path.isfile(b_p):
                print(f"[skip] {subj} run-{run02}: missing file(s): "
                      f"{'events' if not os.path.isfile(ev_p) else ''} {'bold' if not os.path.isfile(b_p) else ''}".strip())
                n_missing += 1
                continue

            # Load BOLD header to get original length and TR
            try:
                bold_img = nb.load(b_p)
                n_vol_orig = int(bold_img.shape[3])
                tr_hdr = float(bold_img.header.get_zooms()[3])
            except Exception as e:
                print(f"[fail] {subj} run-{run02}: cannot read BOLD header: {e}")
                n_failed += 1
                continue

            if abs(tr_hdr - args.tr) > args.tol_tr:
                print(f"[warn] {subj} run-{run02}: TR header={tr_hdr:.6f}s differs from --tr={args.tr:.6f}s (> {args.tol_tr}s). Using --tr.")
            if n_vol_orig <= args.n_trim:
                print(f"[skip] {subj} run-{run02}: n_vol={n_vol_orig} <= n_trim={args.n_trim}")
                n_missing += 1
                continue

            n_vol = n_vol_orig - args.n_trim
            T_seconds = n_vol * args.tr
            target_bins = int(math.floor((T_seconds + 1e-9) / dt))
            print(f"[{subj} {run02}] nTR={n_vol_orig} -> modeled={n_vol} | T={T_seconds:.3f}s | bins={target_bins}")

            # Load events
            try:
                ev = pd.read_csv(ev_p, sep='\t')
            except Exception as e:
                print(f"[fail] {subj} run-{run02}: cannot read events: {e}")
                n_failed += 1
                continue

            if 'trial_type' not in ev.columns or 'onset' not in ev.columns:
                print(f"[fail] {subj} run-{run02}: events missing 'trial_type'/'onset'")
                n_failed += 1
                continue

            # Keep only events in the post-trim window
            ev = ev.loc[ev['onset'].astype(float) >= 0].reset_index(drop=True)

            try:
                X, counts_ev, impulses, nonzero_bins, collisions, n_bins = build_design(ev, T_seconds, dt, CANON)
            except Exception as e:
                print(f"[fail] {subj} run-{run02}: build_design error: {e}")
                n_failed += 1
                continue

            # Guard small float mismatches: trim/pad to target_bins
            if X.shape[0] != target_bins:
                if X.shape[0] > target_bins:
                    X = X[:target_bins, :]
                else:
                    X = np.vstack([X, np.zeros((target_bins - X.shape[0], X.shape[1]), dtype=X.dtype)])

            # Save matrix
            out_fname = f"{subj}_task-PLDs_run-{run02}_desc-origevents_desc-conditionlevel_design_matrix.npy"
            out_path  = os.path.join(args.out_root, out_fname)
            np.save(out_path, X)

            # Sidecar JSON
            meta = dict(
                subject=subj, run=run02,
                tr=args.tr, upsample=args.upsample, dt=dt,
                n_trim=args.n_trim, bold_path=b_p, events_path=ev_p,
                n_tr_original=int(n_vol_orig), n_tr_modeled=int(n_vol),
                T_seconds=float(T_seconds), n_timepoints=int(X.shape[0]),
                columns=CANON,
                counts_events={k:int(v) for k,v in counts_ev.items()},
                counts_design_impulses={CANON[i]: int(impulses[i]) for i in range(len(CANON))},
                counts_design_nonzero_bins={CANON[i]: int(nonzero_bins[i]) for i in range(len(CANON))},
                collisions_per_label={CANON[i]: int(collisions[i]) for i in range(len(CANON))}
            )
            with open(out_path.replace('.npy', '.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            # Row for summary CSV
            row = {
                "subject": subj, "run": run02,
                "n_tr_original": n_vol_orig, "n_tr_modeled": n_vol,
                "dt": dt, "bins": int(X.shape[0]),
                "events_total": int(sum(counts_ev.values())),
                "impulses_total": int(sum(impulses)),
                "collisions_total": int(sum(collisions))
            }
            # Per-label fields
            for i, lab in enumerate(CANON):
                row[f"ev_{lab}"]  = int(counts_ev[lab])
                row[f"imp_{lab}"] = int(impulses[i])
                row[f"nz_{lab}"]  = int(nonzero_bins[i])
                row[f"col_{lab}"] = int(collisions[i])
            summary_rows.append(row)

            print(f"  -> wrote {out_path} shape={X.shape} | events={row['events_total']} | collisions={row['collisions_total']}")

    # Write global mapping JSON (identity)
    with open(map_json_path, 'w') as f:
        json.dump(global_map, f, indent=2)
    print(f"[OK] mapping: {map_json_path}")

    # Write summary CSV
    summary_csv = os.path.join(args.out_root, "design_build_summary.csv")
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_csv, index=False)
        print(f"[OK] summary: {summary_csv} (rows={len(df)})")
    else:
        # still write an empty file for reproducibility
        pd.DataFrame(columns=["subject","run"]).to_csv(summary_csv, index=False)
        print(f"[WARN] no designs built; wrote empty {summary_csv}")

    if args.strict and (n_missing > 0 or n_failed > 0):
        raise SystemExit(f"[ERROR] strict mode: missing={n_missing}, failed={n_failed}")

if __name__ == "__main__":
    main()
