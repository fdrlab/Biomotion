#!/usr/bin/env python3
"""
model_rdms.py

Builds three model predictors for RSA, as in Castaldi et al. (2019):

  bmvsc = [...]
  fvp   = [...]
  gtvfp = [...]
  X = [bmvsc fvp gtvfp]
  X = X - mean(X)           (column-wise)
  X = X ./ std(X)           (column-wise; MATLAB std default => ddof=1)
  R0 = corrcoef(X)
  VIF = diag(inv(R0))

COND_ORDER =
    "happy_females", "happy_males", "angry_females",
    "angry_males", "neutral_females", "scrambled"
    
And saves:
  - RAW 6x6 model RDMs (raw weights)
  - MATLAB-standardized 6x6 model RDMs (mean-centered + /std)

IMPORTANT:
- The 15-long vectors are assumed to be in pdist / upper-triangle order for K=6:
  (1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6),(5,6)
- That order matches np.triu_indices(6,1) for a fixed condition order.


Reference:
Castaldi, E., Piazza, M., Dehaene, S., Vignaud, A., & Eger, E. (2019). Attentional
amplification of neural codes for number independent of other quantities along
the dorsal visual stream. eLife, 8, e45160. https://doi.org/10.7554/elife.45160
"""

import os
import argparse
import numpy as np


K = 6
IU = np.triu_indices(K, 1)  # length 15


def vec15_to_rdm6(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    if v.size != IU[0].size:
        raise ValueError(f"Expected vec15 length {IU[0].size}, got {v.size}")
    M = np.zeros((K, K), float)
    M[IU] = v
    M[(IU[1], IU[0])] = v
    np.fill_diagonal(M, 0.0)  # keep "dissimilarity" convention
    return M


def zstandardize_matlab(X: np.ndarray) -> np.ndarray:
    """
    MATLAB:
      X = X - mean(X)
      X = X ./ repmat(std(X), size(X,1), 1)
    MATLAB std default uses N-1 (ddof=1).
    """
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return Xc / sd


def vif_from_corr(X: np.ndarray) -> np.ndarray:
    """VIF = diag(inv(corrcoef(X)))"""
    R0 = np.corrcoef(X, rowvar=False)
    return np.diag(np.linalg.inv(R0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory for saved .npy model RDMs.")
    ap.add_argument("--prefix", default="model6_castaldi", help="Filename prefix.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Exact MATLAB vectors ---
    bmvsc = np.array([0, 0, 0, 0, 1,
                      0, 0, 0, 1,
                      0, 0, 1,
                      0, 1,
                      1], dtype=float)

    fvp   = np.array([0, 0.75, 0.75, 0.5, 0.5,
                      0.75, 0.75, 0.5, 0.5,
                      0, 0.5, 0.5,
                      0.5, 0.5,
                      0.5], dtype=float)

    gtvfp = np.array([0, 0, 0, 1.25, 0.5,
                      0, 0, 1.25, 0.5,
                      0, 1.25, 0.5,
                      1.25, 0.5,
                      0.5], dtype=float)

    names = ["bmvsc", "fvp", "gtvfp"]
    X_raw = np.column_stack([bmvsc, fvp, gtvfp])  # 15x3

    # --- MATLAB standardization + VIF ---
    X_std = zstandardize_matlab(X_raw)
    R0 = np.corrcoef(X_std, rowvar=False)
    V = vif_from_corr(X_std)

    np.set_printoptions(precision=6, suppress=True)
    print("corrcoef(X_std) =")
    print(R0)
    print("\nVIF = diag(inv(corrcoef(X_std))) =")
    print(V)

    # --- Save RAW and STD 6x6 RDMs ---
    out = []
    for j, nm in enumerate(names):
        M_raw = vec15_to_rdm6(X_raw[:, j])
        M_std = vec15_to_rdm6(X_std[:, j])

        p_raw = os.path.join(args.outdir, f"{args.prefix}_{nm}_RAW.npy") # for raw model RDM visualization
        p_std = os.path.join(args.outdir, f"{args.prefix}_{nm}_STD.npy") # used in analysis

        np.save(p_raw, M_raw.astype(float))
        np.save(p_std, M_std.astype(float))
        out.extend([p_raw, p_std])

    # Save correlation + VIF as plain text too (handy for provenance)
    np.savetxt(os.path.join(args.outdir, f"{args.prefix}_corrcoef_Xstd.tsv"), R0, delimiter="\t")
    np.savetxt(os.path.join(args.outdir, f"{args.prefix}_vif.tsv"), V.reshape(1, -1), delimiter="\t")

    print("\n[OK] Saved:")
    for p in out:
        print(" ", p)
    print(" ", os.path.join(args.outdir, f"{args.prefix}_corrcoef_Xstd.tsv"))
    print(" ", os.path.join(args.outdir, f"{args.prefix}_vif.tsv"))


if __name__ == "__main__":
    main()
