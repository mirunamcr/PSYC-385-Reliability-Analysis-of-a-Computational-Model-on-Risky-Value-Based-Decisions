
#  Split-Half Across the Sample (Pooled) with Bootstrap Correlations
#
# For each split:
#   - Shuffle all trials (all participants) split into two halves
#   - For each half, bootstrap the half's trials 50 times, fit pooled alpha/beta/mu
#   - For each parameter, compute Pearson r between the  bootstrapped estimates
#   - Store one correlation per parameter per split
# Repeat 100 or 200 splits, then plot histograms of those correlations (looking at population-level)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import pearsonr

DATA_PATH = "/Users/miru/Documents/PSYC 385 Thesis/Phase2_data (Raw Otto).csv"
ALLOWED_P = {0.5, 0.9, 1.0}
BOUNDS = [(0.0, 1.0), (0.0, 1.0), (0.0, 10.0)]  # alpha, beta, mu
N_SPLITS = 200         # number of split-halves
BOOT_REPS = 50          # bootstrap refits per half
MIN_TRIALS = 1000       # total trials needed to proceed
RNG = np.random.default_rng(0)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df[df["P_Gamble"].isin(ALLOWED_P)].reset_index(drop=True)
    return df


def scale_payoffs(df):
    scale = np.nanmax(np.concatenate([df["A_Gamble"].values, df["A_Certain"].values]))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    df = df.copy()
    df["A_Gamble_scaled"] = df["A_Gamble"] / scale
    df["A_Certain_scaled"] = df["A_Certain"] / scale
    return df


def subjective_values(p, A_g, A_c, alpha, beta):
    SV_r = (p ** (1 - beta)) * (A_g ** alpha)
    SV_c = (A_c ** alpha)
    return SV_r, SV_c


def nll_joint(params, p, Ag, Ac, choices):
    alpha, beta, mu = params
    SV_r, SV_c = subjective_values(p, Ag, Ac, alpha, beta)
    dSV = SV_r - SV_c
    p_gamble = expit(mu * dSV)
    p_gamble = np.clip(p_gamble, 1e-9, 1 - 1e-9)
    return -np.sum(choices * np.log(p_gamble) + (1 - choices) * np.log(1 - p_gamble))


def fit_safe(p, Ag, Ac, choices, x0=(0.5, 0.5, 0.5)):
    try:
        res = minimize(
            nll_joint,
            x0=x0,
            args=(p, Ag, Ac, choices),
            bounds=BOUNDS,
            method="L-BFGS-B",
            options={"maxiter": 800},
        )
        if res.success and np.all(np.isfinite(res.x)):
            return np.array(res.x, dtype=float)
    except Exception:
        pass
    return np.array([np.nan, np.nan, np.nan], dtype=float)


def bootstrap_half(df_half, boot_reps=BOOT_REPS):
    alpha_list, beta_list, mu_list = [], [], []
    n = len(df_half)
    if n < 2:
        return alpha_list, beta_list, mu_list
    for _ in range(boot_reps):
        idx = RNG.integers(0, n, size=n)
        samp = df_half.iloc[idx]
        params = fit_safe(
            samp["P_Gamble"].values,
            samp["A_Gamble_scaled"].values,
            samp["A_Certain_scaled"].values,
            samp["Gamble"].astype(int).values,
        )
        alpha_list.append(params[0]); beta_list.append(params[1]); mu_list.append(params[2])
    return alpha_list, beta_list, mu_list


def safe_corr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan
    return pearsonr(x[mask], y[mask])[0]


def main():
    df = scale_payoffs(load_data())
    total_trials = len(df)
    if total_trials < MIN_TRIALS:
        raise ValueError("Not enough total trials for population-level split-half.")

    alpha_corrs = []
    beta_corrs = []
    mu_corrs = []

    for _ in range(N_SPLITS):
        perm = RNG.permutation(total_trials)
        half = total_trials // 2
        idx1 = perm[:half]
        idx2 = perm[half:half+half]

        df_h1 = df.iloc[idx1]
        df_h2 = df.iloc[idx2]

        # bootstrap within each half to get multiple pooled estimates
        a1, b1, m1 = bootstrap_half(df_h1)
        a2, b2, m2 = bootstrap_half(df_h2)

        alpha_corrs.append(safe_corr(a1, a2))
        beta_corrs.append(safe_corr(b1, b2))
        mu_corrs.append(safe_corr(m1, m2))

    # Plot histograms of correlations (alpha, beta only)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    alpha_clean = [v for v in alpha_corrs if np.isfinite(v)]
    beta_clean = [v for v in beta_corrs if np.isfinite(v)]

    # summary table (mean, sd)
    summary = pd.DataFrame(
        {
            "parameter": ["alpha", "beta"],
            "mean_r": [np.nanmean(alpha_clean) if alpha_clean else np.nan,
                       np.nanmean(beta_clean) if beta_clean else np.nan],
            "sd_r": [np.nanstd(alpha_clean, ddof=1) if len(alpha_clean) > 1 else np.nan,
                     np.nanstd(beta_clean, ddof=1) if len(beta_clean) > 1 else np.nan],
            "n": [len(alpha_clean), len(beta_clean)],
            "N_SPLITS": [N_SPLITS, N_SPLITS],
            "BOOT_REPS": [BOOT_REPS, BOOT_REPS],
        }
    )
    out_path = "/Users/miru/Documents/PSYC 385 Thesis/split-half across sample.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nCorrelation summary saved to {out_path}")

    axs[0].hist(alpha_clean, bins=np.linspace(-1, 1, 25),
                color="skyblue", edgecolor="k")
    axs[0].set_title(f"Population Split-Half Correlation (Alpha)\nN_SPLITS={N_SPLITS}, BOOT_REPS={BOOT_REPS}")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(beta_clean, bins=np.linspace(-1, 1, 25),
                color="salmon", edgecolor="k")
    axs[1].set_title(f"Population Split-Half Correlation (Beta)\nN_SPLITS={N_SPLITS}, BOOT_REPS={BOOT_REPS}")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlabel("Correlation")

    plt.tight_layout()
    plt.show()
    plt.savefig("/Users/miru/Documents/PSYC 385 Thesis/split-half across sample (200 splits).png")


if __name__ == "__main__":
    main()
