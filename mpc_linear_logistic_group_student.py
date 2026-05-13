
"""
INTE2669 Assignment 2 - Group scaffold
From trusted plaintext linear regression to two-server MPC training,
with a logistic inference extension.

This file is the student scaffold version.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Replace these with the group members' student IDs in the student version.
GROUP_MEMBER_IDS = [4115252, 4115477, 4115486, 4187739]

MODULUS = 2_147_483_647  # a large prime, good enough for this educational simulation
SCALE = 2_000
LEARNING_RATE = 0.05
LINEAR_STEPS = 12
TRAIN_RATIO = 0.7
LOGISTIC_PUBLIC_WEIGHTS = np.array([-0.18, 0.72, 0.96, 0.34, -0.68], dtype=float)
BASE_DIR = Path(__file__).resolve().parent


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def normalise_group_ids(group_ids: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for sid in group_ids:
        sid = re.sub(r"\s+", "", str(sid))
        if not sid:
            continue
        if sid.upper().startswith("PUT_"):
            continue
        cleaned.append(sid)
    if not cleaned:
        raise ValueError("Set GROUP_MEMBER_IDS to one to four real student IDs before running the file.")
    if len(cleaned) > 4:
        raise ValueError("At most four student IDs are allowed.")
    return sorted(cleaned)


def seed_from_group_ids(group_ids: Sequence[str]) -> int:
    ids = normalise_group_ids(group_ids)
    token = "|".join(ids)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def group_label(group_ids: Sequence[str]) -> str:
    ids = normalise_group_ids(group_ids)
    compact = []
    for sid in ids:
        digits = re.sub(r"\D", "", sid)
        compact.append(digits[-4:] if digits else sid[-4:])
    return "_".join(compact)


def generate_hospital_datasets(group_ids: Sequence[str], n_a: int = 60, n_b: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate deterministic synthetic datasets for Hospital A and Hospital B.
    Do not edit this generator in the student version.
    """
    seed = seed_from_group_ids(group_ids)
    rng = np.random.default_rng(seed)

    def make_one(hospital: str, n: int, age_mu: float, age_sigma: float,
                 marker_mu: float, marker_sigma: float,
                 admissions_mu: float, treatment_prob: float,
                 lin_shift: float, log_shift: float) -> pd.DataFrame:
        raw_age = rng.normal(age_mu, age_sigma, n).clip(24, 92)
        raw_marker = rng.normal(marker_mu, marker_sigma, n)
        raw_adm = rng.poisson(admissions_mu, n).clip(0, 7)
        raw_treat = rng.binomial(1, treatment_prob, n)

        x_age = (raw_age - 58.0) / 11.0
        x_marker = raw_marker
        x_admissions = (raw_adm - 1.7) / 1.6
        x_treatment = raw_treat.astype(float)

        noise_linear = rng.normal(0.0, 0.32, n)
        risk_score = (
            1.90
            + 0.78 * x_age
            + 1.42 * x_marker
            + 0.36 * x_admissions
            - 0.62 * x_treatment
            + lin_shift
            + noise_linear
        )

        logit = (
            -0.22
            + 0.72 * x_age
            + 0.96 * x_marker
            + 0.34 * x_admissions
            - 0.68 * x_treatment
            + log_shift
            + rng.normal(0.0, 0.22, n)
        )
        needs_followup = rng.binomial(1, sigmoid(logit), n)

        return pd.DataFrame(
            {
                "hospital": hospital,
                "record_id": [f"{hospital}-{i:03d}" for i in range(1, n + 1)],
                "x_age": np.round(x_age, 4),
                "x_marker": np.round(x_marker, 4),
                "x_admissions": np.round(x_admissions, 4),
                "x_treatment": raw_treat.astype(int),
                "risk_score": np.round(risk_score, 4),
                "needs_followup": needs_followup.astype(int),
            }
        )

    hospital_a = make_one("A", n_a, 54.5, 7.8, 0.22, 0.76, 1.25, 0.32, 0.08, 0.10)
    hospital_b = make_one("B", n_b, 62.0, 8.9, -0.12, 0.98, 1.95, 0.48, -0.06, -0.06)
    return hospital_a, hospital_b


def save_generated_datasets(group_ids: Sequence[str], out_dir: Path | str = BASE_DIR / "generated_group_data") -> Tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    label = group_label(group_ids)
    hospital_a, hospital_b = generate_hospital_datasets(group_ids)
    a_path = out_path / f"hospital_A_group_{label}.csv"
    b_path = out_path / f"hospital_B_group_{label}.csv"
    hospital_a.to_csv(a_path, index=False)
    hospital_b.to_csv(b_path, index=False)
    return a_path, b_path, hospital_a, hospital_b


def make_design_matrix(df: pd.DataFrame, target_col: str = "risk_score") -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = ["x_age", "x_marker", "x_admissions", "x_treatment"]
    # TODO 1:
    # 1. Extract the four feature columns as a float NumPy array.
    # 2. Create an intercept column of ones.
    # 3. Concatenate the intercept and feature columns into X.
    # 4. Extract the target column into y.
    # 5. Return X, y.
    raise NotImplementedError("Complete make_design_matrix")


def split_train_test(df_all: pd.DataFrame, train_ratio: float = TRAIN_RATIO, group_ids: Sequence[str] = GROUP_MEMBER_IDS):
    seed = seed_from_group_ids(group_ids)
    rng = np.random.default_rng(seed + 99)
    idx = np.arange(len(df_all))
    rng.shuffle(idx)
    split = int(round(train_ratio * len(df_all)))
    train_idx = idx[:split]
    test_idx = idx[split:]
    train_df = df_all.iloc[train_idx].reset_index(drop=True)
    test_df = df_all.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    pred = X @ w
    return float(np.mean((pred - y) ** 2))


def plaintext_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # TODO 2:
    # 1. Compute residual = X @ w - y.
    # 2. Return the mean-squared-error gradient (2/n) * X.T @ residual.
    raise NotImplementedError("Complete plaintext_gradient")


def plaintext_train(X: np.ndarray, y: np.ndarray, learning_rate: float = LEARNING_RATE, steps: int = LINEAR_STEPS) -> Tuple[np.ndarray, List[float]]:
    # TODO 3:
    # 1. Initialise w as a zero vector of length X.shape[1].
    # 2. For the requested number of steps, compute the gradient and update w.
    # 3. After each update, append the current train MSE to history.
    # 4. Return the final weights and the history list.
    raise NotImplementedError("Complete plaintext_train")


def local_gradient(X_local: np.ndarray, y_local: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, int]:
    # TODO 4:
    # 1. Compute the local residuals for this hospital only.
    # 2. Compute the local gradient SUM, not yet divided by the total number of records.
    # 3. Return the local gradient sum together with the local record count.
    raise NotImplementedError("Complete local_gradient")


def plaintext_distributed_gradient(
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    grad_a_sum, n_a = local_gradient(X_a, y_a, w)
    grad_b_sum, n_b = local_gradient(X_b, y_b, w)
    return (grad_a_sum + grad_b_sum) / (n_a + n_b)


def plaintext_distributed_train(
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    steps: int = LINEAR_STEPS,
) -> Tuple[np.ndarray, List[float]]:
    # TODO 5:
    # 1. Initialise w as zeros.
    # 2. In each step, compute the distributed plaintext gradient using Hospital A and Hospital B.
    # 3. Update w and record the combined train MSE on all training rows.
    # 4. Return the final weights and the history list.
    raise NotImplementedError("Complete plaintext_distributed_train")




def mod_reduce(value):
    arr = np.asarray(value, dtype=object)
    reduced = np.mod(arr, MODULUS)
    return np.asarray(reduced, dtype=np.int64)


def mod_add_many(*terms):
    total = np.asarray(0, dtype=object)
    for term in terms:
        total = np.mod(total + np.asarray(term, dtype=object), MODULUS)
    return np.asarray(total, dtype=np.int64)


def to_signed(z):
    arr = np.asarray(z, dtype=np.int64)
    signed = np.where(arr > MODULUS // 2, arr - MODULUS, arr)
    if np.isscalar(z):
        return int(signed)
    return signed.astype(np.int64)


def encode_real(x):
    # TODO 6:
    # Multiply by SCALE, round to the nearest integer, cast to int64, and reduce modulo MODULUS.
    # Preserve scalar output when x is a scalar.
    raise NotImplementedError("Complete encode_real")


def decode_fp(z):
    # TODO 7:
    # Convert the field element back to a signed integer and divide by SCALE.
    raise NotImplementedError("Complete decode_fp")


def share_secret(secret, rng: np.random.Generator):
    # TODO 8:
    # 1. Convert secret to an int64 NumPy array modulo MODULUS.
    # 2. Sample a random share s0 of the same shape.
    # 3. Compute s1 = secret - s0 mod MODULUS.
    # 4. Return s0, s1.
    raise NotImplementedError("Complete share_secret")


def reconstruct(shares):
    # TODO 9:
    # Return (share0 + share1) mod MODULUS.
    raise NotImplementedError("Complete reconstruct")


def add_shares(a, b):
    # TODO 10:
    # Add the server-0 shares together and the server-1 shares together, each modulo MODULUS.
    raise NotImplementedError("Complete add_shares")


def sub_shares(a, b):
    # TODO 11:
    # Subtract the server-0 shares and the server-1 shares separately modulo MODULUS.
    raise NotImplementedError("Complete sub_shares")


def share_zero_like(reference):
    zeros = np.zeros_like(np.asarray(reference, dtype=np.int64))
    return zeros, zeros.copy()


def truncate_scaled_product_shares(product_shares, rng: np.random.Generator):
    opened = to_signed(reconstruct(product_shares))
    truncated = np.rint(np.asarray(opened, dtype=float) / SCALE).astype(np.int64) % MODULUS
    return share_secret(truncated, rng)


def mul_public_encoded(a, public_encoded: int, rng: np.random.Generator):
    # TODO 12:
    # 1. Multiply each share by the encoded public constant modulo MODULUS.
    # 2. Because the product has scale SCALE^2, call the truncation helper before returning.
    raise NotImplementedError("Complete mul_public_encoded")


def make_triple(shape, rng: np.random.Generator):
    a_plain = rng.integers(0, MODULUS, size=shape if shape else None, dtype=np.int64)
    b_plain = rng.integers(0, MODULUS, size=shape if shape else None, dtype=np.int64)
    c_plain = (a_plain * b_plain) % MODULUS
    return share_secret(a_plain, rng), share_secret(b_plain, rng), share_secret(c_plain, rng)


def beaver_mul(x_shares, y_shares, triple, rng: np.random.Generator):
    # TODO 13:
    # Implement Beaver-style secure multiplication.
    # Suggested steps:
    # 1. Unpack triple into shares of a, b, c where c = a*b.
    # 2. Compute d = x-a and e = y-b in shared form.
    # 3. Open d and e.
    # 4. Form the product shares using c + d*b + e*a + d*e on server 0 and c + d*b + e*a on server 1.
    # 5. Truncate the SCALE^2 product back to SCALE using the helper.
    raise NotImplementedError("Complete beaver_mul")


def secure_dot(x_row_shared, w_shared, rng: np.random.Generator):
    # TODO 14:
    # 1. Initialise a shared zero accumulator.
    # 2. For each coordinate, multiply x_j and w_j securely with Beaver multiplication.
    # 3. Add each term into the accumulator.
    # 4. Return the shared dot-product result.
    raise NotImplementedError("Complete secure_dot")


def encode_vector_as_shared(vec: np.ndarray, rng: np.random.Generator):
    return [share_secret(v, rng) for v in encode_real(vec)]


def secure_gradient_step(
    X_plain: np.ndarray,
    y_plain: np.ndarray,
    w_shared,
    learning_rate: float,
    rng: np.random.Generator,
):
    # TODO 15:
    # Implement one secret-shared gradient descent update.
    # Your implementation should:
    # 1. Encode X and y into fixed-point integers.
    # 2. For each row, secret-share the features and target.
    # 3. Securely compute the prediction and residual.
    # 4. Accumulate the secure gradient contributions for every weight.
    # 5. Multiply by 2/n and then by the learning rate using public encoded constants.
    # 6. Update the shared weights and return the new weights plus the reconstructed gradient.
    raise NotImplementedError("Complete secure_gradient_step")


def mpc_train_linear(
    X_plain: np.ndarray,
    y_plain: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    steps: int = LINEAR_STEPS,
    scale: int = SCALE,
    seed_offset: int = 1234,
):
    global SCALE
    old_scale = SCALE
    SCALE = int(scale)
    try:
        rng = np.random.default_rng(seed_from_group_ids(GROUP_MEMBER_IDS) + seed_offset)
        w_shared = encode_vector_as_shared(np.zeros(X_plain.shape[1], dtype=float), rng)
        history: List[float] = []

        for _ in range(steps):
            w_shared, _ = secure_gradient_step(X_plain, y_plain, w_shared, learning_rate, rng)
            w_plain = np.array([decode_fp(reconstruct(wj)) for wj in w_shared], dtype=float)
            history.append(mse(X_plain, y_plain, w_plain))

        final_w = np.array([decode_fp(reconstruct(wj)) for wj in w_shared], dtype=float)
        return final_w, history
    finally:
        SCALE = old_scale


def exact_logistic_probs(X_query: np.ndarray, public_weights: np.ndarray) -> np.ndarray:
    # TODO 16:
    # Compute the exact plaintext logistic probabilities using sigmoid(X_query @ public_weights).
    raise NotImplementedError("Complete exact_logistic_probs")


def cubic_sigmoid_approx(z):
    z = np.asarray(z, dtype=float)
    return np.clip(0.5 + 0.2166 * z - 0.0066 * (z ** 3), 0.0, 1.0)


def secure_approx_logistic_probs(X_query: np.ndarray, public_weights: np.ndarray, group_ids: Sequence[str]):
    # TODO 17:
    # Implement the secure approximate logistic inference workflow.
    # For each query row:
    # 1. Secret-share the encoded query features.
    # 2. Secret-share the public weight vector (for simplicity in this teaching scaffold).
    # 3. Securely compute z = x dot w.
    # 4. Securely compute z^3 using Beaver multiplication.
    # 5. Reconstruct z and z^3, then evaluate the cubic approximation
    #    0.5 + 0.2166*z - 0.0066*z^3, clipped to [0, 1].
    # 6. Return all approximate probabilities as a NumPy array.
    raise NotImplementedError("Complete secure_approx_logistic_probs")


def plot_histories(
    history_central: Sequence[float],
    history_distributed: Sequence[float],
    history_mpc: Sequence[float],
    out_path: Path,
):
    plt.figure(figsize=(7.2, 4.5))
    steps = np.arange(1, len(history_central) + 1)
    plt.plot(steps, history_central, marker="o", label="Trusted central plaintext")
    plt.plot(steps, history_distributed, marker="s", label="Distributed plaintext")
    plt.plot(steps, history_mpc, marker="^", label="Two-server MPC")
    plt.xlabel("Training step")
    plt.ylabel("Train MSE")
    plt.title("Linear regression training curves")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_summary_stats_table(hospital_a: pd.DataFrame, hospital_b: pd.DataFrame) -> pd.DataFrame:
    cols = ["x_age", "x_marker", "x_admissions", "risk_score", "needs_followup"]
    rows = []
    for col in cols:
        rows.append(
            {
                "column": col,
                "A_mean": hospital_a[col].mean(),
                "B_mean": hospital_b[col].mean(),
                "A_std": hospital_a[col].std(ddof=1),
                "B_std": hospital_b[col].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def main():
    ids = normalise_group_ids(GROUP_MEMBER_IDS)
    label = group_label(ids)
    a_path, b_path, hospital_a, hospital_b = save_generated_datasets(ids)
    print(f"Group IDs: {ids}")
    print(f"Group label: {label}")
    print(f"Hospital A CSV: {a_path}")
    print(f"Hospital B CSV: {b_path}")
    print("\nHospital A head:")
    print(hospital_a.head(3).to_string(index=False))
    print("\nHospital B head:")
    print(hospital_b.head(3).to_string(index=False))

    stats = make_summary_stats_table(hospital_a, hospital_b)
    print("\nSummary stats:")
    print(stats.round(4).to_string(index=False))

    combined = pd.concat([hospital_a, hospital_b], ignore_index=True)
    train_df, test_df = split_train_test(combined, TRAIN_RATIO, ids)

    train_a = train_df[train_df["hospital"] == "A"].reset_index(drop=True)
    train_b = train_df[train_df["hospital"] == "B"].reset_index(drop=True)

    X_train, y_train = make_design_matrix(train_df, "risk_score")
    X_test, y_test = make_design_matrix(test_df, "risk_score")
    X_a, y_a = make_design_matrix(train_a, "risk_score")
    X_b, y_b = make_design_matrix(train_b, "risk_score")

    zero_w = np.zeros(X_train.shape[1], dtype=float)
    grad0_central = plaintext_gradient(X_train, y_train, zero_w)
    gradA_sum0, nA0 = local_gradient(X_a, y_a, zero_w)
    gradB_sum0, nB0 = local_gradient(X_b, y_b, zero_w)
    grad0_distributed = (gradA_sum0 + gradB_sum0) / (nA0 + nB0)


    w_plain, hist_plain = plaintext_train(X_train, y_train)
    w_dist, hist_dist = plaintext_distributed_train(X_a, y_a, X_b, y_b)
    w_mpc, hist_mpc = mpc_train_linear(X_train, y_train)

    train_curve_path = BASE_DIR / "generated_group_data" / f"training_curve_group_{label}.png"
    plot_histories(hist_plain, hist_dist, hist_mpc, train_curve_path)

    plain_train_mse = mse(X_train, y_train, w_plain)
    plain_test_mse = mse(X_test, y_test, w_plain)
    dist_train_mse = mse(X_train, y_train, w_dist)
    dist_test_mse = mse(X_test, y_test, w_dist)
    mpc_train_mse = mse(X_train, y_train, w_mpc)
    mpc_test_mse = mse(X_test, y_test, w_mpc)

    print("\n=== Linear regression results ===")
    print("Initial central gradient:", np.round(grad0_central, 4))
    print("Initial Hospital A gradient contribution sum:", np.round(gradA_sum0, 4), "| n_A =", nA0)
    print("Initial Hospital B gradient contribution sum:", np.round(gradB_sum0, 4), "| n_B =", nB0)
    print("Initial distributed gradient:", np.round(grad0_distributed, 4))
    print("Trusted central weights:", np.round(w_plain, 4))
    print("Distributed plaintext weights:", np.round(w_dist, 4))
    print("Two-server MPC weights:", np.round(w_mpc, 4))
    print("Absolute |central - distributed|:", np.round(np.abs(w_plain - w_dist), 6))
    print("Absolute |central - MPC|:", np.round(np.abs(w_plain - w_mpc), 6))
    print(f"Central train MSE: {plain_train_mse:.6f} | test MSE: {plain_test_mse:.6f}")
    print(f"Distributed train MSE: {dist_train_mse:.6f} | test MSE: {dist_test_mse:.6f}")
    print(f"MPC train MSE: {mpc_train_mse:.6f} | test MSE: {mpc_test_mse:.6f}")

    # sensitivity study on scale
    w_mpc_low_scale, _ = mpc_train_linear(X_train, y_train, scale=50, seed_offset=4321)
    low_scale_test_mse = mse(X_test, y_test, w_mpc_low_scale)
    print(f"MPC test MSE with lower fixed-point scale 50: {low_scale_test_mse:.6f}")

    # logistic inference
    X_query = test_df.head(8)[["x_age", "x_marker", "x_admissions", "x_treatment"]].to_numpy(dtype=float)
    X_query = np.hstack([np.ones((len(X_query), 1), dtype=float), X_query])
    y_query = test_df.head(8)["needs_followup"].to_numpy(dtype=int)

    exact_probs = exact_logistic_probs(X_query, LOGISTIC_PUBLIC_WEIGHTS)
    secure_probs = secure_approx_logistic_probs(X_query, LOGISTIC_PUBLIC_WEIGHTS, ids)
    exact_labels = (exact_probs >= 0.5).astype(int)
    secure_labels = (secure_probs >= 0.5).astype(int)

    logistic_df = pd.DataFrame(
        {
            "query_record": test_df.head(8)["record_id"].tolist(),
            "true_label": y_query,
            "exact_prob": np.round(exact_probs, 4),
            "secure_poly_prob": np.round(secure_probs, 4),
            "abs_prob_error": np.round(np.abs(exact_probs - secure_probs), 4),
            "exact_label": exact_labels,
            "secure_label": secure_labels,
        }
    )
    logistic_csv = BASE_DIR / "generated_group_data" / f"logistic_query_results_group_{label}.csv"
    logistic_df.to_csv(logistic_csv, index=False)
    print("\n=== Logistic inference query results ===")
    print(logistic_df.to_string(index=False))
    print(f"\nAverage logistic probability error: {np.mean(np.abs(exact_probs - secure_probs)):.6f}")
    print(f"Top-1 label agreement: {(exact_labels == secure_labels).mean() * 100:.1f}%")
    print(f"Training curve image: {train_curve_path}")
    print(f"Logistic query CSV: {logistic_csv}")


if __name__ == "__main__":
    main()
