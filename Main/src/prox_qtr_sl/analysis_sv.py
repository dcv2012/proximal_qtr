import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from Main.src.data_generate import data_gen, origin_para_set
from Main.src.prox_qtr_sl.estimate_prox_qtr_sl import make_treatment_strata, split_fold_with_combo_support
from Main.src.prox_qtr_sl.step1_nuisance import estimate_nuisance
from Main.src.prox_qtr_sl.step2_inner import compute_sv_curve_on_grid, inner_optimization_grid
from Main.src.prox_qtr_sl.step3_outer import (
    optimize_outer_hyperparams,
    prepare_outer_tensors,
    train_outer_policies,
)


def _cross_fit_q22_train_val(
    df_train,
    df_val,
    seed,
    K_folds,
    mmr_loss,
    q22_output_bound,
    nuisance_n_trials,
):
    """
    与 train_policy_prox_qtr_sl 中 Step1 完全一致：分层 K 折留出 OOF q22，
    验证集上对每折可用的 (a1,a2) 估计器预测做均值集成。
    """
    print(f"[analysis_sv] cross-fitting q22: n_train={len(df_train)}, K_folds={K_folds}, seed={seed}", flush=True)
    q22_train_oof = np.zeros(len(df_train))
    q22_train_oof_counts = np.zeros(len(df_train), dtype=int)
    q22_val_preds = np.zeros(len(df_val))
    q22_val_pred_counts = np.zeros(len(df_val), dtype=int)

    train_strata = make_treatment_strata(df_train)
    combo_counts = train_strata.value_counts().sort_index()
    min_combo_count = int(combo_counts.min())
    if min_combo_count < K_folds:
        raise ValueError(
            f"K_folds={K_folds} is too large for cross-fitting: the smallest observed "
            f"(A1, A2) cell has only {min_combo_count} samples. Reduce K_folds or increase n_train."
        )

    required_min = int(np.ceil(2 * K_folds / (K_folds - 1))) if K_folds > 1 else 2
    if min_combo_count < required_min:
        raise ValueError(
            f"Cross-fitting needs at least {required_min} samples for the rarest combo to support "
            f"internal splits (currently {min_combo_count})."
        )

    kf = StratifiedKFold(n_splits=K_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, oof_idx) in enumerate(kf.split(df_train, train_strata)):
        print(f"[analysis_sv] nuisance fold {fold + 1}/{K_folds}", flush=True)
        df_train_fold = df_train.iloc[train_idx].reset_index(drop=True)
        df_oof_fold = df_train.iloc[oof_idx].reset_index(drop=True)

        sub_train_fold, sub_val_fold = split_fold_with_combo_support(df_train_fold, test_size=0.2, seed=seed + fold)

        for a1 in [1, -1]:
            for a2 in [1, -1]:
                if not ((sub_train_fold["A1"] == a1) & (sub_train_fold["A2"] == a2)).any():
                    continue

                predict_q22_fn, _, _ = estimate_nuisance(
                    sub_train_fold,
                    sub_val_fold,
                    a1,
                    a2,
                    n_trials=nuisance_n_trials,
                    mmr_loss_type=mmr_loss,
                    q22_output_bound=q22_output_bound,
                )

                oof_sub_mask = (df_oof_fold["A1"] == a1) & (df_oof_fold["A2"] == a2)
                if oof_sub_mask.sum() > 0:
                    matching_oof_data = df_oof_fold[oof_sub_mask].copy()
                    global_indices = df_train.iloc[oof_idx][oof_sub_mask.values].index
                    preds_part = predict_q22_fn(matching_oof_data)
                    q22_train_oof[global_indices] = preds_part
                    q22_train_oof_counts[global_indices] += 1

                val_sub_mask = (df_val["A1"] == a1) & (df_val["A2"] == a2)
                if val_sub_mask.sum() > 0:
                    matching_val_df = df_val[val_sub_mask].copy()
                    global_val_idx = df_val[val_sub_mask.values].index
                    preds_val_part = predict_q22_fn(matching_val_df)
                    q22_val_preds[global_val_idx] += preds_val_part
                    q22_val_pred_counts[global_val_idx] += 1

    uncovered_train_mask = q22_train_oof_counts != 1
    if np.any(uncovered_train_mask):
        uncovered_summary = (
            df_train.loc[uncovered_train_mask, ["A1", "A2"]].value_counts().sort_index().to_dict()
        )
        raise ValueError(
            "Cross-fitting failed to produce exactly one OOF q22 prediction per training sample. "
            f"Uncovered or duplicated cells: {uncovered_summary}"
        )

    uncovered_val_mask = q22_val_pred_counts == 0
    if np.any(uncovered_val_mask):
        uncovered_val_summary = (
            df_val.loc[uncovered_val_mask, ["A1", "A2"]].value_counts().sort_index().to_dict()
        )
        raise ValueError(
            "Validation ensemble failed to produce q22 predictions for some samples. "
            f"Missing cells: {uncovered_val_summary}"
        )

    q22_val_preds = q22_val_preds / q22_val_pred_counts
    return q22_train_oof, q22_val_preds


def run_single_experiment_with_sv_trace(
    n_train=5000,
    seed=285084,
    max_alt_iters=20,
    tau=0.5,
    phi_type=3,
    model_type="nn",
    dgp="S1",
    mmr_loss="V_statistic",
    q22_output_bound=5.0,
    K_folds=2,
    nuisance_n_trials=10,
    outer_n_trials=10,
):
    r"""
    按 method.tex 算法1 执行单次 AO 实验；Step1 q22 与 MC 一致采用分层交叉拟合。
    每轮用 inner_optimization_grid 更新 q（与 train_policy_prox_qtr_sl 一致），    另计算整条 $\hat SV$ 网格供诊断绘图。
    """
    print(
        f"[analysis_sv] AO trace start: seed={seed} n_train={n_train} phi={phi_type} C={q22_output_bound}",
        flush=True,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    params = origin_para_set
    n_val = int(n_train * 0.25)
    df_train = data_gen(n_train, params, scenario=dgp)
    df_val = data_gen(n_val, params, scenario=dgp)

    q22_train_oof, q22_val_preds = _cross_fit_q22_train_val(
        df_train,
        df_val,
        seed,
        K_folds,
        mmr_loss,
        q22_output_bound,
        nuisance_n_trials,
    )

    Y2_array = df_train["Y2"].values
    A1_array = df_train["A1"].values
    A2_array = df_train["A2"].values
    grid_Q = np.unique(np.sort(Y2_array))
    hn = 0.2 / np.log(n_train)
    epsilon_n = min(1e-4, 0.5 / np.sqrt(n_train))
    delta_n = min(1e-4, np.std(Y2_array) / (6 * np.sqrt(n_train)))

    q_current = np.quantile(Y2_array, tau)
    last_sign_f1 = None
    last_sign_f2 = None

    device_compute = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H1_train_tensor = torch.tensor(df_train["Y0"].values, dtype=torch.float32).unsqueeze(1).to(device_compute)
    H2_train_tensor = torch.cat(
        [
            torch.tensor(df_train["Y0"].values, dtype=torch.float32).unsqueeze(1),
            torch.tensor(df_train["Y1"].values, dtype=torch.float32).unsqueeze(1),
            torch.tensor(df_train["A1"].values, dtype=torch.float32).unsqueeze(1),
        ],
        dim=1,
    ).to(device_compute)

    best_params = optimize_outer_hyperparams(
        df_train,
        q22_train_oof,
        df_val,
        q22_val_preds,
        q_current,
        n_trials=outer_n_trials,
        epochs=200,
        phi_type=phi_type,
        model_type=model_type,
    )

    trace_rows = []
    curve_frames = []
    q_final = q_current
    sv_final = np.nan

    for it in range(max_alt_iters):
        train_dataset = prepare_outer_tensors(df_train, q22_train_oof, q_current)
        val_dataset = prepare_outer_tensors(df_val, q22_val_preds, q_current)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        f1, f2, _ = train_outer_policies(train_loader, val_loader, best_params, phi_type=phi_type, model_type=model_type)
        f1.eval()
        f2.eval()
        with torch.no_grad():
            f1_out = f1(H1_train_tensor).cpu().numpy().flatten()
            f2_out = f2(H2_train_tensor).cpu().numpy().flatten()

        phi1 = stats.norm.cdf((A1_array * f1_out) / hn)
        phi2 = stats.norm.cdf((A2_array * f2_out) / hn)

        q_new, sv_val = inner_optimization_grid(Y2_array, q22_train_oof, phi1, phi2, grid_Q, tau)
        curve = compute_sv_curve_on_grid(Y2_array, q22_train_oof, phi1, phi2, grid_Q, tau)

        sv_final = sv_val
        q_final = q_new

        sign_f1 = (f1_out > 0).astype(int)
        sign_f2 = (f2_out > 0).astype(int)
        policy_flip_count = 0
        if last_sign_f1 is not None:
            policy_flip_count = int(np.sum(sign_f1 != last_sign_f1) + np.sum(sign_f2 != last_sign_f2))

        trace_rows.append(
            {
                "iter": it + 1,
                "q_prev": float(q_current),
                "q_new": float(q_new),
                "sv_at_q_new": float(sv_val),
                "target": float(1 - tau),
                "abs_sv_gap": float(abs(sv_val - (1 - tau))),
                "q_shift": float(abs(q_new - q_current)),
                "policy_flip_count": policy_flip_count,
                "norm_factor": float(curve["norm_factor"]),
                "crossing_qs": ";".join([f"{x:.10f}" for x in curve["crossing_qs"]]),
                "diag_q_rightmost_feasible": float(curve["q_best"]),
                "diag_sv_at_rightmost": float(curve["sv_best"]),
            }
        )

        frame = pd.DataFrame(
            {
                "iter": it + 1,
                "q_grid": curve["grid_Q"],
                "sv_hat": curve["sv_array"],
                "target": curve["target"],
            }
        )
        curve_frames.append(frame)

        # 与 estimate_prox_qtr_sl.train_policy_prox_qtr_sl 一致的 AO 停止准则
        f1_flip_rate = 1.0
        f2_flip_rate = 1.0
        if last_sign_f1 is not None and last_sign_f2 is not None:
            f1_flip_rate = np.sum(sign_f1 != last_sign_f1) / n_train
            f2_flip_rate = np.sum(sign_f2 != last_sign_f2) / n_train

        if (
            it > 0
            and f1_flip_rate <= 0.05
            and f2_flip_rate <= 0.05
            and abs(sv_val - (1 - tau)) <= epsilon_n
        ):
            break
        if it > 0 and f1_flip_rate <= 0.05 and f2_flip_rate <= 0.05 and abs(q_new - q_current) <= delta_n:
            break
        if it > 0 and (
            (f1_flip_rate <= 0.05 and f2_flip_rate <= 0.05)
            and (f1_flip_rate <= 0.0001 or f2_flip_rate <= 0.0001)
        ):
            break

        q_current = q_new
        last_sign_f1, last_sign_f2 = sign_f1, sign_f2

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "results",
        "sv_curve_diagnostics",
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    prefix = (
        f"{dgp}_n{n_train}_tau{tau}_phi{phi_type}_{model_type}_seed{seed}_C{q22_output_bound}_kf{K_folds}"
    )

    trace_path = os.path.join(out_dir, f"sv_trace_{prefix}.csv")
    curves_path = os.path.join(out_dir, f"sv_curve_grid_{prefix}.csv")
    fig_path = os.path.join(out_dir, f"sv_curve_plot_{prefix}.png")

    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(curves_path, index=False)

    n_iter = len(curve_frames)
    n_rows = int(np.ceil(n_iter / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)
    for idx, frame in enumerate(curve_frames):
        r = idx // 2
        c = idx % 2
        ax = axes[r][c]
        ax.plot(frame["q_grid"], frame["sv_hat"], linewidth=1.2)
        ax.axhline(y=frame["target"].iloc[0], linestyle="--")
        row = trace_rows[idx]
        ax.axvline(x=row["q_new"], linestyle=":")
        ax.set_title(f"Iter {idx+1}: q*={row['q_new']:.4f}, SV={row['sv_at_q_new']:.4f}")
        ax.set_xlabel("q")
        ax.set_ylabel("SV_hat")
        ax.grid(alpha=0.3)
    for idx in range(n_iter, n_rows * 2):
        r = idx // 2
        c = idx % 2
        axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"SV trace saved to: {trace_path}")
    print(f"SV curve grid saved to: {curves_path}")
    print(f"SV plot saved to: {fig_path}")
    print(f"Final q={q_final:.6f}, final SV={sv_final:.6f}, AO iters={n_iter}")
    return trace_path, curves_path, fig_path


def main():
    parser = argparse.ArgumentParser(description="SV curve diagnostics for Proximal QTR (cross-fitting aligned with MC)")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=285084)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--phi_type", type=int, default=3, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--model_type", type=str, default="nn", choices=["linear", "nn"])
    parser.add_argument("--dgp", type=str, default="S1", choices=["S1", "S2"])
    parser.add_argument("--max_alt_iters", type=int, default=20)
    parser.add_argument("--k_folds", type=int, default=2)
    parser.add_argument("--mmr_loss", type=str, default="V_statistic", choices=["U_statistic", "V_statistic"])
    parser.add_argument("--q22_output_bound", type=float, default=3.0)
    parser.add_argument("--nuisance_n_trials", type=int, default=10)
    parser.add_argument("--outer_n_trials", type=int, default=10)
    parser.add_argument(
        "--bad_seeds_430_phi1_n5000",
        action="store_true",
        help="Run reps 22,14,25,28 (base seed 285063) with phi=1, n=5000, C=3 for comparative_analysis_430.",
    )
    args = parser.parse_args()

    if args.bad_seeds_430_phi1_n5000:
        base = 285063
        reps = [22, 14, 25, 28]
        for r in reps:
            s = base + r - 1
            print(f"\n=== 430-style bad seed rep={r}, seed={s} ===\n")
            run_single_experiment_with_sv_trace(
                n_train=5000,
                seed=s,
                max_alt_iters=args.max_alt_iters,
                tau=args.tau,
                phi_type=1,
                model_type="nn",
                dgp="S1",
                mmr_loss=args.mmr_loss,
                q22_output_bound=args.q22_output_bound,
                K_folds=args.k_folds,
                nuisance_n_trials=args.nuisance_n_trials,
                outer_n_trials=args.outer_n_trials,
            )
    else:
        run_single_experiment_with_sv_trace(
            n_train=args.n_train,
            seed=args.seed,
            max_alt_iters=args.max_alt_iters,
            tau=args.tau,
            phi_type=args.phi_type,
            model_type=args.model_type,
            dgp=args.dgp,
            mmr_loss=args.mmr_loss,
            q22_output_bound=args.q22_output_bound,
            K_folds=args.k_folds,
            nuisance_n_trials=args.nuisance_n_trials,
            outer_n_trials=args.outer_n_trials,
        )


if __name__ == "__main__":
    main()
