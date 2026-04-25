import copy
import numpy as np
import pandas as pd
import torch


SCENARIOS = {"S1", "S2"}


origin_para_set = {
    "mu_Y0": -0.35,
    "sigma_Y0": 0.2,
    "mu_U0": 0.35,
    "sigma_U0": 0.5,
    "alpha_A1": np.array([-0.2, 0.05, 0.5], dtype=float),
    "mu_Z11": np.array([0.45, 0.25, 0.5, 2.5], dtype=float),
    "sigma_Z11": 0.2,
    "mu_Z12": np.array([0.05, 0.15, 0.45, -2.5], dtype=float),
    "sigma_Z12": 0.2,
    "mu_W11": np.array([0.2, -0.35, 0.8], dtype=float),
    "sigma_W11": 0.5,
    "mu_Y1": np.array([0.3, 0.1, 0.4, -0.6], dtype=float),
    "sigma_Y1": 0.5,
    "mu_U1": np.array([0.2, 0.1, 0.0, 0.8], dtype=float),
    "sigma_U1": 0.5,
    "alpha_A2": np.array([-0.2, 0.0, 0.02, 0.03, -0.03, 0.5], dtype=float),
    "mu_Z21": np.array([0.4, 0.0, 0.0, 0.1, 0.2, 2.5, 0.1, 0.2, -0.5], dtype=float),
    "sigma_Z21": 1.0,
    "mu_Z22": np.array([-0.05, 0.1, 0.1, 0.1, -0.2, -0.5, -0.05, 0.2, 0.5], dtype=float),
    "sigma_Z22": 1.0,
    "mu_W21": np.array([0.35, 0.2, 0.5, 2.5, -0.2, 0.2], dtype=float),
    "sigma_W21": 0.2,
    "mu_W22": np.array([0.2, 0.1, 0.5, -2.5, -0.5, -0.2], dtype=float),
    "sigma_W22": 0.2,
    "sigma_Y2": 0.2,
}


def validate_scenario(scenario: str) -> str:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}, got {scenario!r}")
    return scenario





def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sanitize_policy_output(raw_output: np.ndarray, stage_name: str) -> np.ndarray:
    raw_output = np.asarray(raw_output, dtype=float)
    if not np.isfinite(raw_output).all():
        print(f"Warning: {stage_name} produced NaN/Inf during counterfactual generation. Auto-correcting to 0.")
        raw_output = np.nan_to_num(raw_output, nan=0.0, posinf=0.0, neginf=0.0)
    return raw_output


def sample_y2(common_data: dict, sigma_y2: float, scenario: str) -> np.ndarray:
    scenario = validate_scenario(scenario)
    
    # Extract variables for convenience to allow flexible functional forms
    A1 = common_data["A1"]
    A2 = common_data["A2"]
    U0 = common_data["U0"]
    U1 = common_data["U1"]
    Y0 = common_data["Y0"]
    Y1 = common_data["Y1"]
    Z11 = common_data["Z11"]
    Z12 = common_data["Z12"]
    W11 = common_data["W11"]
    Z21 = common_data["Z21"]
    Z22 = common_data["Z22"]
    W21 = common_data["W21"]
    W22 = common_data["W22"]
    
    if scenario == "S1":
        '''mean_y2 = (
            -1.5
            - 2.5 * A2
            - 1.5 * A1
            - 0.5 * A1 * A2
            + 5.0 * U1
            + 4.0 * U0
            - 4.0 * A1 * U0
            - 3.0 * A2 * U1
        )'''
        mean_y2 = (-2.25 - 1.5 * A1 - 2.25 * A2 - 0.5 * A1 * A2  + 5 * U1 + 4 * U0 - 3 * A2 * U1)

    elif scenario == "S2":
        u0_sq = U0 ** 2
        u1_sq = U1 ** 2
        mean_y2 = (
            -1.0
            + 4.0 * u0_sq
            + 5.0 * u1_sq
            - (1.0 - 3.0 * u0_sq) * A1
            - (1.5 - 2.5 * u1_sq) * A2
            - 0.5 * A1 * A2
        )
    return mean_y2 + np.random.normal(0.0, sigma_y2, len(A1))


def generate_common_latent_process(sample_size: int, para_set: dict, A1: np.ndarray | None = None, A2: np.ndarray | None = None):
    N = sample_size
    Y0 = np.random.normal(para_set["mu_Y0"], para_set["sigma_Y0"], N)
    U0 = np.random.normal(para_set["mu_U0"], para_set["sigma_U0"], N)

    if A1 is None:
        lin_pred_A1 = para_set["alpha_A1"][0] + para_set["alpha_A1"][1] * Y0 + para_set["alpha_A1"][2] * U0
        A1 = 2 * np.random.binomial(1, expit(lin_pred_A1), N) - 1
    else:
        A1 = np.asarray(A1, dtype=float)

    Z11 = (
        para_set["mu_Z11"][0]
        + para_set["mu_Z11"][1] * A1
        + para_set["mu_Z11"][2] * Y0
        + para_set["mu_Z11"][3] * U0
        + np.random.normal(0.0, para_set["sigma_Z11"], N)
    )
    Z12 = (
        para_set["mu_Z12"][0]
        + para_set["mu_Z12"][1] * A1
        + para_set["mu_Z12"][2] * Y0
        + para_set["mu_Z12"][3] * U0
        + np.random.normal(0.0, para_set["sigma_Z12"], N)
    )
    W11 = (
        para_set["mu_W11"][0]
        + para_set["mu_W11"][1] * Y0
        + para_set["mu_W11"][2] * U0
        + np.random.normal(0.0, para_set["sigma_W11"], N)
    )
    Y1 = (
        para_set["mu_Y1"][0]
        + para_set["mu_Y1"][1] * A1
        + para_set["mu_Y1"][2] * Y0
        + para_set["mu_Y1"][3] * U0
        + np.random.normal(0.0, para_set["sigma_Y1"], N)
    )
    U1 = (
        para_set["mu_U1"][0]
        + para_set["mu_U1"][1] * A1
        + para_set["mu_U1"][2] * Y0
        + para_set["mu_U1"][3] * U0
        + np.random.normal(0.0, para_set["sigma_U1"], N)
    )

    if A2 is None:
        lin_pred_A2 = (
            para_set["alpha_A2"][0]
            + para_set["alpha_A2"][1] * A1
            + para_set["alpha_A2"][2] * Y0
            + para_set["alpha_A2"][3] * U0
            + para_set["alpha_A2"][4] * Y1
            + para_set["alpha_A2"][5] * U1
        )
        A2 = 2 * np.random.binomial(1, expit(lin_pred_A2), N) - 1
    else:
        A2 = np.asarray(A2, dtype=float)

    Z21 = (
        para_set["mu_Z21"][0]
        + para_set["mu_Z21"][1] * Z11
        + para_set["mu_Z21"][2] * Z12
        + para_set["mu_Z21"][3] * A2
        + para_set["mu_Z21"][4] * Y1
        + para_set["mu_Z21"][5] * U1
        + para_set["mu_Z21"][6] * A1
        + para_set["mu_Z21"][7] * Y0
        + para_set["mu_Z21"][8] * U0
        + np.random.normal(0.0, para_set["sigma_Z21"], N)
    )
    Z22 = (
        para_set["mu_Z22"][0]
        + para_set["mu_Z22"][1] * Z11
        + para_set["mu_Z22"][2] * Z12
        + para_set["mu_Z22"][3] * A2
        + para_set["mu_Z22"][4] * Y1
        + para_set["mu_Z22"][5] * U1
        + para_set["mu_Z22"][6] * A1
        + para_set["mu_Z22"][7] * Y0
        + para_set["mu_Z22"][8] * U0
        + np.random.normal(0.0, para_set["sigma_Z22"], N)
    )
    W21 = (
        para_set["mu_W21"][0]
        + para_set["mu_W21"][1] * W11
        + para_set["mu_W21"][2] * Y1
        + para_set["mu_W21"][3] * U1
        + para_set["mu_W21"][4] * Y0
        + para_set["mu_W21"][5] * U0
        + np.random.normal(0.0, para_set["sigma_W21"], N)
    )
    W22 = (
        para_set["mu_W22"][0]
        + para_set["mu_W22"][1] * W11
        + para_set["mu_W22"][2] * Y1
        + para_set["mu_W22"][3] * U1
        + para_set["mu_W22"][4] * Y0
        + para_set["mu_W22"][5] * U0
        + np.random.normal(0.0, para_set["sigma_W22"], N)
    )

    return {
        "Y0": Y0,
        "U0": U0,
        "A1": A1,
        "Z11": Z11,
        "Z12": Z12,
        "W11": W11,
        "Y1": Y1,
        "U1": U1,
        "A2": A2,
        "Z21": Z21,
        "Z22": Z22,
        "W21": W21,
        "W22": W22,
    }


def assemble_dataframe(common_data: dict, Y2: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Y0": common_data["Y0"],
            "U0": common_data["U0"],
            "A1": common_data["A1"],
            "Z11": common_data["Z11"],
            "Z12": common_data["Z12"],
            "W11": common_data["W11"],
            "Y1": common_data["Y1"],
            "U1": common_data["U1"],
            "A2": common_data["A2"],
            "Z21": common_data["Z21"],
            "Z22": common_data["Z22"],
            "W21": common_data["W21"],
            "W22": common_data["W22"],
            "Y2": Y2,
        }
    )


def data_gen(sample_size: int, para_set: dict, scenario: str = "S1") -> pd.DataFrame:
    common_data = generate_common_latent_process(sample_size, para_set)
    Y2 = sample_y2(common_data, para_set["sigma_Y2"], scenario)
    return assemble_dataframe(common_data, Y2)


def intervened_data_gen(sample_size: int, para_set: dict, a: list = [1, 1], scenario: str = "S1") -> pd.DataFrame:
    common_data = generate_common_latent_process(sample_size, para_set, A1=np.full(sample_size, a[0]), A2=np.full(sample_size, a[1]))
    Y2 = sample_y2(common_data, para_set["sigma_Y2"], scenario)
    return assemble_dataframe(common_data, Y2)


def dynamic_intervened_data_gen(sample_size: int, para_set: dict, f1=None, f2=None, device="cpu", seed=None, scenario: str = "S1") -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    N = sample_size
    Y0 = np.random.normal(para_set["mu_Y0"], para_set["sigma_Y0"], N)
    U0 = np.random.normal(para_set["mu_U0"], para_set["sigma_U0"], N)

    with torch.no_grad():
        if f1 is not None:
            H1_input = torch.tensor(Y0, dtype=torch.float32).unsqueeze(1).to(device)
            f1_out = sanitize_policy_output(f1(H1_input).cpu().numpy().flatten(), "f1")
            A1 = np.sign(f1_out)
            A1[A1 == 0] = 1
        else:
            A1 = np.ones(N)

    Z11 = (
        para_set["mu_Z11"][0]
        + para_set["mu_Z11"][1] * A1
        + para_set["mu_Z11"][2] * Y0
        + para_set["mu_Z11"][3] * U0
        + np.random.normal(0.0, para_set["sigma_Z11"], N)
    )
    Z12 = (
        para_set["mu_Z12"][0]
        + para_set["mu_Z12"][1] * A1
        + para_set["mu_Z12"][2] * Y0
        + para_set["mu_Z12"][3] * U0
        + np.random.normal(0.0, para_set["sigma_Z12"], N)
    )
    W11 = (
        para_set["mu_W11"][0]
        + para_set["mu_W11"][1] * Y0
        + para_set["mu_W11"][2] * U0
        + np.random.normal(0.0, para_set["sigma_W11"], N)
    )
    Y1 = (
        para_set["mu_Y1"][0]
        + para_set["mu_Y1"][1] * A1
        + para_set["mu_Y1"][2] * Y0
        + para_set["mu_Y1"][3] * U0
        + np.random.normal(0.0, para_set["sigma_Y1"], N)
    )
    U1 = (
        para_set["mu_U1"][0]
        + para_set["mu_U1"][1] * A1
        + para_set["mu_U1"][2] * Y0
        + para_set["mu_U1"][3] * U0
        + np.random.normal(0.0, para_set["sigma_U1"], N)
    )

    with torch.no_grad():
        if f2 is not None:
            H2_input = torch.cat(
                [
                    torch.tensor(Y0, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(Y1, dtype=torch.float32).unsqueeze(1),
                    torch.tensor(A1, dtype=torch.float32).unsqueeze(1),
                ],
                dim=1,
            ).to(device)
            f2_out = sanitize_policy_output(f2(H2_input).cpu().numpy().flatten(), "f2")
            A2 = np.sign(f2_out)
            A2[A2 == 0] = 1
        else:
            A2 = np.ones(N)

    Z21 = (
        para_set["mu_Z21"][0]
        + para_set["mu_Z21"][1] * Z11
        + para_set["mu_Z21"][2] * Z12
        + para_set["mu_Z21"][3] * A2
        + para_set["mu_Z21"][4] * Y1
        + para_set["mu_Z21"][5] * U1
        + para_set["mu_Z21"][6] * A1
        + para_set["mu_Z21"][7] * Y0
        + para_set["mu_Z21"][8] * U0
        + np.random.normal(0.0, para_set["sigma_Z21"], N)
    )
    Z22 = (
        para_set["mu_Z22"][0]
        + para_set["mu_Z22"][1] * Z11
        + para_set["mu_Z22"][2] * Z12
        + para_set["mu_Z22"][3] * A2
        + para_set["mu_Z22"][4] * Y1
        + para_set["mu_Z22"][5] * U1
        + para_set["mu_Z22"][6] * A1
        + para_set["mu_Z22"][7] * Y0
        + para_set["mu_Z22"][8] * U0
        + np.random.normal(0.0, para_set["sigma_Z22"], N)
    )
    W21 = (
        para_set["mu_W21"][0]
        + para_set["mu_W21"][1] * W11
        + para_set["mu_W21"][2] * Y1
        + para_set["mu_W21"][3] * U1
        + para_set["mu_W21"][4] * Y0
        + para_set["mu_W21"][5] * U0
        + np.random.normal(0.0, para_set["sigma_W21"], N)
    )
    W22 = (
        para_set["mu_W22"][0]
        + para_set["mu_W22"][1] * W11
        + para_set["mu_W22"][2] * Y1
        + para_set["mu_W22"][3] * U1
        + para_set["mu_W22"][4] * Y0
        + para_set["mu_W22"][5] * U0
        + np.random.normal(0.0, para_set["sigma_W22"], N)
    )

    common_data = {
        "Y0": Y0,
        "U0": U0,
        "A1": A1,
        "Z11": Z11,
        "Z12": Z12,
        "W11": W11,
        "Y1": Y1,
        "U1": U1,
        "A2": A2,
        "Z21": Z21,
        "Z22": Z22,
        "W21": W21,
        "W22": W22,
    }
    Y2 = sample_y2(common_data, para_set["sigma_Y2"], scenario)
    return assemble_dataframe(common_data, Y2)
