import numpy as np
from scipy.stats import norm, multivariate_normal, uniform, randint

from src.data.data_class import MMRTrainDataSet_h, MMRTestDataSet_h
from src.data.data_class import MMRTrainDataSet_q, MMRTestDataSet_q

from src.utils.setseed import set_seed

set_seed(20048)

def generate_data(n_sample: int, scenario: str):
    GammaX = np.array([0.25, 0.25])
    SigmaX = np.array([[0.25**2, 0],[0, 0.25**2]])
    bX = np.array([0.25, 0.25])
    
    alpha0, alphaA, alphaX = 0.25, 0.25, np.array([0.25, 0.25])
    mu0, muA, muX = 0.25, 0.25, np.array([0.25, 0.25])
    kappa0, kappaA, kappaX = 0.25, 0.25, np.array([0.25, 0.25])
    
    sigmaZW, sigmaZU, sigmaWU = 0.5, 0.5, 0.5
    sigmaZ = sigmaW = sigmaU = 1
    sigmaY = 0.25
    omega = 1
    b0 = 2

    Sigma = np.array([
        [sigmaZ**2, sigmaZW, sigmaZU],
        [sigmaZW, sigmaW**2, sigmaWU],
        [sigmaZU, sigmaWU, sigmaU**2]
    ])


    scenario_dict = {
        'S1': (
            lambda X: 4 + np.abs(X[:, 0] - 1) - np.abs(X[:, 1] + 1),
            lambda X: 2,
            lambda bA, bW, A, X, W, temp:
                b0 + bA * A + np.sum(X**2, axis=1) +
                (bW - 1.5 * A + A * (np.sin(X[:, 0]) - np.cos(X[:, 1])) - omega) * temp +
                omega * W.reshape(-1)
        ),
        'S2': (
            lambda X: 4 - 4 * X[:, 0] * X[:, 1],
            lambda X: 2,
            lambda bA, bW, A, X, W, temp:
                b0 + bA * A + np.sum(X**2, axis=1) +
                (bW - 3 * A + A * X[:, 0] - omega) * temp +
                omega * W.reshape(-1)
        ),
        'S3': (
            lambda X: 4 - 2 * X[:, 0] + X[:, 1],
            lambda X: 2,
            lambda bA, bW, A, X, W, temp:
                b0 + bA * A + X @ bX +
                (bW - 3 * A + A * X[:, 1] - omega) * temp +
                omega * W.reshape(-1)
        ),
        'S4': (
            lambda X: 2 + np.cos(X[:, 0]) - np.sin(X[:, 1]),
            lambda X: 1,
            lambda bA, bW, A, X, W, temp:
                b0 + bA * A + X @ bX +
                (bW - 3 * A + 3 * A * np.exp(X[:, 1]) - A * np.exp(X[:, 0]) - omega) * temp +
                omega * W.reshape(-1)
        ),
        'S5': (
            lambda X: 4 + np.exp(X[:, 0]) - np.exp(2 * X[:, 1]),
            lambda X: 1,
            lambda bA, bW, A, X, W, temp:
                b0 + bA * A + X @ bX +
                (bW - 2.5 * A + 2 * A * np.abs(X[:, 1] + 1) - 3 * A * np.abs(X[:, 0] - 1) - omega) * temp +
                omega * W.reshape(-1)
        ),
        'S6': (
            lambda X: 4 + np.cbrt(X[:, 0]) - 3 * np.cbrt(X[:, 1]),
            lambda X: 1,
            lambda bA, bW, A, X, W, temp:
                b0 + bA * A + np.sum(X**2, axis=1) +
                (bW - 1.5 * A + A * X[:, 1] ** 2 - omega) * temp +
                omega * W.reshape(-1)
        ),
    }

    if scenario not in scenario_dict:
        raise ValueError(f"scenario must be one of {list(scenario_dict.keys())}, got {scenario}")

    bA_func, bW_func, y_formula = scenario_dict[scenario]


    X = np.random.multivariate_normal(GammaX, SigmaX, size=n_sample)
    
    logits = X @ np.array([0.125, 0.125])
    pA = 1 / (1 + np.exp(logits))
    A = np.random.binomial(n=1, p=pA)
    
    samples = np.random.multivariate_normal(np.zeros(3), Sigma, n_sample)
    Z = samples[:, 0] + alpha0 + alphaA * A + X @ alphaX
    W = samples[:, 1] + mu0 + muA * A + X @ muX
    U = samples[:, 2] + kappa0 + kappaA * A + X @ kappaX
    temp = mu0 + X @ muX + sigmaWU / sigmaU**2 * (U - kappa0 - X @ kappaX)

    # generate Y, Y1, Y0
    def gen_Y(A_val):
        bA = bA_func(X)
        bW = bW_func(X)
        mu = y_formula(bA, bW, A_val, X, W, temp)
        return np.random.normal(mu, sigmaY).reshape(-1, 1)

    Y = gen_Y(A)
    Y1 = gen_Y(np.ones_like(A))
    Y0 = gen_Y(np.zeros_like(A))
    E1, E0 = Y1.mean(), Y0.mean()

    # reshape
    return (
        A.reshape(-1, 1),
        W.reshape(-1, 1),
        X,
        Y.reshape(-1, 1),
        Z.reshape(-1, 1),
        E1, E0
    )

def generate_train_simulation_h(n_sample: int, scenario: str, **kwargs):
    A, W, X, Y, Z, _, _ = generate_data(n_sample, scenario=scenario)
    return MMRTrainDataSet_h(treatment=A,
                             treatment_proxy=Z,
                             outcome_proxy=W,
                             outcome=Y,
                             backdoor=X)


def generate_train_simulation_q(n_sample: int, scenario: str, **kwargs):
    A, W, X, Y, Z, _, _ = generate_data(n_sample, scenario=scenario)
    A_target = np.zeros(n_sample)
    return MMRTrainDataSet_q(treatment=A,
                             treatment_target=A_target,
                             treatment_proxy=Z,
                             outcome_proxy=W,
                             outcome=Y,
                             backdoor=X)


def generate_test_simulation_h(n_sample: int, scenario: str, **kwargs):
    A, W, X, Y, Z, E1, E0 = generate_data(n_sample, scenario=scenario)
    return MMRTestDataSet_h(treatment=A,
                             treatment_proxy=Z,
                             outcome_proxy=W,
                             outcome=Y,
                             backdoor=X)


def generate_test_simulation_q(n_sample: int, scenario: str, **kwargs):
    A, W, X, Y, Z, E1, E0 = generate_data(n_sample, scenario=scenario)
    return MMRTestDataSet_q(treatment=A,
                           treatment_proxy=Z,
                           outcome_proxy=W,
                           outcome=Y,
                           backdoor=X,
                           structural=[E1, E0])




if __name__ == "__main__":
    set_seed(20048)
    
    s = generate_train_simulation_q(1000, scenario='S1')
    a = s.treatment_target.shape
    print(a)
    