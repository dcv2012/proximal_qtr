from typing import Dict, Any, Optional, Tuple, Union

from sklearn.preprocessing import StandardScaler

from src.data.data_class import MMRTrainDataSet_h, MMRTestDataSet_h


def standardise(data: MMRTrainDataSet_h) -> Tuple[MMRTrainDataSet_h, Dict[str, StandardScaler]]:
    treatment_proxy_scaler = StandardScaler()
    treatment_proxy_s = treatment_proxy_scaler.fit_transform(data.treatment_proxy)

    treatment_scaler = StandardScaler()
    treatment_s = treatment_scaler.fit_transform(data.treatment)

    outcome_scaler = StandardScaler()
    outcome_s = outcome_scaler.fit_transform(data.outcome)

    outcome_proxy_scaler = StandardScaler()
    outcome_proxy_s = outcome_proxy_scaler.fit_transform(data.outcome_proxy)

    backdoor_s = None
    backdoor_scaler = None
    if data.backdoor is not None:
        backdoor_scaler = StandardScaler()
        backdoor_s = backdoor_scaler.fit_transform(data.backdoor)

    train_data = MMRTrainDataSet_h(treatment=treatment_s,
                                treatment_proxy=treatment_proxy_s,
                                outcome_proxy=outcome_proxy_s,
                                outcome=outcome_s,
                                backdoor=backdoor_s)

    scalers = dict(treatment_proxy_scaler=treatment_proxy_scaler,
                   treatment_scaler=treatment_scaler,
                   outcome_proxy_scaler=outcome_proxy_scaler,
                   outcome_scaler=outcome_scaler,
                   backdoor_scaler=backdoor_scaler)

    return train_data, scalers
