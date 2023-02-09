import warnings

import pandas as pd

from bofire.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.strategies.randomforest import RandomForest
from tests.bofire.domain.test_features import (
    VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)
if2 = ContinuousInput(
    **{
        **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)

if3 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    }
)

if4 = CategoricalInput(
    **{
        **VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if4",
    }
)

if5 = CategoricalDescriptorInput(
    **{
        **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if5",
    }
)

if6 = CategoricalDescriptorInput(
    **{
        **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if6",
    }
)

# if7 = DummyFeature(key="if7")

if8 = CategoricalDescriptorInput(
    **{
        **VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if8",
    }
)

of1 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    }
)

of2 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of2",
    }
)

# Copied from test_base.py
domains = [
    Domain(
        input_features=[if1, if3, if5],  # no fixed features
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if3, if5],  # no fixed features
        output_features=[of1, of2],
        constraints=[],
    ),
]

data = [
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if3": ["c1", "c2", "c3", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        }
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if3": ["c1", "c2", "c3", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "of1": [10, 11, 12, 13],
            "of2": [0.1, 0.98, 1, 0.75],
        }
    ),
    pd.DataFrame.from_dict(
        {"if1": [3.01, 6], "if3": ["c1", "c2"], "if5": ["c1", "c3"], "of1": [9.8, 11]}
    ),
]


def test_init():
    test_domain = domains[0]
    test_domain.experiments = data[0]
    rf_strategy = RandomForest(domain=test_domain)
    assert len(rf_strategy.models[0].feature_importances_) == 5


def test_predict():
    test_domain = domains[0]
    test_domain.experiments = data[0]

    rf_strategy = RandomForest(domain=test_domain)

    # predict the training data... uncertainty should be zero
    p = rf_strategy.predict(test_domain.experiments)
    print("predictions (training data):")
    print(p)
    print(p.iloc[:, 0])
    print(p.iloc[:, 1].values)

    # predict modified data
    p = rf_strategy.predict(data[2])
    print("predictions (modified data):")
    print(p)
    print(p.iloc[:, 0])
    print(p.iloc[:, 1].values)


def test_random():
    test_domain = domains[0]
    rf_strat = RandomForest(domain=test_domain)
    random_samples = rf_strat.make_random_candidates(12)
    assert random_samples.shape[0] == 12


def test_ask_1234():
    test_domain = domains[1]
    test_domain.experiments = data[1]
    rf_strat = RandomForest(domain=test_domain)
    proposals = rf_strat.ask(12)
    assert proposals.shape[0] == 12


if __name__ == "__main__":
    test_random()