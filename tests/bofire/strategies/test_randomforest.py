import warnings

import numpy as np
import pytest

from bofire.domain import Domain
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.random import RandomStrategy
from bofire.strategies.randomforest import RandomForest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if0 = ContinuousInput(key="if0", lower_bound=0, upper_bound=1)
if1 = ContinuousInput(key="if1", lower_bound=0, upper_bound=2)
if2 = ContinuousInput(key="if2", lower_bound=0, upper_bound=3)
if3 = CategoricalInput(key="if3", categories=["c1", "c2", "c3"])
if4 = CategoricalInput(
    key="if4", categories=["A", "B", "C"], allowed=[True, True, True]
)
if5 = CategoricalInput(key="if5", categories=["A", "B"], allowed=[True, True])
if6 = CategoricalDescriptorInput(
    key="if6",
    categories=["A", "B", "C"],
    descriptors=["d1", "d2"],
    values=[[1, 2], [3, 7], [5, 1]],
)
if7 = DiscreteInput(key="if7", values=[0, 1, 5])

of1 = ContinuousOutput(key="of1")

c1 = LinearEqualityConstraint(features=["if0", "if1"], coefficients=[1, 1], rhs=1)
c2 = LinearInequalityConstraint(features=["if0", "if1"], coefficients=[1, 1], rhs=1)
c3 = NonlinearEqualityConstraint(expression="if0**2 + if1**2 - 1")
c4 = NonlinearInequalityConstraint(expression="if0**2 + if1**2 - 1")
c5 = NChooseKConstraint(
    features=["if0", "if1", "if2"], min_count=0, max_count=2, none_also_valid=False
)

supported_domains = [
    Domain(
        # continuous features
        input_features=[if0, if1],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # continuous features incl. with fixed values
        input_features=[if0, if1, if2],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # all feature types
        input_features=[if1, if3, if6, if7],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # all feature types incl. with fixed values
        input_features=[if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # all feature types, linear equality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c1],
    ),
    Domain(
        # all feature types, linear inequality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c2],
    ),
    Domain(
        # all feature types, nonlinear inequality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c4],
    ),
]

unsupported_domains = [
    Domain(
        # nonlinear equality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c3],
    ),
    Domain(
        # combination of linear equality and nonlinear inequality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c1, c4],
    ),
    # Domain(
    #     # n-choose-k
    #     input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
    #     output_features=[of1],
    #     constraints=[c5],
    # ),
]

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


@pytest.mark.parametrize(
    "domain, n_proposals",
    [
        (supported_domains[0], 3),
        (supported_domains[1], 3),
        (supported_domains[2], 3),
        (supported_domains[3], 3),
        (supported_domains[5], 1)
        # (supported_domains[5], 1),
        # (supported_domains[6], 1)
    ],
)
def test_ask(domain, n_proposals):
    rand_strat = RandomStrategy(domain=domain)
    random_data = rand_strat.ask(12)
    for i, outvar in enumerate(domain.outputs.get_keys()):
        lb = domain.outputs[i].dict()["objective"]["lower_bound"]
        ub = domain.outputs[i].dict()["objective"]["upper_bound"]
        random_data[outvar] = np.random.rand(random_data.shape[0]) * (ub - lb) + lb
    domain.experiments = random_data
    rf_strat = RandomForest(domain=domain)
    proposals = rf_strat.ask(n_proposals)
    assert proposals.shape[0] == n_proposals
