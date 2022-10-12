import random

import numpy as np
import pytest

from everest.strategies.botorch.qparego import BoTorchQparegoStrategy, AcquisitionFunctionEnum
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective, WeightedMCMultiOutputObjective)
from everest.benchmarks.multiobjective import DTLZ2
from everest.strategies.botorch.base import (CategoricalEncodingEnum,
                                             CategoricalMethodEnum,
                                             DescriptorEncodingEnum,
                                             DescriptorMethodEnum)
from everest.strategies.strategy import RandomStrategy
from everest.strategies.tests.test_model_spec import VALID_MODEL_SPEC_LIST
from everest.utils.tests.test_multiobjective import (dfs, invalid_domains,
                                                     valid_domains)

VALID_BOTORCH_QPAREGO_STRATEGY_SPEC = {
    # "num_sobol_samples": 1024,
    # "num_restarts": 8,
    # "num_raw_samples": 1024,
    "descriptor_encoding": random.choice(list(DescriptorEncodingEnum)),
    "descriptor_method": random.choice(list(DescriptorMethodEnum)),
    "categorical_encoding": random.choice(list(CategoricalEncodingEnum)),
    "base_acquisition_function": random.choice(list(AcquisitionFunctionEnum)),
    "categorical_method": "EXHAUSTIVE",
}

BOTORCH_QPAREGO_STRATEGY_SPECS = {
    "valids": [
        VALID_BOTORCH_QPAREGO_STRATEGY_SPEC,
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": 1},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "model_specs": VALID_MODEL_SPEC_LIST},
    ],
    "invalids": [
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "descriptor_encoding": None},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "categorical_encoding": None},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "categorical_encoding": "ORDINAL", "categorical_method": "FREE"},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize("domain", [
    invalid_domains[0],
    invalid_domains[1],
])
def test_invalid_qparego_init_domain(domain):
    with pytest.raises(ValueError):
        BoTorchQparegoStrategy.from_domain(domain)



@pytest.mark.parametrize("num_test_candidates, base_acquisition_function", [
    (num_test_candidates, base_acquisition_function)
    for num_test_candidates in range(1,3)
    for base_acquisition_function in list(AcquisitionFunctionEnum)
])
def test_qparego(num_test_candidates, base_acquisition_function):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = RandomStrategy.from_domain(benchmark.domain)
    experiments = benchmark.run_candidate_experiments(random_strategy.ask(candidate_count=10)[0])
    # init strategy
    my_strategy = BoTorchQparegoStrategy.from_domain(benchmark.domain, base_acquisition_function=base_acquisition_function)
    my_strategy.tell(experiments)
    # ask
    candidates, _ = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates
