from typing import Union

from bofire.data_models.objectives.categorical import (
    CategoricalObjective,
    ConstrainedCategoricalObjective,
)
from bofire.data_models.objectives.identity import (
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
)
from bofire.data_models.objectives.objective import Objective
from bofire.data_models.objectives.sigmoid import (
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    SigmoidObjective,
)
from bofire.data_models.objectives.target import (
    CloseToTargetObjective,
    ConstrainedObjective,
    TargetObjective,
)

AbstractObjective = Union[
    Objective,
    IdentityObjective,
    SigmoidObjective,
    ConstrainedObjective,
    CategoricalObjective,
]

AnyCategoricalObjective = ConstrainedCategoricalObjective

AnyConstraintObjective = Union[
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
]

AnyRealObjective = Union[MaximizeObjective, MinimizeObjective, CloseToTargetObjective]

AnyObjective = Union[
    MaximizeObjective,
    MinimizeObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
    CloseToTargetObjective,
    ConstrainedCategoricalObjective,
]
