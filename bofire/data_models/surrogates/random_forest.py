from typing import Literal, Optional, Union

from pydantic import Field, validator
from typing_extensions import Annotated

from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class RandomForestSurrogate(TrainableBotorchSurrogate):
    type: Literal["RandomForestSurrogate"] = "RandomForestSurrogate"

    # hyperparams passed down to `RandomForestRegressor`
    n_estimators: int = 100
    criterion: Literal[
        "squared_error",
        "absolute_error",
        "friedman_mse",
        "poisson",
    ] = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[int, float, Literal["auto", "sqrt", "log2"]] = 1.0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    random_state: Optional[int] = None
    ccp_alpha: Annotated[float, Field(ge=0)] = 0.0
    max_samples: Optional[Union[int, float]] = None

    @validator("outputs")
    def validate_outputs(cls, outputs):
        """validates outputs

        Raises:
            ValueError: if output type is not ContinuousOutput

        Returns:
            List[ContinuousOutput]
        """
        for o in outputs:
            if not isinstance(o, ContinuousOutput):
                raise ValueError("all outputs need to be continuous")
        return outputs
