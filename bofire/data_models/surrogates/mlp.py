from typing import Annotated, Literal, Sequence, Type

from pydantic import Field

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MLPEnsemble(TrainableBotorchSurrogate):
    type: Literal["MLPEnsemble"] = "MLPEnsemble"
    n_estimators: Annotated[int, Field(ge=1)] = 5
    hidden_layer_sizes: Sequence = (100,)
    activation: Literal["relu", "logistic", "tanh"] = "relu"
    dropout: Annotated[float, Field(ge=0.0)] = 0.0
    batch_size: Annotated[int, Field(ge=1)] = 10
    n_epochs: Annotated[int, Field(ge=1)] = 200
    lr: Annotated[float, Field(gt=0.0)] = 1e-4
    weight_decay: Annotated[float, Field(ge=0.0)] = 0.0
    subsample_fraction: Annotated[float, Field(gt=0.0)] = 1.0
    shuffle: bool = True
    scaler: ScalerEnum = ScalerEnum.NORMALIZE

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models

        Args:
            outputs: objective functions for the surrogate
            my_type: continuous or categorical output

        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return True if isinstance(my_type, ContinuousOutput) else False
