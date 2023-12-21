from typing import Literal, Type

from pydantic import Field, validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import (
    AnyCategoricalKernal,
    AnyContinuousKernel,
    HammondDistanceKernel,
    MaternKernel,
)
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MixedSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["MixedSingleTaskGPSurrogate"] = "MixedSingleTaskGPSurrogate"
    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(ard=True, nu=2.5)
    )
    categorical_kernel: AnyCategoricalKernal = Field(
        default_factory=lambda: HammondDistanceKernel(ard=True)
    )

    @validator("input_preprocessing_specs")
    def validate_categoricals(cls, v, values):
        """Checks that at least one one-hot encoded categorical feauture is present."""
        if CategoricalEncodingEnum.ONE_HOT not in v.values():
            raise ValueError(
                "MixedSingleTaskGPSurrogate can only be used if at least one one-hot encoded categorical feature is present."
            )
        return v

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
