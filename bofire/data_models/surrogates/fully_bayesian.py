from typing import Literal, Type

from pydantic import Field, field_validator
from typing_extensions import Annotated

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class SaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["SaasSingleTaskGPSurrogate"] = "SaasSingleTaskGPSurrogate"
    warmup_steps: Annotated[int, Field(ge=1)] = 256
    num_samples: Annotated[int, Field(ge=1)] = 128
    thinning: Annotated[int, Field(ge=1)] = 16

    @field_validator("thinning")
    @classmethod
    def validate_thinning(cls, thinning, info):
        if info.data["num_samples"] / thinning < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return value

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
