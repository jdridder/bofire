from typing import Literal

from pydantic import conint, validator

from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class SaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["SaasSingleTaskGPSurrogate"] = "SaasSingleTaskGPSurrogate"
    warmup_steps: conint(ge=1) = 256  # type: ignore
    num_samples: conint(ge=1) = 128  # type: ignore
    thinning: conint(ge=1) = 16  # type: ignore

    @validator("thinning")
    def validate_thinning(cls, value, values):
        if values["num_samples"] / value < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return value

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
