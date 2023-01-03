from abc import abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pydantic import Field, validator

from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.domain.util import BaseModel
from bofire.utils.enum import OutputFilteringEnum


class Model(BaseModel):

    input_features: InputFeatures
    output_features: OutputFeatures
    input_preprocessing_specs: Dict = Field(default_factory=lambda: {})

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        # we also validate the number of input features here
        if len(values["input_features"]) == 0:
            raise ValueError("At least one input feature has to be provided.")
        v = values["input_features"]._validate_transform_specs(v)
        return v

    @validator("output_features")
    def validate_output_features(cls, v, values):
        if len(v) == 0:
            raise ValueError("At least one output feature has to be provided.")
        return v

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # validate
        X = self.input_features.validate_inputs(X)
        # transform
        Xt = self.input_features.transform(X, self.input_preprocessing_specs)
        # predict
        preds, stds = self._predict(Xt)
        # postprocess
        return pd.DataFrame(
            data=np.hstack((preds, stds)),
            columns=["%s_pred" % featkey for featkey in self.output_features.get_keys()]
            + ["%s_sd" % featkey for featkey in self.output_features.get_keys()],
        )

    @abstractmethod
    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.array, np.array]:
        pass


class TrainableModel:

    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def fit(self, experiments: pd.DataFrame):
        # preprocess
        experiments = self._preprocess_experiments(experiments)
        # validate
        X = self.input_features.validate_inputs(experiments)
        Y = experiments[self.output_features.get_keys()].values
        # transform
        transformed_X = self.input_features.transform(
            X, self.input_preprocessing_specs
        ).values
        # fit
        self._fit(X=transformed_X, Y=Y)

    def _preprocess_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        if self._output_filtering is None:
            return experiments
        elif self._output_filtering == OutputFilteringEnum.ALL:
            return self.output_features.preprocess_experiments_all_valid_outputs(
                experiments=experiments,
                output_feature_keys=self.output_features.get_keys(),
            )
        elif self._output_filtering == OutputFilteringEnum.ANY:
            return self.output_features.preprocess_experiments_any_valid_outputs(
                experiments=experiments,
                output_feature_keys=self.output_features.get_keys(),
            )
        else:
            raise ValueError("Unknown output filtering option requested.")

    @abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    def cross_validate(self, experiments: pd.DataFrame):
        return