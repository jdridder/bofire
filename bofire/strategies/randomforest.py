from typing import Optional, Sequence, Tuple, Type

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bofire.domain.constraints import Constraint
from bofire.domain.features import CategoricalInput, Feature
from bofire.domain.objectives import Objective
from bofire.strategies.strategy import PredictiveStrategy
from bofire.utils.enum import CategoricalEncodingEnum


class RandomForest(PredictiveStrategy):
    """Surrogate-assisted optimization with RandomForests.

    The algorithm can handle constrained mixed variable problems with single and multiple objectives.
    It is intended as a cheap alternative to ENTMOOT.

    Description:
    - Model: RandomForest with a distance-based uncertainty measure.
    - Proposals: BayesOpt with random Chebyshef scalarization and optimistic confidence bound.
    - Optimization: Brute-force.
    - Pareto approximation: Not implemented.
    """

    models: Optional[Sequence[RandomForestRegressor]]

    # def __init__(self, problem: opti.Problem, n_samples: int = 10000, **kwargs):
    #     """RandomForest algorithm.

    #     Args:
    #         problem: Problem definition.
    #         n_samples: Number of samples for the brute-force optimization.
    #     """
    #     super().__init__(problem)
    #     self.n_samples = n_samples
    #     self._initialize_problem()

    # def _initialize_problem(self) -> None:
    #     # Check for initial data
    #     if self.problem.data is None:
    #         raise UnsuitableAlgorithmError("RandomForest requires initial data.")

    #     # Estimate range of outputs from observed values.
    #     Y = self.problem.get_Y()
    #     self.y_range = Y.max(axis=0) - Y.min(axis=0)

    #     # Check for output constraints
    #     if self.problem.output_constraints is not None:
    #         raise UnsuitableAlgorithmError(
    #             "Output constraints not implemented for RandomForest."
    #         )

    def _fit(self, experiments: pd.DataFrame) -> None:

        Xt = self.domain.inputs.transform(
            experiments, specs=self.input_preprocessing_specs
        )

        # Fit a model for each target variable
        self.models = []
        for y_name in self.domain.outputs.get_keys():
            y = experiments[y_name].to_numpy()
            self.models.append(RandomForestRegressor().fit(Xt, y))

        return None

    # def _tell(self) -> None:
    #     """custom"""
    #     pass

    # def _min_distance(self, X1: pd.DataFrame, X2: pd.DataFrame):
    #     """Matrix of L1-norm-distances between each point in X1 and X2"""
    #     kwargs = dict(
    #         continuous="normalize", discrete="normalize", categorical="onehot-encode"
    #     )
    #     inputs = self.problem.inputs
    #     Xt1 = inputs.transform(X1, **kwargs)
    #     Xt2 = inputs.transform(X2, **kwargs)

    #     # set all onehot-encode values to 0.5 so that the L1-distance becomes 1
    #     cat_cols = [c for c in Xt1 if "§" in c]
    #     Xt1[cat_cols] /= 2
    #     Xt2[cat_cols] /= 2

    #     D = torch.cdist(torch.tensor(Xt1.values), torch.tensor(Xt2.values)).numpy()
    #     return D.min(axis=1)

    # def _uncertainty(self, X: pd.DataFrame, X_data: Optional[pd.DataFrame] = None):
    #     """Uncertainty estimate ~ distance to the closest data point."""
    #     if X_data is None:
    #         X_data = self.problem.data
    #     min_dist = self._min_distance(X, X_data)
    #     min_dist = min_dist / self.problem.n_inputs
    #     return self.y_range * min_dist[:, np.newaxis]

    def _predict(self, experiments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run predictions for the provided experiments. Only input features have to be provided.

        Args:
            experiments (pd.DataFrame): Experimental data for which predictions should be performed.

        Returns:
            pd.DataFrame: Dataframe with the predicted values.
        """
        if self.models is None:
            raise Exception(
                "RandomForest._predict was called but no models have been fitted yet"
            )
        Xt = experiments  # experiments has already been transformed / preprocessed
        Ym = [model.predict(Xt.to_numpy()) for model in self.models]
        # Ys = self._uncertainty(X)
        ypred = pd.DataFrame(Ym).T
        yuncert = pd.DataFrame(Ym).T

        return ypred, yuncert

    # def _ask(
    #     self,
    #     n_proposals: Optional[int] = None,
    #     kappa: float = 0.5,
    #     weights: Optional[np.ndarray] = None,
    # ) -> pd.DataFrame:
    #     """Was called propose in brain/bo"""
    #     if n_proposals is None:
    #         n_proposals = 1

    #     if weights is None:
    #         n_objectives = len(self.problem.objectives)
    #         weights = opti.sampling.simplex.sample(n_objectives, n_proposals)
    #     else:
    #         weights = np.atleast_2d(weights)

    #     proposals = []

    #     for w in weights:
    #         X = self.problem.sample_inputs(self.n_samples)
    #         Xt = self.problem.inputs.transform(X, categorical="dummy-encode")
    #         Y_mean = pd.DataFrame(
    #             self.model.predict(Xt.values), columns=self.problem.outputs.names
    #         )
    #         Z_mean = self.problem.objectives(Y_mean)

    #         # take the pending proposals into account for the uncertainty
    #         X_data = self.problem.data[self.problem.inputs.names]
    #         X_data = pd.concat([X_data] + proposals, axis=0)
    #         Y_std = self._uncertainty(X, X_data)
    #         Z_std = Y_std  # may not be true for close-to-target objectives

    #         # optimistic confidence bound
    #         cb = Z_mean.to_numpy() - kappa * Z_std
    #         # normalize so that the Pareto front is in the unit range
    #         s = opti.metric.is_pareto_efficient(cb)
    #         cb_min = cb[s].min(axis=0)
    #         cb_max = cb[s].max(axis=0)
    #         cb = (cb - cb_min) / np.clip(cb_max - cb_min, 1e-7, None)

    #         # weighted max norm
    #         A = np.max(w * cb, axis=1)
    #         best = np.argmin(A)
    #         proposals.append(X.iloc[[best]])

    #     return pd.concat(proposals)

    # def get_model_parameters(self) -> pd.DataFrame:
    #     # get the columns labels for the one-hot encoded inputs in case there are categoricals
    #     cols = self.problem.inputs.transform(
    #         self.problem.data, categorical="dummy-encode"
    #     ).columns

    #     # return the feature importances
    #     m = self.problem.n_outputs
    #     params = pd.DataFrame(
    #         index=self.problem.outputs.names,
    #         data=np.tile(self.model.feature_importances_, reps=(m, 1)),
    #         columns=cols,
    #     )
    #     params.index.name = "output"
    #     return params

    # def to_config(self) -> dict:
    #     return {
    #         "method": "RandomForest",
    #         "problem": self.problem.to_config(),
    #         "parameters": {"n_samples": self.n_samples},
    #     }

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise
        """
        return True

    def _choose_from_pool(self):
        pass

    def has_sufficient_experiments(self):
        pass

    @property
    def input_preprocessing_specs(self):
        return {
            key: CategoricalEncodingEnum.DUMMY
            for key in self.domain.inputs.get_keys(CategoricalInput)
        }
