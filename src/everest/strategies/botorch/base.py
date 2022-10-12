import copy
import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from everest.domain.constraints import (ConcurrencyConstraint,
                                        LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.domain.features import (CategoricalDescriptorInputFeature,
                                     CategoricalInputFeature,
                                     ContinuousInputFeature,
                                     ContinuousOutputFeature,
                                     ContinuousOutputFeature_woDesFunc,
                                     InputFeature, OutputFeature,
                                     is_continuous)
from everest.strategies.botorch import tkwargs
from everest.strategies.botorch.utils.models import get_and_fit_model
from everest.strategies.strategy import (CategoricalEncodingEnum,
                                         CategoricalMethodEnum,
                                         DescriptorEncodingEnum,
                                         DescriptorMethodEnum, KernelEnum,
                                         ModelPredictiveStrategy,
                                         RandomStrategy, ScalerEnum)
from everest.strategies.utils import is_power_of_two
from everest.utils.transformer import Transformer
from pydantic import Field, PositiveInt
from pydantic.class_validators import validator

from botorch.acquisition import MCAcquisitionObjective
from botorch.cross_validation import gen_loo_cv_folds
from botorch.models import MixedSingleTaskGP, ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.samplers import SobolQMCNormalSampler

# this is aichembuddy's logger
logger = logging.getLogger("aichembuddy.botorch.sobo")
logger.setLevel(logging.WARNING)

# we set also the log level of botorch to info to see what is going on there
from botorch.settings import log_level

log_level(logging.WARNING)

class BotorchBasicBoStrategy(ModelPredictiveStrategy):
    num_sobol_samples: PositiveInt = 512
    num_restarts: PositiveInt = 8
    num_raw_samples: PositiveInt = 1024
    descriptor_encoding: DescriptorEncodingEnum  = DescriptorEncodingEnum.DESCRIPTOR # set defaults, cause when you have only continuous features its annoying to define categorical stuff
    descriptor_method : DescriptorMethodEnum = DescriptorMethodEnum.EXHAUSTIVE
    categorical_encoding: CategoricalEncodingEnum = CategoricalEncodingEnum.ORDINAL
    categorical_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    objective: Optional[MCAcquisitionObjective]
    sampler: Optional[SobolQMCNormalSampler]
    model: Optional[GPyTorchModel]
    transformer: Optional[Transformer] 
    features2idx: Dict = Field(default_factory=lambda: {})
    input_feature_keys: List[str] = Field(default_factory=lambda: [])
    is_fitted: bool = False
    use_combined_bounds:bool = True # parameter to switch to legacy behavior

    @validator("num_sobol_samples")
    def validate_num_sobol_samples(cls, v):
        if is_power_of_two(v) == False:
            raise ValueError(
                "number sobol samples have to be of the power of 2 to increase performance"
            )
        return v

    @validator("num_raw_samples")
    def validate_num_raw_samples(cls, v):
        if is_power_of_two(v) == False:
            raise ValueError(
                "number raw samples have to be of the power of 2 to increase performance"
            )
        return v
    
    @validator("categorical_method")
    def validate_descriptor_method(cls, v, values):
        if v == CategoricalMethodEnum.FREE and values["categorical_encoding"]==CategoricalEncodingEnum.ORDINAL:
            raise ValueError(
                "Categorical encoding is incompatible with chosen handling method"
            )
        return v

    def _init_domain(self):
        """set up the transformer and the objective
        """
        if self.descriptor_encoding == DescriptorEncodingEnum.CATEGORICAL:
            self.descriptor_method = DescriptorMethodEnum(self.categorical_method.value)

        self.transformer = Transformer(
            domain=self.domain,
            descriptor_encoding=self.descriptor_encoding,
            categorical_encoding=self.categorical_encoding,
            scale_inputs = None,
            scale_outputs = None
        )

        for feat in self.domain.get_feature_keys(InputFeature):
            tr = self.transformer.features2transformedFeatures.get(feat, [feat])
            self.features2idx[feat] = (
                np.array(range(len(tr))) + len(self.input_feature_keys)
            ).tolist()
            self.input_feature_keys += tr

        torch.manual_seed(self.seed)

        self.sampler = SobolQMCNormalSampler(self.num_sobol_samples)  # ,seed=self.seed)
        self.init_objective()

    # helper functions
    def get_model_spec(self, output_feature_key):
        for spec in self.model_specs:
            if spec.output_feature == output_feature_key:
                return spec
        raise ValueError("No model_spec found for feature %s" % output_feature_key)

    def get_feature_indices(self,output_feature_key):
        indices = []
        for key in self.domain.get_feature_keys(InputFeature):
            if key in self.get_model_spec(output_feature_key).input_features:
                indices += self.features2idx[key]
        return indices

    def get_training_tensors(self,transformed: pd.DataFrame, output_feature_key: str):
        train_X = torch.from_numpy(
            transformed[self.input_feature_keys].values
        ).to(**tkwargs)
        train_Y = torch.from_numpy(
            transformed[output_feature_key].values.reshape([-1, 1])
        ).to(**tkwargs)
        return train_X, train_Y
    
    def _fit(self, transformed: pd.DataFrame):
        """[summary]

        Args:
            transformed (pd.DataFrame): [description]
        """
        models = []
        for i, ofeat in enumerate(self.domain.get_features(ContinuousOutputFeature, exact=True)):
            transformed_temp = self.domain.preprocess_experiments_one_valid_output(transformed, ofeat.key)
            train_X, train_Y = self.get_training_tensors(transformed_temp, ofeat.key)

            models.append(
                get_and_fit_model(
                    train_X=train_X,
                    train_Y=train_Y,
                    active_dims=self.get_feature_indices(ofeat.key),
                    cat_dims=self.categorical_dims,
                    scaler_name=self.get_model_spec(ofeat.key).get("scaler", ScalerEnum.NORMALIZE),
                    bounds = self.get_bounds(optimize=False) if self.use_combined_bounds else None,
                    kernel_name=self.get_model_spec(ofeat.key).get("kernel", KernelEnum.MATERN_25),
                    use_ard=self.get_model_spec(ofeat.key).get("ard", True),
                    use_categorical_kernel=self.categorical_encoding==CategoricalEncodingEnum.ORDINAL
                )
            )
        if len(models) == 1:
            self.model = models[0]
        else:
            self.model = ModelListGP(*models)
        self.is_fitted = True
        return

    def _predict(self, transformed: pd.DataFrame):
        X = torch.from_numpy(transformed[self.input_feature_keys].values).to(**tkwargs)
        preds = self.model.posterior(X=X).mean.cpu().detach().numpy()
        stds = np.sqrt(self.model.posterior(X=X).variance.cpu().detach().numpy())
        return preds, stds

    def _calc_acquisition(self, transformed: pd.DataFrame, combined: bool = False):
        X = torch.from_numpy(transformed[self.input_feature_keys].values).to(**tkwargs)
        if combined is False:
            X = X.unsqueeze(-2)
        return self.acqf.forward(X).cpu().detach().numpy()
        
    @property
    def categorical_dims(self):
        desc_categorical_features = self.domain.get_features(
            CategoricalDescriptorInputFeature
        )
        categorical_features = self.domain.get_features(CategoricalInputFeature,exact=True)
        indices = []
        for feat in categorical_features:
            indices += self.features2idx[feat.key]
        if self.descriptor_encoding != DescriptorEncodingEnum.DESCRIPTOR:
            for feat in desc_categorical_features:
                indices += self.features2idx[feat.key]
        return indices

    def _ask(self, candidate_count: int) -> Tuple[pd.DataFrame, List[dict]]:

        """[summary]

        Args:
            candidate_count (int, optional): [description]. Defaults to 1.

        Returns:
            pd.DataFrame: [description]
        """

        assert candidate_count > 0, "candidate_count has to be larger than zero."
        logger.info(
            "%i experiments requested with num_restarts: %i, raw_samples: %i"
            % (candidate_count, self.num_restarts, self.num_raw_samples)
        )

        # optimize
        # we have to distuinguish the following scenarios
        # - no categoricals - check
        # - categoricals with one hot and free variables
        # - categoricals with one hot and exhaustive screening, could be in combination with garrido merchan - check
        # - categoricals with one hot and OEN, could be in combination with garrido merchan - OEN not implemented
        # - descriptized categoricals not yet implemented
        num_categorical_features = len(
            self.domain.get_features(CategoricalInputFeature)
        )
        num_categorical_combinations = len(self.domain.get_categorical_combinations())

        if (
            (num_categorical_features == 0)
            or (num_categorical_combinations == 1)
            or ((self.categorical_method == CategoricalMethodEnum.FREE) and (self.descriptor_method == DescriptorMethodEnum.FREE))
        ) and len(self.domain.get_constraints(ConcurrencyConstraint)) == 0:
            candidates = optimize_acqf(
                acq_function=self.acqf,
                bounds=self.get_bounds(),
                q=candidate_count,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearEqualityConstraint
                ),
                inequality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearInequalityConstraint
                ),
                fixed_features=self.get_fixed_features(),
                return_best_only=True,
            )
            # options={"seed":self.seed})

        elif (
            (self.categorical_method == CategoricalMethodEnum.EXHAUSTIVE) 
            or (self.descriptor_method == DescriptorMethodEnum.EXHAUSTIVE)
        ) and len(self.domain.get_constraints(ConcurrencyConstraint)) == 0:
            # TODO: marry this withe groups of XY
            candidates = optimize_acqf_mixed(
                acq_function=self.acqf,
                bounds=self.get_bounds(),
                q=candidate_count,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearEqualityConstraint
                ),
                inequality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearInequalityConstraint
                ),
                fixed_features_list=self.get_categorical_combinations(),
            )
            # options={"seed":self.seed})

        elif len(self.domain.get_constraints(ConcurrencyConstraint)) > 0:
            candidates = optimize_acqf_mixed(
                acq_function=self.acqf,
                bounds=self.get_bounds(),
                q=candidate_count,
                num_restarts=self.num_restarts,
                raw_samples=self.num_raw_samples,
                equality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearEqualityConstraint
                ),
                inequality_constraints=RandomStrategy.get_linear_constraints(
                    self.domain, LinearInequalityConstraint
                ),
                fixed_features_list=self.get_fixed_values_list(),
            )

        else:
            raise IOError()

        # postprocess the results
        # TODO: in case of free we have to transform back the candidates first and then compute the metrics
        # otherwise the prediction holds only for the infeasible solution, this solution should then also be
        # applicable for >1d descriptors
        preds = self.model.posterior(X=candidates[0]).mean.detach().numpy()
        stds = np.sqrt(
            self.model.posterior(X=candidates[0]).variance.detach().numpy()
        )

        df_candidates = pd.DataFrame(
            data=np.nan,
            index=range(candidate_count),
            columns=self.input_feature_keys
            + [i + "_pred" for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            + [i + "_sd" for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            + [i + "_des" for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            # ["reward","acqf","strategy"]
        )

        for i, feat in enumerate(self.domain.get_features(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])):
            df_candidates[feat.key + "_pred"] = preds[:, i]
            df_candidates[feat.key + "_sd"] = stds[:, i]
            df_candidates[feat.key + "_des"] = feat.desirability_function(preds[:, i])

        df_candidates[self.input_feature_keys] = candidates[0].detach().numpy()

        # additional information is no further stored in df_candidates
        # df_candidates["reward"] = self.objective.forward(samples=self.trainer.model.posterior(candidates[0]).mean,X=None).detach().numpy()
        # df_candidates["acqf"]=candidates[1].detach().numpy()
        # df_candidates["strategy"] = self.name

        configs = self.get_candidate_log(candidates)
        return self.transformer.inverse_transform(df_candidates), configs

    def _tell(self) -> None:
        if self.has_sufficient_experiments():
            # todo move this up to predictive strategy
            self.fit()
            self.init_acqf()
        return

    def init_acqf(self) -> None:
        self._init_acqf()
        return

    @abstractmethod
    def _init_acqf(self,) -> None:
        pass

    def init_objective(self) -> None:
        self._init_objective()
        return

    @abstractmethod
    def _init_objective(self,) -> None:
        pass

    def get_bounds(self, optimize=True):
        """[summary]

        Raises:
            IOError: [description]

        Returns:
            [type]: [description]
        """
        lower = []
        upper = []

        for var in self.domain.get_features(InputFeature):
            if isinstance(var, ContinuousInputFeature):
                if optimize:
                    lower.append(var.lower_bound)
                    upper.append(var.upper_bound)
                else:
                    lb, ub = var.get_real_feature_bounds(self.experiments[var.key])
                    lower.append(lb), upper.append(ub)
            elif isinstance(var, CategoricalInputFeature):
                if (
                    isinstance(var, CategoricalDescriptorInputFeature)
                    and self.descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
                ):
                    if optimize:
                        df = var.to_df().loc[var.get_allowed_categories()]
                        lower += df.min().values.tolist()
                        upper += df.max().values.tolist()
                    else:
                        df = var.get_real_descriptor_bounds(self.experiments[var.key])
                        lower += df.loc["lower"].tolist()
                        upper += df.loc["upper"].tolist()
                elif self.categorical_encoding == CategoricalEncodingEnum.ORDINAL:
                    lower.append(0)
                    upper.append(len(var.categories)-1)
                else:
                    for _ in var.categories:
                        lower.append(0.0)
                        upper.append(1.0) 
            else:
                raise IOError("Feature type not known!")

        return torch.tensor([lower, upper]).to(**tkwargs)

    def get_fixed_features(self): 
        """provides the values of all fixed features

        Raises:
            NotImplementedError: [description]

        Returns:
            fixed_features (dict): Dictionary of fixed features, keys are the feature indices, values the transformed feature values
        """
        fixed_features = {}
        # we need the transform object now with instantiated encoders which means it has to be called at least once before
        # get_fixed_features is called
        if not self.transformer.is_fitted:
            if self.experiments is not None:
                experiments = self.experiments
            else:
                #TODO: catch if no experiements are provided
                print("Call strategy.tell first. The transfomer needs to be fitted here")
            _ = self.transformer.fit_transform(experiments) 

        for _, var in enumerate(self.domain.get_features(InputFeature)):
            if var.fixed_value() is not None:
                if is_continuous(var):
                    # we use the scaler in botorch and thus, we have no scaler stored in transfrom.encoders by convention
                    # if var.key in self.transform.encoders.keys():
                    #     fixed_features[self.transform.features2idx[var.key][0]]= self.transform.encoders[var.key].transform(var.fixed_value())
                    # else:
                    fixed_features[self.features2idx[var.key][0]] = var.fixed_value()

                elif (
                    isinstance(var, CategoricalDescriptorInputFeature)
                    and self.descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
                ):
                    for j, idx in enumerate(self.features2idx[var.key]):
                        category_index = var.categories.index(var.fixed_value())
                        # values = var.values[category_index][j]
                        # if var.key in self.transform.encoders.keys():
                        #     fixed_features[idx]= self.transform.encoders[var.descriptors[j]].transform(values)
                        # else:
                        fixed_features[idx] = var.values[category_index][j]
                
                elif isinstance(var, CategoricalInputFeature):
                    if self.categorical_encoding == CategoricalEncodingEnum.ONE_HOT:
                        transformed = (
                            self.transformer.encoders[var.key]
                            .transform(np.array([[var.fixed_value()]]))
                            .toarray()
                        )
                        for j, idx in enumerate(self.features2idx[var.key]):
                            fixed_features[idx] = transformed[0, j] 
                    elif self.categorical_encoding == CategoricalEncodingEnum.ORDINAL:
                        transformed = self.transformer.encoders[var.key].transform(
                            np.array([[var.fixed_value()]])
                        )
                        fixed_features[
                            self.features2idx[var.key][0]
                        ] = transformed[0][0]
                    else:
                        pass
                else:
                    raise NotImplementedError(
                        "The feature type %s is not known" % var.__class__.__name__
                    )
        # in case the optimization method is free and not allowed categories are present 
        # one has to fix also them
        if self.categorical_method == CategoricalMethodEnum.FREE and self.categorical_encoding==CategoricalEncodingEnum.ONE_HOT:
            for feat in self.get_true_categorical_features():
                if feat.is_fixed() == False:
                    for cat in feat.get_forbidden_categories():
                        transformed = (
                                self.transformer.encoders[feat.key]
                                .transform(np.array([[cat]]))
                                .toarray()
                            )
                        # we fix those indices to zero where one has a 1 as response from the transformer
                        for j, idx in enumerate(self.features2idx[feat.key]):
                            if transformed[0,j] == 1.:
                                fixed_features[idx] = 0
        return fixed_features

    def get_true_categorical_features(self) -> list:
        """Get those features wich are treated as categoricals, which includes also CategoricalDescriptor features
        if `CATEGORICAL` is used as `descriptor_encoding`.

        Returns:
            list: list of features treated as categoricals
        """
        if self.descriptor_encoding == DescriptorEncodingEnum.CATEGORICAL:
            return self.domain.get_features(CategoricalInputFeature)
        else:
            return self.domain.get_features(CategoricalInputFeature, excludes= [CategoricalDescriptorInputFeature])

    def get_categorical_combinations(self):
        """provides all possible combinations of fixed values 

        Returns:
            list_of_fixed_features List[dict]: Each dict contains a combination of fixed values
        """
        fixed_basis = self.get_fixed_features()
        include = CategoricalInputFeature
        exclude = None
        
        if (
            (self.descriptor_method == DescriptorMethodEnum.FREE) 
            and (self.categorical_method == CategoricalMethodEnum.FREE)
        ):
            return [{}]
        elif self.descriptor_method == DescriptorMethodEnum.FREE:
            exclude=CategoricalDescriptorInputFeature
        elif self.categorical_method == CategoricalMethodEnum.FREE:
            include = CategoricalDescriptorInputFeature
            
        combos = self.domain.get_categorical_combinations(include=include, exclude=exclude)
        # now build up the fixed feature list
        if len(combos) == 1:
            return [fixed_basis]
        else:
            list_of_fixed_features = []

            for combo in combos:
                fixed_features = copy.deepcopy(fixed_basis)

                for pair in combo:
                    feat, val = pair
                    feature = self.domain.get_feature(feat)
                    if (
                        isinstance(feature, CategoricalDescriptorInputFeature)
                        and self.descriptor_encoding
                        == DescriptorEncodingEnum.DESCRIPTOR
                    ):
                        index = feature.categories.index(val)

                        for j, idx in enumerate(self.features2idx[feat]):
                            fixed_features[idx] = feature.values[index][j]

                    elif isinstance(feature, CategoricalInputFeature):
                        if self.categorical_encoding == CategoricalEncodingEnum.ONE_HOT:
                            transformed = (
                                self.transformer.encoders[feat]
                                .transform(np.array([[val]]))
                                .toarray()
                            )

                        elif (
                            self.categorical_encoding == CategoricalEncodingEnum.ORDINAL
                        ):
                            transformed = self.transformer.encoders[feat].transform(
                                np.array([[val]])
                            )

                        for j, idx in enumerate(self.features2idx[feat]):
                            fixed_features[idx] = transformed[0, j]

                list_of_fixed_features.append(fixed_features)
        return list_of_fixed_features

    def get_concurrency_combinations(self):

        '''
        generate a list of fixed values dictionaries from concurrency constraints
        '''

        # generate botorch-friendly fixed values
        used_features, unused_features = self.domain.get_concurrency_combinations()
        fixed_values_list_cc = []
        for used, unused in zip(used_features, unused_features):
            fixed_values = {}

            # sets unused features to zero
            for f_key in unused:
                fixed_values[self.features2idx[f_key][0]] = 0.0
            
            fixed_values_list_cc.append(fixed_values)

        if len(fixed_values_list_cc) == 0:
            fixed_values_list_cc.append({})    # any better alternative here?

        return fixed_values_list_cc

    def get_fixed_values_list(self):

        # CARTESIAN PRODUCTS: fixed values from categorical combinations X fixed values from concurrency constraints
        fixed_values_full = []

        if (
            (self.categorical_method == CategoricalMethodEnum.FREE and self.descriptor_method == DescriptorMethodEnum.FREE)
            or (self.categorical_method == CategoricalMethodEnum.FREE and self.descriptor_encoding == DescriptorEncodingEnum.CATEGORICAL)
        ):
            ff1 = self.get_fixed_features()
            for ff2 in self.get_concurrency_combinations():
                ff = ff1.copy()
                ff.update(ff2)
                fixed_values_full.append(ff)
        else:
            for ff1 in self.get_categorical_combinations():
                for ff2 in self.get_concurrency_combinations():
                    ff = ff1.copy()
                    ff.update(ff2)
                    fixed_values_full.append(ff)


        return fixed_values_full

    def has_sufficient_experiments(self,) -> bool:
        if self.experiments is None:
            return False
        degrees_of_freedom =  len(self.domain.get_features(InputFeature)) - len(self.get_fixed_features()) 
        #degrees_of_freedom = len(self.domain.get_features(InputFeature)) + 1
        if self.experiments.shape[0] > degrees_of_freedom + 1:
            return True
        return False

    def get_candidate_log(self, candidates):
        return [{"acquisition_value":candidates[1]} for _ in range(len(candidates[0]))]

    # TODO: test this at a later stage at this is not super important for the first release
    def feature_importances(self, plot: bool = False):
        if plot: from everest.utils.viz import plot_fi
        if self.is_fitted != True:
            raise ValueError(
                "Cannot calculate feature_importances without a fitted model."
            )
        elif isinstance(self.model,ModelListGP):
            models = self.model.models
        else:
            models = [self.model]
        importances = {}
        for m, featkey in zip(models,self.domain.get_feature_keys(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])):
            # in case of MixedGP ignore it
            if not isinstance(m,MixedSingleTaskGP) and self.get_model_spec(featkey).get("ard",True):
                ls = m.covar_module.base_kernel.lengthscale
                feature_keys = []
                for key in self.domain.get_feature_keys(InputFeature):
                    if key in self.get_model_spec(featkey).input_features:
                        feature_keys += self.transformer.features2transformedFeatures.get(key,[key])
                fi = (1./ls).detach().cpu().numpy().ravel()
                importances[featkey] = [feature_keys,fi]
                if plot: plot_fi(featurenames = feature_keys, importances = fi, comment = featkey)
        return importances

    def cross_validate(self,transformed: pd.DataFrame, nfolds: int = -1):

        if nfolds != -1:
            raise NotImplementedError("Only LOOCV implemented so far!")

        results = []
        for i, ofeat in enumerate(self.domain.get_features(ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])):
            transformed_temp = self.domain.preprocess_experiments_one_valid_output(transformed, ofeat.key)
            train_X, train_Y = self.get_training_tensors(transformed_temp, ofeat.key)
            
            # create folds
            cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y)

            # setup the model
            model_cv = get_and_fit_model(
                train_X=cv_folds.train_X,
                train_Y=cv_folds.train_Y,
                active_dims=self.get_feature_indices(ofeat.key),
                cat_dims=self.categorical_dims,
                scaler_name=self.get_model_spec(ofeat.key).get("scaler", ScalerEnum.NORMALIZE),
                kernel_name=self.get_model_spec(ofeat.key).get("kernel", KernelEnum.MATERN_25),
                use_ard=self.get_model_spec(ofeat.key).get("ard", True),
                use_categorical_kernel=self.categorical_encoding==CategoricalEncodingEnum.ORDINAL,
                cv=True
            )

            with torch.no_grad():
                posterior = model_cv.posterior(
                    cv_folds.test_X, observation_noise=False
                )

            observed = cv_folds.test_Y.squeeze().detach().numpy()
            mean = posterior.mean.squeeze().detach().numpy
            std = np.sqrt(posterior.variance.squeeze().detach().numpy())
            results.append((observed, mean, std))
        return results
