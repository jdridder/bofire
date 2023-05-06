from typing import Callable, Dict, List, Optional

import base64
import io
import warnings

import botorch
import pandas as pd
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import ChainedInputTransform, OneHotToNumeric
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.data_models.enum import CategoricalEncodingEnum, OutputFilteringEnum, MolecularEncodingEnum
from bofire.data_models.surrogates.api import MixedTanimotoGPSurrogate as DataModel
from bofire.data_models.kernels.continuous import MaternKernel
from bofire.data_models.kernels.categorical import HammingDistanceKernel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_scaler
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors import GammaPrior
from torch import Tensor


class MixedTanimotoGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        mol_dims: List[int],
        mol_kernel_factory: Callable[[torch.Size, int, List[int]], Kernel],
        cat_dims: Optional[List[int]] = None,
        cont_kernel_factory: Optional[Callable[[torch.Size, int, List[int]], Kernel]] = None,
        cat_kernel_factory: Optional[Callable[[torch.Size, int, List[int]], Kernel]] = None,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: Optional[OutcomeTransform] = None,  # TODO
        input_transform: Optional[InputTransform] = None,  # TODO
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            cat_dims: A list of indices corresponding to the columns of
                the input `X` that should be considered categorical features.
            cont_kernel_factory: A method that accepts  `batch_shape`, `ard_num_dims`,
                and `active_dims` arguments and returns an instantiated GPyTorch
                `Kernel` object to be used as the base kernel for the continuous
                dimensions. If omitted, this model uses a Matern-2.5 kernel as
                the kernel for the ordinal parameters.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass. Only input transforms are allowed which do not
                transform the categorical dimensions. If you want to use it
                for example in combination with a `OneHotToNumeric` input transform
                one has to instantiate the transform with `transform_on_train` == False
                and pass in the already transformed input.
        """
        if len(mol_dims) == 0:
            raise ValueError(
                "Must specify molecular dimensions for MixedTanimotoGP"
            )

        if cat_dims is None:
            cat_dims = []

        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)

        if cont_kernel_factory is None:
            cont_kernel_factory = ScaleKernel(base_kernel=MaternKernel(ard=True, nu=2.5)).to_gpytorch()

        if cat_kernel_factory is None:
            cat_kernel_factory = ScaleKernel(base_kernel=HammingDistanceKernel(ard=True)).to_gpytorch()

        if likelihood is None:
            # This Gamma prior is quite close to the Horseshoe prior
            min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
            likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_noise, transform=None, initial_value=1e-3
                ),
                noise_prior=GammaPrior(0.9, 10.0),
            )

        d = train_X.shape[-1]
        mol_dims = normalize_indices(indices=mol_dims, d=d)
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims) - set(mol_dims))

        if len(mol_dims) == d:
            covar_module = mol_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(mol_dims),
                )

        elif len(cat_dims) == 0:
            sum_kernel = cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                ) + mol_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(mol_dims),
                active_dims=mol_dims,
            )

            prod_kernel = cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                ) * mol_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(mol_dims),
                active_dims=mol_dims,
            )

            covar_module = sum_kernel + prod_kernel

        else:
            sum_kernel = cont_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            ) + mol_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(mol_dims),
                active_dims=mol_dims,
            ) + cat_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
            )

            prod_kernel = cont_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            ) * mol_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(mol_dims),
                active_dims=mol_dims,
            ) * cat_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
            )
            covar_module = sum_kernel + prod_kernel

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )


class MixedTanimotoGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.continuous_kernel = data_model.continuous_kernel
        self.categorical_kernel = data_model.categorical_kernel
        self.molecular_kernel = data_model.molecular_kernel
        self.scaler = data_model.scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.MixedSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        features2idx, _ = self.inputs._get_transform_info(
            self.input_preprocessing_specs
        )
        non_numerical_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR and value != MolecularEncodingEnum.FINGERPRINTS and value != MolecularEncodingEnum.FRAGMENTS and value != MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS and value != MolecularEncodingEnum.MOL_DESCRIPTOR
        ]

        ord_dims = []
        for feat in self.inputs.get():
            if feat.key not in non_numerical_features:
                ord_dims += features2idx[feat.key]

        # these are the categorical dimensions after applying the OneHotToNumeric transform
        cat_dims = list(
            range(len(ord_dims), len(ord_dims) + len(non_numerical_features))
        )
        # these are the categorical features within the OneHotToNumeric transform
        categorical_features = {
            features2idx[feat][0]: len(features2idx[feat])
            for feat in non_numerical_features
        }

        o2n = OneHotToNumeric(
            dim=tX.shape[1],
            categorical_features=categorical_features,
            transform_on_train=False,
        )
        tf = ChainedInputTransform(tf1=scaler, tf2=o2n)

        # So far, looking only for features that apply to tanimoto kernel
        molecular_features = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value == MolecularEncodingEnum.FINGERPRINTS or value == MolecularEncodingEnum.FRAGMENTS or value == MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS
        ]
        mol_dims = []
        for mol_feat in molecular_features:
            for i in features2idx[mol_feat]:
                mol_dims.append(i)

        # fit the model
        self.model = MixedTanimotoGP(
            train_X=o2n.transform(tX),
            train_Y=tY,
            cat_dims=cat_dims,
            mol_dims=mol_dims,
            cont_kernel_factory=self.continuous_kernel.to_gpytorch,
            cat_kernel_factory=self.categorical_kernel.to_gpytorch,
            mol_kernel_factory=self.molecular_kernel.to_gpytorch,
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=tf,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)

    def _dumps(self) -> str:
        """Dumps the actual model to a string via pickle as this is not directly json serializable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO()
        torch.save(self.model, buffer, pickle_module=cloudpickle_module)  # type: ignore
        return base64.b64encode(buffer.getvalue()).decode()
        # return codecs.encode(pickle.dumps(self.model), "base64").decode()

    def loads(self, data: str):
        """Loads the actual model from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, pickle_module=cloudpickle_module)  # type: ignore