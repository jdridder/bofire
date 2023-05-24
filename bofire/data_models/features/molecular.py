from typing import ClassVar, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import validator

from bofire.data_models.enum import CategoricalEncodingEnum, MolecularEncodingEnum
from bofire.data_models.features.categorical import (
    _CAT_SEP,
    CategoricalInput,
    TTransform,
)
from bofire.data_models.features.feature import Input
from bofire.utils.cheminformatics import (
    smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2mol,
)


class MolecularInput(Input):
    type: Literal["MolecularInput"] = "MolecularInput"
    order: ClassVar[int] = 6

    def validate_experimental(
        self, values: pd.Series, strict: bool = False
    ) -> pd.Series:
        for smi in values:
            smiles2mol(smi)
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        for smi in values:
            smiles2mol(smi)
        return values

    def fixed_value(self, transform_type: Optional[TTransform] = None) -> None:
        return None

    def is_fixed(self) -> bool:
        return False

    # TODO: model descriptors as pydantic class
    def to_fingerprints(
        self, values: pd.Series, bond_radius: int = 5, n_bits: int = 2048
    ) -> pd.DataFrame:
        # validate it
        data = smiles2fingerprints(
            values.to_list(), bond_radius=bond_radius, n_bits=n_bits
        )
        return pd.DataFrame(
            data=data,
            columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
        )

    def to_bag_of_characters(
        self, values: pd.Series, max_ngram: int = 5
    ) -> pd.DataFrame:
        # todo: add selfies later
        data = smiles2bag_of_characters(values.to_list(), max_ngram=max_ngram)
        return pd.DataFrame(
            data=data,
            columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
        )

    def to_fragments(self, values: pd.Series):
        data = smiles2fragments(values.to_list())
        return pd.DataFrame(
            data=data,
            columns=[f"{self.key}{_CAT_SEP}{i}" for i in range(data.shape[1])],
        )

    def sample(self, n: int) -> pd.Series:
        raise ValueError("Sampling not supported for `MolecularInput`.")

    def get_bounds(
        self, transform_type: TTransform, values: pd.Series
    ) -> Tuple[List[float], List[float]]:
        # TODO: this is only needed for optimization for which we need also
        # MolecularCategorical, this will be added later.
        raise NotImplementedError("`get_bounds` not yet implemented.")


class CategoricalMolecularInput(CategoricalInput, MolecularInput):
    type: Literal["CategoricalMolecularInput"] = "CategoricalMolecularInput"
    order: ClassVar[int] = 7

    @validator("categories")
    def validate_categories_valid_smiles(cls, v, values):
        for smi in v:
            smiles2mol(smi)
        return values

    def get_bounds(
        self,
        transform_type: MolecularEncodingEnum | CategoricalEncodingEnum,
        values: pd.Series | None = None,
    ) -> Tuple[List[float], List[float]]:
        if isinstance(transform_type, CategoricalEncodingEnum):
            return super().get_bounds(transform_type, values)
        if transform_type == MolecularEncodingEnum.FINGERPRINTS:
            # in case that values is None, we return the optimization bounds
            # else we return the complete bounds
            if values is None:
                df = self.to_fingerprints(self.categories).loc[  # type: ignore
                    self.get_allowed_categories()
                ]
            else:
                df = self.to_fingerprints(values=values)
            lower = df.min().values.tolist()  # type: ignore
            upper = df.max().values.tolist()  # type: ignore
            return lower, upper
        else:
            raise ValueError
