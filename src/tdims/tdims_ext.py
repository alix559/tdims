"""Public TDiMS API.

``get_representation`` matches the IBM/materials entry point. Descriptor
construction runs in the Mojo module ``tdims_encode``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

_MOJO_DIR = Path(__file__).resolve().parents[1] / "mojo"
_ENCODER = None

FragmentArg = Union[bool, Sequence[str]]
MergeArg = Union[Callable[..., Any], str]


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        if str(_MOJO_DIR) not in sys.path:
            sys.path.insert(0, str(_MOJO_DIR))
        import mojo.importer  # noqa: F401

        import tdims_encode

        _ENCODER = tdims_encode
    return _ENCODER


def _fragment_options(fragment_set: FragmentArg) -> tuple[str, list[str]]:
    if fragment_set is True:
        return "cep", []
    if fragment_set is False:
        return "none", []
    if isinstance(fragment_set, (list, tuple, set)):
        return "custom", [str(item) for item in fragment_set]
    raise TypeError(
        "fragment_set must be True, False, or a sequence of SMILES strings"
    )


def _merge_options(func_merge: MergeArg) -> tuple[str, Optional[Callable[..., Any]]]:
    if isinstance(func_merge, str):
        name = func_merge
        fn = None
    else:
        name = getattr(func_merge, "__name__", "")
        fn = func_merge
    if name == "amax":
        name = "max"
    if name in {"sum", "max", "min"} and (
        fn is None or name == getattr(fn, "__name__", name)
    ):
        return name, None
    if fn is None:
        raise ValueError("func_merge must be sum, max, min, or a callable")
    return "custom", fn


def _distance_options(func_dis: Any) -> tuple[bool, float, Optional[Callable[..., Any]]]:
    if callable(func_dis):
        return False, 0.0, func_dis
    return True, float(func_dis), None


def get_representation(
    sm_list,
    model: str = "tdims",
    radius: int = 1,
    func_dis=-2,
    func_merge=sum,
    fragment_set: FragmentArg = True,
    atom_set: bool = True,
    fingerprint_set: bool = True,
    display: bool = False,
):
    """Encode SMILES as a TDiMS descriptor matrix.

    Returns the dense matrix and the feature-name list. Failed SMILES become
    all-zero rows and are recorded on ``get_representation.failed_smiles``.
    """
    if model != "tdims":
        raise ValueError("Invalid model input")

    fragment_mode, fragments = _fragment_options(fragment_set)
    merge_name, merge_fn = _merge_options(func_merge)
    use_power, power, dis_fn = _distance_options(func_dis)
    options = {
        "radius": int(radius),
        "use_power": use_power,
        "func_dis_power": power,
        "func_dis_fn": dis_fn,
        "merge": merge_name,
        "merge_fn": merge_fn,
        "fragment_mode": fragment_mode,
        "fragments": fragments,
        "atom_set": bool(atom_set),
        "fingerprint_set": bool(fingerprint_set),
    }

    start = time.time()
    matrix, names, failed = _encoder().encode(list(sm_list), options)
    elapsed = time.time() - start

    matrix = np.asarray(matrix, dtype=float)
    names = [str(name) for name in names]
    failed_smiles = [str(item) for item in failed]
    get_representation.failed_smiles = failed_smiles

    if failed_smiles:
        warnings.warn(
            f"{len(failed_smiles)} of {len(list(sm_list))} SMILES could not be "
            "parsed by RDKit and produced all-zero feature rows. "
            "See get_representation.failed_smiles.",
            stacklevel=2,
        )
    if display:
        print(f"Full embedding shape: {matrix.shape}")
        print(f"Execution time for full embedding: {elapsed: .6f} sec")
    return matrix, names


get_representation.failed_smiles = []


def get_representation_with_fs_selection(
    sm_list,
    y,
    model: str = "tdims",
    radius: int = 1,
    func_dis=-2,
    func_merge=sum,
    fragment_set: FragmentArg = True,
    atom_set: bool = True,
    fingerprint_set: bool = True,
    display: bool = False,
):
    """TDiMS descriptors followed by Lasso ``SelectFromModel``."""
    if model != "tdims":
        raise ValueError("Invalid model input")

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso

    x_all, key_all = get_representation(
        sm_list,
        model,
        radius=radius,
        func_dis=func_dis,
        func_merge=func_merge,
        fragment_set=fragment_set,
        atom_set=atom_set,
        fingerprint_set=fingerprint_set,
    )
    start = time.time()
    selector = SelectFromModel(
        estimator=Lasso(random_state=0, alpha=0.0001, max_iter=100000, tol=1e-4),
        threshold="mean",
    ).fit(x_all, y)
    x_slc = selector.transform(x_all)
    key_slc = np.array(key_all)[selector.get_support()]
    if display:
        print(f"\nFeature were selected from {x_all.shape} to {x_slc.shape}")
        print(
            "Execution time for feature selection:"
            f" {time.time() - start: .6f} sec"
        )
    return x_slc, list(key_slc), key_all


def run_tdims_regression_cv(*args, **kwargs):
    """Nested cross-validation. Implemented in ``tdims.regression``."""
    from tdims.regression import run_tdims_regression_cv as _run

    return _run(*args, **kwargs)
