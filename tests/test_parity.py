"""Compare the Mojo TDiMS kernel with frozen IBM/materials outputs."""

from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tdims import tdims_ext

GOLDEN = Path(__file__).with_name("golden.json")


def _kwargs(raw: dict):
    merged = dict(raw)
    name = merged.pop("func_merge")
    merged["func_merge"] = {"sum": sum, "max": max, "min": min}[name]
    return merged


def _canonical_name(name: str) -> str:
    """Pair labels are unordered. IBM's set iteration swaps equal-length sides."""
    left, separator, right = name.partition(" & ")
    if not separator or left == right or left < right:
        return name
    return f"{right} & {left}"


def _as_map(matrix: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    columns = {}
    for index, name in enumerate(names):
        columns[_canonical_name(name)] = np.asarray(matrix[:, index], dtype=float)
    return columns


class TDiMSParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with GOLDEN.open() as handle:
            cls.cases = json.load(handle)

    def test_golden_feature_values(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    matrix, names = tdims_ext.get_representation(
                        case["smiles"], **_kwargs(case["kwargs"])
                    )
                got = _as_map(matrix, names)
                expected = {
                    _canonical_name(name): values
                    for name, values in case["features"].items()
                }
                self.assertEqual(set(got), set(expected))
                for name, values in expected.items():
                    np.testing.assert_allclose(
                        got[name],
                        np.asarray(values, dtype=float),
                        rtol=1e-9,
                        atol=1e-12,
                    )

    def test_feature_order_is_stable(self):
        smiles = ["COC(N)=O", "ClC(Br)I", "c1ccccc1"]
        _, first = tdims_ext.get_representation(
            smiles,
            radius=1,
            func_dis=-2,
            func_merge=max,
            fragment_set=False,
        )
        _, second = tdims_ext.get_representation(
            smiles,
            radius=1,
            func_dis=-2,
            func_merge=max,
            fragment_set=False,
        )
        self.assertEqual(first, second)

    def test_failed_smiles_are_zero_rows(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matrix, names = tdims_ext.get_representation(
                ["CCO", "not_a_smiles", "O"],
                radius=1,
                func_dis=-2,
                func_merge=max,
                fragment_set=False,
            )
        self.assertTrue(caught)
        self.assertEqual(tdims_ext.get_representation.failed_smiles, ["not_a_smiles"])
        self.assertTrue(np.all(matrix[1] == 0))
        self.assertGreater(len(names), 0)

    def test_custom_distance_function(self):
        smiles = ["CCO", "CC(=O)O"]

        def inverse_square(distance: float) -> float:
            return distance**-2

        powered, names_powered = tdims_ext.get_representation(
            smiles,
            radius=1,
            func_dis=-2,
            func_merge=max,
            fragment_set=False,
            fingerprint_set=False,
        )
        custom, names_custom = tdims_ext.get_representation(
            smiles,
            radius=1,
            func_dis=inverse_square,
            func_merge=max,
            fragment_set=False,
            fingerprint_set=False,
        )
        self.assertEqual(names_powered, names_custom)
        np.testing.assert_allclose(powered, custom)

    def test_feature_selection_runs(self):
        smiles = ["CCO", "CC(=O)O", "c1ccccc1", "OCCO", "CCN"]
        target = [1.0, 2.0, 0.5, 1.5, 0.2]
        selected, keys, all_keys = tdims_ext.get_representation_with_fs_selection(
            smiles,
            target,
            radius=1,
            func_dis=-2,
            func_merge=max,
            fragment_set=False,
        )
        self.assertEqual(selected.shape[0], len(smiles))
        self.assertEqual(selected.shape[1], len(keys))
        self.assertLessEqual(len(keys), len(all_keys))
        self.assertGreater(len(keys), 0)

    def test_nested_cv_smoke(self):
        smiles = ["CCO", "CC(=O)O", "c1ccccc1", "OCCO", "CCN", "CCC"]
        target = [1.0, 2.0, 0.5, 1.5, 0.2, 0.8]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = tdims_ext.run_tdims_regression_cv(
                smiles,
                target,
                radius=1,
                func_dis=-2,
                func_merge=max,
                fragment_set=False,
                cv=2,
                n_repeats=1,
                inner_cv=2,
                mode="quick",
                model_name="ridge",
                use_feature_selection=False,
                fit_final_estimator=False,
                n_jobs=1,
            )
        self.assertEqual(len(result["r2_scores"]), 2)
        self.assertEqual(result["X_shape"][0], len(smiles))


if __name__ == "__main__":
    unittest.main()
