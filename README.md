# TDiMS

Mojo implementation of [TDiMS](https://github.com/IBM/materials/tree/main/models/tdims) (Topological Distance of intraMolecular Substructures). The descriptor summarizes pairwise topological distances between intramolecular substructures.

RDKit parses SMILES, matches fragments, and canonicalizes subgraph labels. The distance matrix, distance transform (`func_dis`), merge (`sum` / `max` / `min`), and feature matrix are computed in Mojo (`src/mojo/tdims_encode.mojo`).

Heteroatom columns follow first-seen atom order. The original Python code iterates a `set`, so its column order changes with `PYTHONHASHSEED`. Feature names and values match that implementation.

## Setup

```bash
pixi run example
pixi run test
pixi run example-mojo
```

## Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, "src")

from tdims import tdims_ext

smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
matrix, names = tdims_ext.get_representation(
    smiles,
    radius=1,
    func_dis=-2,
    func_merge=max,
    fragment_set=False,
    atom_set=True,
    fingerprint_set=True,
    display=True,
)
```

`fragment_set=True` uses the CEP fragment list from the IBM code. A list of SMILES uses those fragments instead, with a `_frag` suffix on the feature name.

`tdims_ext.run_tdims_regression_cv(smiles, y)` runs the nested cross-validation from the original study: descriptors are fit on SMILES alone, and feature selection plus the hyperparameter search stay inside each outer-training split. `mode="quick"` is the reduced grid; `mode="full"` is the paper grid.

## Citation

This is a reimplementation of the descriptor in [IBM/materials `models/tdims`](https://github.com/IBM/materials/tree/main/models/tdims), distributed under Apache-2.0. Cite the TDiMS paper if you use it in academic work.

## License

Apache License 2.0. The CEP fragment list and the descriptor definition come from IBM/materials.
