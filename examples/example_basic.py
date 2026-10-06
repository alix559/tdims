"""Generate TDiMS descriptors for a few SMILES strings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tdims import tdims_ext


def main() -> None:
    smiles = [
        "COC(N)=O",
        "Oc1cocn1",
        "C#Cc1ncc[nH]1",
    ]
    print(f"Number of molecules: {len(smiles)}")
    matrix, feature_names = tdims_ext.get_representation(
        smiles,
        model="tdims",
        radius=1,
        func_dis=-2,
        func_merge=max,
        fragment_set=False,
        atom_set=True,
        fingerprint_set=True,
        display=True,
    )
    print(f"Nonzero entries: {int((matrix != 0).sum())}")
    if feature_names:
        print("\nFirst 5 feature names:")
        for name in feature_names[:5]:
            print(f"- {name}")


if __name__ == "__main__":
    main()
