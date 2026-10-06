from std.python import Python
from tdims_encode import encode

def main() raises:
    var smiles = Python.list("COC(N)=O", "Oc1cocn1", "C#Cc1ncc[nH]1")
    var options = Python.evaluate(
        '{"radius": 1, "use_power": True, "func_dis_power": -2.0,'
        ' "func_dis_fn": None, "merge": "max", "merge_fn": None,'
        ' "fragment_mode": "none", "fragments": [],'
        ' "atom_set": True, "fingerprint_set": True}'
    )
    var encoded = encode(smiles, options)
    var matrix = encoded[0]
    var names = encoded[1]
    print("shape", matrix.shape)
    print("n_features", len(names))
    var shown = 0
    for name in names:
        if shown == 5:
            break
        print(name)
        shown += 1
