# TDiMS descriptor kernel.
#
# Topological Distance of intraMolecular Substructures (IBM/materials).
# Substructure labels come from RDKit; pairwise distances, the distance
# transform, merging, and the feature matrix are computed here.
#
# Apache-2.0, derived from IBM/materials models/tdims.

from std.math import pow
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

# RDKit's topological distance matrix uses this sentinel between components.
comptime DISCONNECTED = 100000000

def cep_fragments() -> List[String]:
    var frags = List[String]()
    frags.append("C1C=CC=C1")
    frags.append("S1N=C2C=CN=CC2=N1")
    frags.append("S1N=C2C=CC=CC2=N1")
    frags.append("O1C=CC2=CSC=C12")
    frags.append("C1=CC=NC=C1")
    frags.append("C1=NC=NC=N1")
    frags.append("O1C=CC=C1")
    frags.append("[SiH2]1C=CC=C1")
    frags.append("[SiH2]1C=C2C=CC=CC2=C1")
    frags.append("N1C=CC2=CSC=C12")
    frags.append("C1=CC2=CC=CC=C2C=C1")
    frags.append("S1C=CC=C1")
    frags.append("S1C=C2SC=CC2=C1")
    frags.append("O1C=C2C=CC=CC2=C1")
    frags.append("[Se]1C=CC=C1")
    frags.append("N1C=C2C=CC=CC2=C1")
    frags.append("N1C=CC=C1")
    frags.append("S1C=C2C(=C1)C1=CC=CC=C1C1=CC=CC=C21")
    frags.append("C1=CC=CC=C1")
    frags.append("S1C=C2N=CC=NC2=C1")
    frags.append("S1C=CN=C1")
    frags.append("[SiH2]1C=CC2=CSC=C12")
    frags.append("S1C=C2C=CC=CC2=C1")
    frags.append("S1C=C2[Se]C=CC2=C1")
    frags.append("C1C=C2C=CC=CC2=C1")
    frags.append("C1C=CC2=CSC=C12")
    return frags^


def mol_is_valid(mol: PythonObject) raises -> Bool:
    try:
        var _n = Int(py=mol.GetNumAtoms())
        return _n >= 0
    except:
        return False


def empty_ints() -> List[Int]:
    return []


def topological_distances(n: Int, adj: List[List[Int]]) -> List[Int]:
    """All-pairs shortest paths on the heavy-atom bond graph."""
    var dist = List[Int]()
    for _ in range(n * n):
        dist.append(DISCONNECTED)
    for src in range(n):
        var queue = List[Int]()
        var head = 0
        dist[src * n + src] = 0
        queue.append(src)
        while head < len(queue):
            var node = queue[head]
            head += 1
            var base = dist[src * n + node]
            var degree = len(adj[node])
            for k in range(degree):
                var neighbor = adj[node][k]
                var slot = src * n + neighbor
                if dist[slot] == DISCONNECTED:
                    dist[slot] = base + 1
                    queue.append(neighbor)
    return dist^


def is_atom_subset(small: List[Int], big: List[Int]) -> Bool:
    """True when every index in `small` also appears in `big`."""
    var hit = 0
    for atom in small:
        var found = False
        for other in big:
            if atom == other:
                found = True
                break
        if found:
            hit += 1
    return hit == len(small)


def covered(atoms: List[Int], occupied: List[List[Int]]) -> Bool:
    for i in range(len(occupied)):
        if is_atom_subset(atoms, occupied[i]):
            return True
    return False


def order_by_length_desc(names: List[String]) -> List[Int]:
    """Stable sort: longer labels first, equal lengths keep input order."""
    var order = List[Int]()
    for i in range(len(names)):
        order.append(i)
    for i in range(1, len(order)):
        var current = order[i]
        var j = i
        while j > 0:
            var prev = order[j - 1]
            if names[prev].byte_length() < names[current].byte_length():
                order[j] = prev
                j -= 1
            else:
                break
        order[j] = current
    return order^


def contains_name(names: List[String], name: String) -> Bool:
    for existing in names:
        if existing == name:
            return True
    return False


def induced_smiles(
    Chem: PythonObject, mol: PythonObject, ids: List[Int]
) raises -> String:
    """Canonical SMILES of the induced subgraph.

    Matches AtomGraph.translate_to_mol(..., sanitize=False) followed by
    MolToSmiles: atom symbol, formal charge, explicit H only when uncharged,
    and the original bond types. Aromaticity is carried by aromatic bonds
    that survive inside the neighborhood, not by the source atom flags.
    """
    var rw = Chem.RWMol()
    var mapping = Dict[Int, Int]()
    var chosen = Dict[Int, Bool]()
    for idx in ids:
        var atom = mol.GetAtomWithIdx(idx)
        var new_atom = Chem.Atom(atom.GetSymbol())
        var charge = Int(py=atom.GetFormalCharge())
        new_atom.SetFormalCharge(charge)
        if charge == 0:
            new_atom.SetNumExplicitHs(Int(py=atom.GetNumExplicitHs()))
        mapping[idx] = Int(py=rw.AddAtom(new_atom))
        chosen[idx] = True
    var bonds = mol.GetBonds()
    for bond in bonds:
        var begin = Int(py=bond.GetBeginAtomIdx())
        var end = Int(py=bond.GetEndAtomIdx())
        if begin in chosen and end in chosen:
            _ = rw.AddBond(mapping[begin], mapping[end], bond.GetBondType())
    return String(py=Chem.MolToSmiles(rw.GetMol()))


def add_matches(
    mut names: List[String],
    mut instances: List[List[List[Int]]],
    mol: PythonObject,
    query: PythonObject,
    key: String,
) raises:
    var matches = mol.GetSubstructMatches(query)
    if len(matches) == 0:
        return
    var bucket = List[List[Int]]()
    for group in matches:
        var atoms = List[Int]()
        for atom_idx in group:
            atoms.append(Int(py=atom_idx))
        bucket.append(atoms^)
    names.append(key.copy())
    instances.append(bucket^)


def add_fragment(
    mut names: List[String],
    mut instances: List[List[List[Int]]],
    Chem: PythonObject,
    mol: PythonObject,
    frag_smiles: String,
    suffix: String,
) raises:
    var query = Chem.MolFromSmiles(frag_smiles)
    if not mol_is_valid(query):
        print(String(t"Invalid fragment SMILES was skipped: {frag_smiles}"))
        return
    var key = String(t"{frag_smiles}_{suffix}")
    try:
        add_matches(names, instances, mol, query, key)
    except:
        print(String(t"Invalid fragment SMILES was skipped: {frag_smiles}"))


def add_heteroatoms(
    mut names: List[String],
    mut instances: List[List[List[Int]]],
    Chem: PythonObject,
    mol: PythonObject,
    n_atoms: Int,
) raises:
    # First-seen heavy-atom symbols, skipping carbon. The reference walks a
    # Python set, whose order follows PYTHONHASHSEED; first-seen order is
    # the same rule with a stable result.
    var seen = Dict[String, Bool]()
    var symbols = List[String]()
    for i in range(n_atoms):
        var symbol = String(py=mol.GetAtomWithIdx(i).GetSymbol())
        if symbol == "C":
            continue
        if symbol in seen:
            continue
        seen[symbol.copy()] = True
        symbols.append(symbol.copy())
    for symbol in symbols:
        var query = Chem.MolFromSmiles(symbol)
        if not mol_is_valid(query):
            continue
        try:
            add_matches(names, instances, mol, query, symbol.copy())
        except:
            pass


def add_fingerprint_envs(
    mut names: List[String],
    mut instances: List[List[List[Int]]],
    Chem: PythonObject,
    mol: PythonObject,
    dist: List[Int],
    n_atoms: Int,
    radius: Int,
) raises:
    var fp_names = List[String]()
    var fp_instances = List[List[List[Int]]]()
    var fp_index = Dict[String, Int]()
    for center in range(n_atoms):
        var ids = List[Int]()
        for other in range(n_atoms):
            if dist[center * n_atoms + other] <= radius:
                ids.append(other)
        var label = induced_smiles(Chem, mol, ids)
        if label in fp_index:
            fp_instances[fp_index[label]].append(ids^)
        else:
            fp_index[label.copy()] = len(fp_names)
            fp_names.append(label.copy())
            var bucket = List[List[Int]]()
            bucket.append(ids^)
            fp_instances.append(bucket^)

    var visit = order_by_length_desc(fp_names)
    var occupied = List[List[Int]]()
    for i in range(len(instances)):
        for j in range(len(instances[i])):
            occupied.append(instances[i][j].copy())

    var kept_at = Dict[String, Int]()
    for rank in range(len(visit)):
        var fp_i = visit[rank]
        for j in range(len(fp_instances[fp_i])):
            var atoms = fp_instances[fp_i][j].copy()
            if covered(atoms, occupied):
                continue
            var label = fp_names[fp_i]
            if label in kept_at:
                instances[kept_at[label]].append(atoms.copy())
            else:
                var bucket = List[List[Int]]()
                bucket.append(atoms.copy())
                kept_at[label.copy()] = len(names)
                names.append(label.copy())
                instances.append(bucket^)
            occupied.append(atoms.copy())


def mean_distance(
    left: List[Int], right: List[Int], dist: List[Int], n_atoms: Int
) raises -> Float64:
    var total = 0.0
    var count = 0
    for a in left:
        for b in right:
            total += Float64(dist[a * n_atoms + b])
            count += 1
    if count == 0:
        raise Error("substructure instance has no atoms")
    return total / Float64(count)


def transform_distance(
    average: Float64,
    use_power: Bool,
    power: Float64,
    distance_fn: PythonObject,
) raises -> Float64:
    if use_power:
        if average == 0.0 and power < 0.0:
            raise Error(
                "mean topological distance is zero and cannot take a negative power"
            )
        return pow(average, power)
    return Float64(py=distance_fn(average))


def merge_distances(
    values: List[Float64], mode: String, merge_fn: PythonObject
) raises -> Float64:
    if len(values) == 0:
        return 0.0
    if mode == "max":
        var best = values[0]
        for i in range(1, len(values)):
            if values[i] > best:
                best = values[i]
        return best
    if mode == "min":
        var best = values[0]
        for i in range(1, len(values)):
            if values[i] < best:
                best = values[i]
        return best
    if mode == "sum":
        var total = 0.0
        for value in values:
            total += value
        return total
    var py_values = Python.list()
    for value in values:
        py_values.append(value)
    return Float64(py=merge_fn(py_values))


def pair_feature(
    left: List[List[Int]],
    right: List[List[Int]],
    same: Bool,
    dist: List[Int],
    n_atoms: Int,
    use_power: Bool,
    power: Float64,
    dis_fn: PythonObject,
    mode: String,
    merge_fn: PythonObject,
) raises -> Float64:
    var values = List[Float64]()
    if same:
        for i in range(len(left)):
            for j in range(i + 1, len(right)):
                var average = mean_distance(left[i], right[j], dist, n_atoms)
                values.append(
                    transform_distance(average, use_power, power, dis_fn)
                )
    else:
        for i in range(len(left)):
            for j in range(len(right)):
                var average = mean_distance(left[i], right[j], dist, n_atoms)
                values.append(
                    transform_distance(average, use_power, power, dis_fn)
                )
    return merge_distances(values, mode, merge_fn)


def bond_graph(mol: PythonObject) raises -> List[List[Int]]:
    var n_atoms = Int(py=mol.GetNumAtoms())
    var adj = List[List[Int]]()
    for _ in range(n_atoms):
        adj.append(empty_ints())
    var bonds = mol.GetBonds()
    for bond in bonds:
        var begin = Int(py=bond.GetBeginAtomIdx())
        var end = Int(py=bond.GetEndAtomIdx())
        adj[begin].append(end)
        adj[end].append(begin)
    return adj^


def molecule_features(
    mut feature_names: List[String],
    mut feature_values: List[Float64],
    Chem: PythonObject,
    mol: PythonObject,
    radius: Int,
    use_power: Bool,
    power: Float64,
    dis_fn: PythonObject,
    mode: String,
    merge_fn: PythonObject,
    fragment_mode: String,
    fragments: PythonObject,
    atom_set: Bool,
    fingerprint_set: Bool,
    smiles: String,
) raises:
    var n_atoms = Int(py=mol.GetNumAtoms())
    var adj = bond_graph(mol)
    var dist = topological_distances(n_atoms, adj)

    var names = List[String]()
    var instances = List[List[List[Int]]]()

    if fragment_mode == "cep":
        var frags = cep_fragments()
        for frag in frags:
            add_fragment(names, instances, Chem, mol, frag, "CEPfrag")
    elif fragment_mode == "custom":
        for item in fragments:
            add_fragment(
                names, instances, Chem, mol, String(py=item), "frag"
            )

    if atom_set:
        add_heteroatoms(names, instances, Chem, mol, n_atoms)

    if fingerprint_set:
        add_fingerprint_envs(
            names, instances, Chem, mol, dist, n_atoms, radius
        )

    var order = order_by_length_desc(names)
    var eps = 1.0e-10
    for a in range(len(order)):
        for b in range(a, len(order)):
            var left_i = order[a]
            var right_i = order[b]
            var label = String(t"{names[left_i]} & {names[right_i]}")
            var value = pair_feature(
                instances[left_i],
                instances[right_i],
                left_i == right_i,
                dist,
                n_atoms,
                use_power,
                power,
                dis_fn,
                mode,
                merge_fn,
            )
            if value < 0.0:
                print(String(t"Minus feature for mol:{smiles} feature:{label}"))
            elif value - eps > 0.0:
                feature_names.append(label^)
                feature_values.append(value)


def encode(smiles: PythonObject, options: PythonObject) raises -> PythonObject:
    """Build the TDiMS matrix for a sequence of SMILES.

    Returns ``(matrix, feature_names, failed_smiles)``.
    """
    var Chem = Python.import_module("rdkit.Chem")
    var RDLogger = Python.import_module("rdkit.RDLogger")
    RDLogger.DisableLog("rdApp.*")
    var np = Python.import_module("numpy")

    var radius = Int(py=options["radius"])
    var use_power = Bool(py=options["use_power"])
    var power = Float64(py=options["func_dis_power"])
    var dis_fn = options["func_dis_fn"]
    var mode = String(py=options["merge"])
    var merge_fn = options["merge_fn"]
    var fragment_mode = String(py=options["fragment_mode"])
    var fragments = options["fragments"]
    var atom_set = Bool(py=options["atom_set"])
    var fingerprint_set = Bool(py=options["fingerprint_set"])

    var n_mol = len(smiles)
    var all_names = List[String]()
    var lookups = List[Dict[String, Float64]]()
    var failed = Python.list()

    for row in range(n_mol):
        var sm = String(py=smiles[row])
        var mol = Chem.MolFromSmiles(sm)
        var lookup = Dict[String, Float64]()
        if not mol_is_valid(mol):
            failed.append(sm)
            lookups.append(lookup^)
            continue
        var feature_names = List[String]()
        var feature_values = List[Float64]()
        molecule_features(
            feature_names,
            feature_values,
            Chem,
            mol,
            radius,
            use_power,
            power,
            dis_fn,
            mode,
            merge_fn,
            fragment_mode,
            fragments,
            atom_set,
            fingerprint_set,
            sm,
        )
        for i in range(len(feature_names)):
            var name = feature_names[i]
            lookup[name.copy()] = feature_values[i]
            if not contains_name(all_names, name):
                all_names.append(name.copy())
        lookups.append(lookup^)

    var n_feat = len(all_names)
    var matrix = np.zeros(Python.tuple(n_mol, n_feat), dtype=np.float64)
    for row in range(n_mol):
        for col in range(n_feat):
            var name = all_names[col].copy()
            if name in lookups[row]:
                matrix[row][col] = lookups[row][name]

    var py_names = Python.list()
    for name in all_names:
        py_names.append(name)
    return Python.tuple(matrix, py_names, failed)


@export
def PyInit_tdims_encode() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("tdims_encode")
        builder.def_function[encode]("encode")
        return builder.finalize()
    except e:
        abort(String("failed to create module: ", e))
