import tarfile

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from rdkit.Chem import QED, Descriptors
from torch.utils.data import Dataset

from gflownet.utils import sascore


class QM9Dataset(Dataset):
    def __init__(self, h5_file=None, xyz_file=None, train=True, targets=["gap"], split_seed=142857, ratio=0.9):
        if h5_file is not None:
            self.df = pd.HDFStore(h5_file, "r")["df"]
        elif xyz_file is not None:
            self.load_tar()
        rng = np.random.default_rng(split_seed)
        idcs = np.arange(len(self.df))  # TODO: error if there is no h5_file provided. Should h5 be required
        rng.shuffle(idcs)
        self.targets = targets
        if train:
            self.idcs = idcs[: int(np.floor(ratio * len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(self.df))) :]
        self.mol_to_graph = lambda x: x

    def setup(self, task, ctx):
        self.mol_to_graph = ctx.mol_to_graph

    def get_stats(self, target=None, percentile=0.95):
        if target is None:
            target = self.targets[0]
        y = self.df[target]
        return y.min(), y.max(), np.sort(y)[int(y.shape[0] * percentile)]

    def load_tar(self, xyz_file):
        self.df = load_tar(xyz_file)

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return (
            self.mol_to_graph(Chem.MolFromSmiles(self.df["SMILES"][self.idcs[idx]])),
            torch.tensor([self.df[t][self.idcs[idx]] for t in self.targets]).float(),
        )


def load_tar(xyz_file):
    labels = ["rA", "rB", "rC", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]
    f = tarfile.TarFile(xyz_file, "r")
    all_mols = []
    for pt in f:
        pt = f.extractfile(pt)  # type: ignore3
        data = pt.read().decode().splitlines()  # type: ignore
        all_mols.append(data[-2].split()[:1] + list(map(float, data[1].split()[2:])))
    df = pd.DataFrame(all_mols, columns=["SMILES"] + labels)
    mols = df["SMILES"].map(Chem.MolFromSmiles)
    df["qed"] = mols.map(QED.qed)
    df["sa"] = mols.map(sascore.calculateScore)
    df["mw"] = mols.map(Descriptors.MolWt)
    return df


def convert_h5():
    # File obtained from
    # https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    # (from http://quantum-machine.org/datasets/)
    df = load_tar("qm9.xyz.tar")
    with pd.HDFStore("qm9.h5", "w") as store:
        store["df"] = df
