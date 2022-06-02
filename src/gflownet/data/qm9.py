import tarfile

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from torch.utils.data import Dataset
from rdkit.Chem import Descriptors

class QM9Dataset(Dataset):
    def __init__(self, h5_file=None, xyz_file=None, train=True, targets='gap', split_seed=142857, ratio=0.9):
        if h5_file is not None:
            self.df = pd.HDFStore(h5_file, 'r')['df']
        elif xyz_file is not None:
            self.load_tar()
        rng = np.random.default_rng(split_seed)
        idcs = np.arange(len(self.df))  # TODO: error if there is no h5_file provided. Should h5 be required
        rng.shuffle(idcs)
        self.targets = targets
        self.calculate_logP()
        self.calculate_QED()
        self.calculate_molwt()
        if train:
            self.idcs = idcs[:int(np.floor(ratio * len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(self.df))):]

    def gather_rewards(self, idx):
        rewards = []
        for target in self.targets:
            rewards.append(self.df[target][idx])
        return np.asarray(rewards)

    def get_stats(self, target, percentile=0.95):
        # Return a list of stats
        y = self.df[target]
        return y.min(), y.max(), np.sort(y)[int(y.shape[0] * percentile)]

    def calculate_logP(self):
        logP = []
        for idx in self.idcs:
            molecule = self.df['SMILES'][idx]
            logP.append(Descriptors.MolLogP(molecule))
        self.df["logP"] = logP

    def calculate_QED(self):
        QED = []
        for idx in self.idcs:
            molecule = self.df['SMILES'][idx]
            QED.append(Descriptors.qed(molecule))
        self.df["QED"] = QED

    def calculate_molwt(self):
        mol_weight = []
        for idx in self.idcs:
            molecule = self.df['SMILES'][idx]
            mol_weight.append(Descriptors.MolWt(molecule))
        self.df["molecular_weight"] = mol_weight

    def load_tar(self, xyz_file):
        f = tarfile.TarFile(xyz_file, 'r')
        labels = ['rA', 'rB', 'rC', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        all_mols = []
        for pt in f:
            pt = f.extractfile(pt)
            data = pt.read().decode().splitlines()
            all_mols.append(data[-2].split()[:1] + list(map(float, data[1].split()[2:])))
        self.df = pd.DataFrame(all_mols, columns=['SMILES'] + labels)

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        rewards = self.gather_rewards(idx)
        return (Chem.MolFromSmiles(self.df['SMILES'][self.idcs[idx]]), rewards)
