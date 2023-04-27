from collections import defaultdict
import gzip

import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit import RDLogger
from tqdm import tqdm
import torch
import torch.nn as nn

RDLogger.DisableLog('rdApp.*')

# Downloaded from: https://zenodo.org/record/2543724#.Y_d-XezMJqs
f = open('/data/chem/excape/pubchem.chembl.dataset4publication_inchi_smiles.tsv', 'r')
f.readline()
pts = defaultdict(lambda: [0,0])

# Should take a minute or two.
for i in tqdm(f, total=70850163, desc='Loading ExCAPE-DB'): 
    i = i.split('\t')
    if i[4]:  # Filter for active compounds only
        gene, pXC50, smi = i[8], float(i[4]), i[11]
        if gene == 'MAPK10':  # MAPK10 is the name for JNK3 in chembl (unless I'm terribly wrong)
            pts[smi][0] = pXC50
        elif gene == 'GSK3B':
            pts[smi][1] = pXC50

# Some SMILES in Chembl are more than rdkit can handle, filter those out
smis = sorted([i for i in pts.keys() if Chem.MolFromSmiles(i) is not None])
# Get ECFP6 (radius=3)
fps = [Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i), 3)
       for i in smis]

Xs = np.float32(fps)
Ys = np.float32([pts[i] for i in smis])

np.random.seed(142857)
torch.random.manual_seed(142857)
dev = torch.device('cuda')  # Set appropriately

# After some playing around with a validation set, this seems to work OK
# Gets slightly better (~2.13) validation error than a random forest (~2.25), 15/85 split.
nemb = 256
jnk3_gsk3b_mlp = nn.Sequential(nn.Linear(2048, nemb), nn.ReLU(),
                      nn.Linear(nemb, nemb), nn.ReLU(),
                      nn.Linear(nemb, nemb), nn.ReLU(),
                      nn.Linear(nemb, nemb), nn.ReLU(),
                      nn.Linear(nemb, 2))
jnk3_gsk3b_mlp.to(dev)
Xs = torch.tensor(Xs, device=dev)
Ys = torch.tensor(Ys, device=dev)
opt = torch.optim.Adam(jnk3_gsk3b_mlp.parameters(), 5e-4, weight_decay=1e-4, betas=(0.95,0.999))
# 100k steps is when the validation loss seems to plateau
# This takes ~3 minutes on an A100
p = tqdm(range(100000), desc='Training Model')
mbs = 512
for i in p:
    # Yes, we're doing full batch training. Works fine for ECFP6 features.
    yhat = jnk3_gsk3b_mlp(Xs)
    loss = (Ys-yhat).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    # Loss should be ~0.091 for most of training, improvements seem to happen thanks to 
    # the magic of overparameterization/double descent/all that stuff.
    p.set_description(f'{loss.item():.3f}', refresh=False)
p.close()

jnk3_gsk3b_mlp.to('cpu')
torch.save(jnk3_gsk3b_mlp.state_dict(), gzip.open('jnk3_gsk3b_mlp.pt.gz', 'wb'))