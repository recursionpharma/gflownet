pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-geometric==2.0.3 torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.2+cu113.html
pip install rdkit-pypi tables sympy scipy
pip install -U networkx
pip install -e /run/determined/workdir/home/rs/gfn/gflownet/ #TODO: pull from github
# MXMNet requirements
pip install sympy git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
