#!/bin/bash

# Annealing

n_mp_procs=128
n_train_steps=5000
layers=4

# # Dirac Beta
# save_path="resultsv3/dirac_5d_horizon_"
# for hor in 16 32 64
# do
#     python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --use_const_beta --const_beta 8 --n_layers $layers
# done

# Const Dist Beta (2, 1)
# save_path="resultsv3/const_5d_horizon_"
# for hor in 16 32 64
# do
#     python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --n_layers $layers
# done

# Annealing
# save_path="resultsv3/annealing_5d_horizon_"
# for hor in 16 32 64
# do
#     python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --annealing --n_layers $layers
# done


# Uniform
# r_dim=5
# save_path="resultsv3/uniform_"$r_dim"d_horizon_"
# for hor in 16 32 64
# do
#     python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --annealing --n_layers $layers --uniform --r_dim $r_dim
# done

# Const (1, 1)
r_dim=3
save_path="resultsv3/const_1_1_"$r_dim"d_horizon_"
for hor in 16 32 64
do
    python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --n_layers $layers --r_dim $r_dim
done