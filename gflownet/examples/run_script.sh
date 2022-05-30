#!/bin/bash

# Annealing

n_mp_procs=128
n_train_steps=5000
layers=4
directory="results_thermo/"

# Dirac Beta
for r_dim in 3 4 5
do
    save_path="results_thermo/dirac_"$r_dim"d_horizon_"
    for hor in 16 32 64
    do
        python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --use_const_beta --const_beta 8 --n_layers $layers --r_dim $r_dim
done
done

# Const Dist Beta (2, 1)
# for r_dim in 3 4 5
# do
#     save_path=$directory"const_"$r_dim"d_horizon_"
#     for hor in 16 32 64
#     do
#         python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --n_layers $layers  --r_dim $r_dim
#     done
# done

# Annealing
# for r_dim in 3 4 5
# do
#     save_path=$directory/"annealing_"$r_dim"d_horizon_"
#     for hor in 16 32 64
#     do
#         python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --annealing --n_layers $layers --r_dim $r_dim
#     done
# done


# Uniform
# for r_dim in 3 4 5
# do
#     save_path=$directory/"uniform_"$r_dim"d_horizon_"
#     for hor in 16 32 64
#     do
#         python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --uniform --n_layers $layers --uniform --r_dim $r_dim
#     done
# done

# Const (1, 1)
# r_dim=3
# for r_dim in 3 4 5
# do
#     save_path=$directory/"const_1_1_"$r_dim"d_horizon_"
#     for hor in 16 32 64
#     do
#         python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor".pkl.gz" --n_train_steps $n_train_steps --n_layers $layers --r_dim $r_dim --alpha 1 --beta 1
#     done
# done