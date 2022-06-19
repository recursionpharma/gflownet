#!/bin/bash

# Annealing

n_mp_procs=128
n_train_steps=5000
layers=4
directory="gridworld_results/results_final/"
method=$1
echo $method
if [ $method = "dirac" ];
then
echo "Runing dirac now"
# Dirac Beta

for r_dim in 3 4 5
do
    
    for hor in 16 32 64
    do
    for beta in 2 4 8 16
    do
    save_path=$directory"/dirac/beta_"$beta"/rewards_"$r_dim"d/horizon_"
        python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor"/" --n_train_steps $n_train_steps --use_const_beta --const_beta $beta --n_layers $layers --r_dim $r_dim --use_thermo_encoding
done
done
done

# Const Dist Beta (2, 1)
elif [ $method = "const" ];
then
echo "Runing const now"
for r_dim in 3 4 5
do
    save_path=$directory"const_2_1/rewards_"$r_dim"d/horizon_"
    for hor in 16 32 64
    do
        python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor"/" --n_train_steps $n_train_steps --n_layers $layers  --r_dim $r_dim --use_thermo_encoding
    done
done

elif [ $method = "annealing" ];
then
# Annealing
echo "Runing annealing now"
for r_dim in 3 4 5
do
    save_path=$directory"annealing/rewards_"$r_dim"d/horizon_"
    for hor in 16 32 64
    do
        python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor"/" --n_train_steps $n_train_steps --annealing --n_layers $layers --r_dim $r_dim --use_thermo_encoding
    done
done


elif [ $method = "uniform" ];
then
# Uniform
for r_dim in 3 4 5
do
    save_path=$directory/"uniform/rewards_"$r_dim"d/horizon_"
    for hor in 16 32 64
    do
        python grid_cond_gfn.py --n_mp_procs $n_mp_procs --horizon $hor --progress --save_path $save_path$hor"/" --n_train_steps $n_train_steps --uniform --n_layers $layers --uniform --r_dim $r_dim --use_thermo_encoding
    done
done

fi

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