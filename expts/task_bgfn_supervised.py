import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/bgfn_supervised_enc_cond_logZ_subTB_Dec_8"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000, # use 1000 might be faster
    "num_workers": 8,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            "num_mlp_layers": 2, 
            },
        },
    #"opt": {"learning_rate": 1e-4}, # o.g. 1e-4
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 256,
    "max_nodes": 7,
    "offline_ratio": 1 / 2,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["bgfn_supervised_enc_cond_logZ"],
        "seed": seed,
        "opt": {"learning_rate": 1e-4}, # o.g. 1e-4,
        "run_valid_dl": True,
        
        "task": {
        "basic_graph": {
            "test_split_seed": 1, #142857, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": data_ratio, 
            "reward_func": reward, 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "offline_sampling_g_distribution": data_distribution,
            "supervised_reward_predictor": "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/supervised_gnn_models_for_bgfn_Dec_7/",
            "flow_reg": True,
            "valid_sample_cond_info": False,
            "valid_offline_ratio": 1,
            **algo,
        },

        "cond": {
            "logZ":{
                "sample_dist": "uniform",
                "dist_params": dist_params, # [1.0, 20.0]
                "num_thermometer_dim": 32,
                "num_valid_logZ_samples": 10,
            },
        },
        
    }
    for reward in ['const', 'count', 'even_neighbors', 'cliques']
    for data_distribution in ["uniform"] #["uniform", "log_rewards", "log_p", "l2_log_error_gfn", "l1_error_gfn"] # TODO implement "loss_gfn"
    for seed in [1]
    for data_ratio in [0.9]
    for dist_params in [[1.0, 20.0], [3.0, 15.0], [5.0, 12.0]]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", 
                "do_parameterize_p_b": False,
                },
        },
    ]
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()