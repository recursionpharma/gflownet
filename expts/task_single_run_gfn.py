import sys

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_single_runs"

base_hps = {
    "num_training_steps": 10000,
    "validate_every": 1, # use 1000 might be faster
    "num_workers": 2,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            "num_mlp_layers": 2, 
            },
        },
    "opt": {"learning_rate": 1e-4}, # o.g. 1e-4
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 128, # 256
    "max_nodes": 7,
    "offline_ratio": 1, #0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_gfn/",
        "log_tags": ["test_dev"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": 1, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 1.0, #0.9,
            "reward_func": 'even_neighbors', 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "offline_sampling_g_distribution": "uniform",
            "use_true_log_Z": False,
            #"l2_reg_log_Z_lambda": 0.0005,
            **algo,
        },
        
    }
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", 
                "do_parameterize_p_b": False
                },
        },
    ]
]


from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[0])
trial.print_every = 1
trial.run()

#from gflownet.tasks.basic_graph_task import BGSupervisedTrainer

#trial = BGSupervisedTrainer(hps[0])
#@trial.print_every = 1
#trial.run()