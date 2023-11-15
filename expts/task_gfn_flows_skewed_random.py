import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_TB_flows_skewed_random"
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
    "opt": {"learning_rate": 1e-4},
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 1024,
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_flows_skewed_random"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9,
            "reward_func": reward, 
            "reward_skewed_random": reward_skew,
            "reward_param": reward_param, # when reward_skewed_random = True, 0.0 = sparse rewards
            },
        },  
        
        "algo": {
            **base_algo_hps,
            **algo,
        },
        
    }
    for reward in ['const'] # in this experiment the reward is overwritten
    for reward_skew in [True]
    for reward_param in [0.0, 1.0]
    for seed in [1, 2, 3]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
        },
    ]
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()