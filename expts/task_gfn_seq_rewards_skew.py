import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_seq_rewards_skew_Jan_15"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000,
    "num_workers": 8,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            },
        },
    "opt": {"learning_rate": 1e-4},
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 256, # 256
    "max_nodes": 15,
    "max_len": 16, # needs to be max_len = max_nodes + 1
    "offline_ratio": 0, # 1
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_seq_rewards_skew_v2"],
        "seed": seed,
        
        "task": {
        "toy_seq": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9,
            "reward_func": 'edit', 
            "reward_reshape": True,
            "reward_corrupt": False,
            "reward_shuffle": False,
            "reward_param": lam,
            },
        },  
        
        "algo": {
            **base_algo_hps,
            **algo,
        },
        
    }
    for lam in [0.0, 0.5, 1.0, 1.5]
    for seed in [1, 2, 3]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", #"SubTBMC", 
                "cum_subtb": False ,
                "do_parameterize_p_b": False
                },
        },
    ]
]

from gflownet.tasks.toy_seq import ToySeqTrainer

trial = ToySeqTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()
