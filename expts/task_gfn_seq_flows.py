import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_seq_CONST_flows_Jan_23"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000, # 100
    "num_workers": 8, # o.g. 8
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)

    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "seq_transformer": {
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
        "log_tags": ["gfn_seq_flows"],
        "seed": seed,

        "task": {
        "toy_seq": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9, #0.9,
            "reward_func": 'const', # currently only edit reward is supported
            },
        }, 

        #"cond": {
        #    "temperature": {
        #        "sample_dist": "constant",
        #        "dist_params": [1.0],
        #        "num_thermometer_dim": 1, # don't change this unless update cached data
        #    }
        #},
        
        "algo": {
            **base_algo_hps,
            #"train_random_action_prob": 0.05,
            "valid_offline_ratio": 1,
            **algo,
        },
        
    }
    for seed in [1, 2, 3]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", 
                "cum_subtb": False,
                "do_parameterize_p_b": False
                },
        },
    ]
]


from gflownet.tasks.toy_seq import ToySeqTrainer

trial = ToySeqTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()
