import sys

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_single_seq_runs_2"

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 10, # 100
    "num_workers": 0, # o.g. 8
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
    "max_nodes": 10,
    "max_len": 11, # needs to be max_len = max_nodes + 1
    "offline_ratio": 0, # 1
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_gfn/",
        "log_tags": ["test_seq_dev_v2"],
        "seed": 1,

        "task": {
        "toy_seq": {
            "test_split_seed": 1, 
            "do_supervised": True, 
            "do_tabular_model": False, 
            "regress_to_P_F": True,
            "regress_to_Fsa": True,
            "train_ratio": 0.9,
            "reward_func": 'const', # currently only edit reward is supported
            "reward_reshape": False,
            "reward_corrupt": False,
            "reward_shuffle": False,
            "reward_param": 0.0,
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
            #"offline_sampling_g_distribution": 'log_rewards',
            #**algo,
        },
        
    }
    #for algo in [
    #    {
    #        "method": "TB", # either TB or FM
    #        "tb": {
    #            "variant": "SubTB1", 
    #            "cum_subtb": False,
    #            "do_parameterize_p_b": False
    #            },
    #    },
    #]
]


#from gflownet.tasks.toy_seq import ToySeqTrainer
#trial = ToySeqTrainer(hps[0])

from gflownet.tasks.toy_seq import SeqSupervisedTrainer
trial = SeqSupervisedTrainer(hps[0])

trial.print_every = 1
trial.run()
