import sys

root = "./logs/gfn_single_runs/"

base_hps = {
    "num_training_steps": 20000,
    "validate_every": 100,
    "num_workers": 4,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {"num_layers": 2, "num_emb": 256},
    "opt": {"adam_eps": 1e-8, "learning_rate": 3e-4},
    "device": 'cuda',
}

base_algo_hps = {
    "global_batch_size": 64,
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_gfn/",
        "log_tags": ["test_tag_cgn"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": 0, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9,
            "reward_func": 'count', 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "method": "TB", # either TB or FM
            "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
        },
        
    }
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[0])
trial.print_every = 1
trial.run()