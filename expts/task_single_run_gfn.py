import sys

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_single_runs"

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1, # use 1000 might be faster
    "num_workers": 0, # o.g. 8
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
    "global_batch_size": 256, # 256
    "max_nodes": 7,
    "offline_ratio": 0, # 1 / 2
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_gfn/",
        "log_tags": ["test_dev"],
        "seed": 1,
        "run_valid_dl": False,
        
        "task": {
        "basic_graph": {
            "test_split_seed": 1, 
            "do_supervised": True, 
            "do_tabular_model": False, 
            "regress_to_P_F": True,
            "regress_to_Fsa": True,
            "train_ratio": 0.9,
            "reward_func": 'count', 
            "reward_reshape": True,
            "reward_corrupt": False,
            "reward_shuffle": False,
            "reward_param": 0.5,
            },
        },  
        
        "algo": {
            **base_algo_hps,
            #"dir_model_pretrain_for_sampling": "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_SubTB_flows_Nov_21/run_0/model_state.pt",
            #"supervised_reward_predictor": "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/supervised_gnn_models_for_bgfn_Dec_7/",
            #"offline_sampling_g_distribution": "uniform",
            #"flow_reg": False,
            #"valid_sample_cond_info": False,
            #"valid_offline_ratio": 1,
            **algo,
        },

        #"cond": {
        #    "logZ":{
        #        "sample_dist": None, #"uniform",  # set to None to not sample cond_info for logZ
        #        "dist_params": [1.0, 20.0],
        #        "num_thermometer_dim": 32,
        #        "num_valid_logZ_samples": 5,
        #    },
        #},
        
    }
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


from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[0])
trial.print_every = 1
trial.run()

#from gflownet.tasks.basic_graph_task import BGSupervisedTrainer

#trial = BGSupervisedTrainer(hps[0])
#@trial.print_every = 1
#trial.run()