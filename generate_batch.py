import os
import itertools

def generate_batch_files():

    # Define parameter combinations to test
    param_combinations = {
        # model and data parameters
        'model_name': ['resnet_features_1d', 'cnn_features_1d'],
        'data_name': ['XJTU', 'XJTUFFT'],
        'data_dir': [r'/root/autodl-tmp/datasets/XJTU-SY_Bearing_Datasets'],
        'transfer_task': [[[0], [1]], [[1], [0]], [[0, 1], [2]], [[0, 2], [1]], [[1, 2], [0]]],
        'task_type': ['multi_class', 'multi_label'],
        'normlizetype': ['0-1'],
        
        # training parameters
        # 'batch_size': [256, 512],
        # 'bottleneck': [True, False],
        # 'bottleneck_num': [256, 512],
        # 'last_batch': [True, False],
        
        # distance metric parameters
        # 'distance_metric': [True, False],
        'distance_loss': ['MK-MMD', 'JMMD', 'CORAL'],
        
        # domain adversarial parameters
        # 'domain_adversarial': [True, False],
        'adversarial_loss': ['DA', 'CDA', 'CDA+E'],
        'hidden_size': [1024],
        
        # optimization parameters
        'opt': ['sgd', 'adam'],
        'lr': [1e-3],
        'momentum': [0.9],
        'weight_decay': [1e-4, 1e-5],
        'lr_scheduler': ['step', 'exp'],
        'gamma': [0.1, 0.99],
        'steps': ['150,250']
    }

    # Generate all combinations
    keys = param_combinations.keys()
    values = param_combinations.values()
    combinations = list(itertools.product(*values))

    # Generate batch files
    batch_content = "#!/bin/bash\n\n"  # Start with shebang
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if params['lr_scheduler'] == 'step' and params['gamma'] == 0.99:
            continue
        if params['lr_scheduler'] == 'exp' and params['gamma'] == 0.1:
            continue
        if params['lr_scheduler'] == 'stepLR' and params['gamma'] != 0.1:
            continue

        # Add experiment number as comment
        batch_content += f"# Experiment {i+1}\n"
        
        # Build command using key-value pairs
        command_parts = ["python train_advanced.py"]
        for key, value in params.items():
            command_parts.append(f"--{key} {value}")
        
        # Join all parts with spaces and add newlines
        batch_content += " ".join(command_parts) + "\n\n"

    # Save all experiments to a single file
    filepath = os.path.join('.', 'run_all_experiments.sh')
    with open(filepath, 'w', newline='\n') as f:
        f.write(batch_content)

if __name__ == '__main__':
    generate_batch_files()
    print("Batch files generated successfully!")
