import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools
import pytorchtools

import dl_ivss
from utils import *
from store_result import make_calibration_plot
import train

class TrainingConfig:
    def __init__(self):
        self.n_epochs = 50
        self.patience = 7
        self.init_type = 'xavier'
        self.line_str = 'DBP+SBP+HR'
        self.base_path = '/home/PO_AKI/result'
        self.data_path = '/home/PO_AKI/data/train_validation_test_data'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_features', '-f', type=int, default=0,
                      help='number of tabular features (options: 0,3,11,28)')
    parser.add_argument('--device_num', '-d', type=int, default=0,
                      help='GPU device number')
    return parser.parse_args()

def get_hyperparameter_grid(num_features):
    return {
        'batch_size': [32, 64, 128, 256],
        'what_label': ['PO_AKI', 'Critical_AKI'],
        'what_model': ['DL_IVSS'],
        'learning_rate': [0.001, 0.0001],
        'weight_decay': [0.0001, 0.00001, 0.001],
        'num_features': [num_features]
    }

def load_data(config, what_label, num_features):
    base_path = f'{config.data_path}/{what_label}/discovery_cohort'
    
    def load_numpy(path):
        return np.load(path)
    
    # Load main data
    data = {
        'train': {
            'x': load_numpy(f'{base_path}/train/normalization_data.npy'),
            'y': load_numpy(f'{base_path}/train/label.npy')
        },
        'valid': {
            'x': load_numpy(f'{base_path}/valid/normalization_data.npy'),
            'y': load_numpy(f'{base_path}/valid/label.npy')
        }
    }
    
    # Load additional features if needed
    if num_features != 0:
        data['train']['demo'] = load_numpy(f'{base_path}/train/normalization_feature{num_features}.npy')
        data['valid']['demo'] = load_numpy(f'{base_path}/valid/normalization_feature{num_features}.npy')
    
    return data

def prepare_dataloaders(data, batch_size):
    # Convert to PyTorch tensors
    train_features = torch.Tensor(data['train']['x'])
    train_targets = torch.Tensor(data['train']['y'])
    val_features = torch.Tensor(data['valid']['x'])
    val_targets = torch.Tensor(data['valid']['y'])
    
    # Create datasets
    train_data = AKIDataset(train_features, train_targets)
    val_data = AKIDataset(val_features, val_targets)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)
    
    return train_loader, valid_loader

def initialize_model(model_name, num_features, device):
    if model_name == 'DL_IVSS':
        model = dl_ivss.dl_ivss(feature_num=num_features, fc_on_feature=True).to(device)
        return model
    raise ValueError(f"Unknown model: {model_name}")

def train_model(config, model, train_loader, valid_loader, params, demo_data=None):
    early_stopping = pytorchtools.EarlyStopping(
        patience=config.patience,
        verbose=False,
        path=os.path.join(params['RESULT_PATH'], 'checkpoint.pt')
    )
    
    metrics = {
        'train': {'acc': [], 'auc': [], 'loss': [], 'f1': []},
        'valid': {'acc': [], 'auc': [], 'loss': [], 'f1': []}
    }
    
    best_metrics = {
        'BEST_AUC_v': 0,
        'BEST_ACC_v': 0,
        'BEST_F1_v': 0,
        'BEST_AUPRC_v': 0,
        'BEST_LOSS_v': float('inf')
    }
    
    y_int_train = np.array(train_loader.dataset.targets, dtype=np.int32)
    counts_train = np.bincount(y_int_train)
    
    for epoch in range(config.n_epochs):
        print(f'# Epoch {epoch+1}/{config.n_epochs}')
        
        if demo_data:
            train_results, valid_results, best_metrics = train.run_train_valid(
                model, train_loader, valid_loader, counts_train, epoch,
                params, best_metrics, train_demography=demo_data['train'],
                valid_demography=demo_data['valid']
            )
        else:
            train_results, valid_results, best_metrics = train.run_train_valid(
                model, train_loader, valid_loader, counts_train, epoch,
                params, best_metrics
            )
        
        # Update metrics history
        for metric in metrics['train']:
            metrics['train'][metric].append(train_results[metric])
            metrics['valid'][metric].append(valid_results[metric])
        
        if early_stopping(valid_results['loss'], model):
            print("Early stopping triggered")
            break
            
    return metrics, best_metrics

def save_results(config, metrics, best_metrics, params):
    result_path = params['RESULT_PATH']
    
    # Save training curves
    for metric in ['acc', 'loss', 'auc', 'f1']:
        make_calibration_plot(
            metrics['train'][metric],
            metrics['valid'][metric],
            label=metric.capitalize(),
            path=result_path
        )
    
    # Save best validation results
    best_result_dict = {
        'best_val_acc': best_metrics['BEST_ACC_v'],
        'best_val_loss': best_metrics['BEST_LOSS_v'],
        'best_val_f1': best_metrics['BEST_F1_v'],
        'best_val_auc': best_metrics['BEST_AUC_v'],
        'best_val_auprc': best_metrics['BEST_AUPRC_v']
    }
    save_pickle(best_result_dict, os.path.join(result_path, 'best_validation_result.pkl'))

def update_best_hyperparameters(model_path, best_metrics, params, config):
    try:
        current_best = open_pickle(os.path.join(model_path, 'hyperparameter_best_auc.pkl'))
        current_best_auc = current_best['auroc']
    except:
        current_best_auc = 0
    
    if best_metrics['BEST_AUC_v'] > current_best_auc:
        print(f'Updating best parameters: {params["RESULT_PATH"]}')
        
        best_hyperparameter = {
            'batch_size': params['batch_size'],
            'what_model': params['what_model'],
            'what_init': config.init_type,
            'num_features': params['num_features'],
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay']
        }
        
        save_pickle({'auroc': best_metrics['BEST_AUC_v']},
                   os.path.join(model_path, 'hyperparameter_best_auc.pkl'))
        save_pickle(best_hyperparameter,
                   os.path.join(model_path, 'best_hyperparameter.pkl'))

def main():
    args = parse_args()
    config = TrainingConfig()
    device = torch.device(f'cuda:{args.device_num}')
    
    os.makedirs(config.base_path, exist_ok=True)
    
    params = get_hyperparameter_grid(args.num_features)
    params_combinations = list(itertools.product(*params.values()))
    print(f'Total {len(params_combinations)} hyperparameter combinations')
    
    for param_idx, values in enumerate(params_combinations, 1):
        param_dict = dict(zip(params.keys(), values))
        
        # Setup paths
        paths = {
            'label': os.path.join(config.base_path, param_dict['what_label']),
            'features': lambda p: os.path.join(p, str(param_dict['num_features'])),
            'model': lambda p: os.path.join(p, param_dict['what_model']),
            'result': lambda p: os.path.join(p, f"{param_dict['batch_size']}batch_size_"
                                              f"{param_dict['learning_rate']}lr_"
                                              f"{param_dict['weight_decay']}weight_decay_"
                                              f"{config.n_epochs}epoch")
        }
        
        current_path = paths['label']
        for path_func in [paths['features'], paths['model'], paths['result']]:
            current_path = path_func(current_path)
            os.makedirs(current_path, exist_ok=True)
        
        param_dict['RESULT_PATH'] = current_path
        
        if os.path.exists(current_path):
            print(f'Results already exist for {current_path}')
            continue
            
        # Load and prepare data
        data = load_data(config, param_dict['what_label'], param_dict['num_features'])
        train_loader, valid_loader = prepare_dataloaders(data, param_dict['batch_size'])
        
        # Initialize model
        model = initialize_model(param_dict['what_model'], param_dict['num_features'], device)
        weights_init(model, config.init_type)
        
        # Add remaining parameters
        param_dict.update({
            'device': device,
            'line_str': config.line_str
        })
        
        # Train model
        demo_data = {'train': data['train'].get('demo'), 'valid': data['valid'].get('demo')} \
                    if param_dict['num_features'] != 0 else None
        metrics, best_metrics = train_model(config, model, train_loader, valid_loader, param_dict, demo_data)
        
        # Save results
        save_results(config, metrics, best_metrics, param_dict)
        update_best_hyperparameters(paths['model'](paths['features'](paths['label'])),
                                  best_metrics, param_dict, config)

if __name__ == "__main__":
    main()