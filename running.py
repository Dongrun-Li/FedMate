import torch
import numpy as np
import copy
import pickle
import os
import tools
from tools import average_weights_weighted


def one_round_training(rule):
    """Factory function to return appropriate training round function"""
    training_methods = {
        'FedAvg': train_round_fedavg,
        'FedPer': train_round_fedper,
        'Local': train_round_standalone,
        'FedMate': train_round_fedmate,
    }
    
    if rule not in training_methods:
        raise ValueError(f"Unknown training rule: {rule}")
    
    return training_methods[rule]


def select_clients(args, rnd):
    """Select participating clients for current round"""
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if rnd >= args.epochs:
        m = num_users
    
    idx_users = np.random.choice(range(num_users), m, replace=False)
    return sorted(idx_users)


def aggregate_metrics(local_losses1, local_losses2, local_acc1, local_acc2):
    """Aggregate local metrics into averages"""
    return (
        sum(local_losses1) / len(local_losses1),
        sum(local_losses2) / len(local_losses2),
        sum(local_acc1) / len(local_acc1),
        sum(local_acc2) / len(local_acc2)
    )


def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    """Standalone training (no federation)"""
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    
    idx_users = select_clients(args, rnd)
    local_losses1, local_losses2, local_acc1, local_acc2 = [], [], [], []
    
    for idx in idx_users:
        local_client = local_clients[idx]
        w, loss1, loss2, acc1, acc2 = local_client.local_training(
            local_epoch=args.local_epoch, round=rnd
        )
        
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
    
    return aggregate_metrics(local_losses1, local_losses2, local_acc1, local_acc2)


def train_round_fedavg(args, global_model, local_clients, rnd, **kwargs):
    """Standard FedAvg training round"""
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    
    idx_users = select_clients(args, rnd)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1, local_acc2, agg_weights = [], [], []
    
    global_weight = global_model.state_dict()
    
    # Local training phase
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weights.append(local_client.agg_weight)
        
        # Update local model and train
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(
            local_epoch=args.local_epoch, round=rnd
        )
        
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
    
    # Global aggregation
    global_weight = average_weights_weighted(local_weights, agg_weights)
    global_model.load_state_dict(global_weight)
    
    return aggregate_metrics(local_losses1, local_losses2, local_acc1, local_acc2)


def train_round_fedper(args, global_model, local_clients, rnd, **kwargs):
    """FedPer training round (same as FedAvg but different local updates)"""
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    
    idx_users = select_clients(args, rnd)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1, local_acc2, agg_weights = [], [], []
    
    global_weight = global_model.state_dict()
    
    # Local training phase
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weights.append(local_client.agg_weight)
        
        # Update local model and train
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(
            local_epoch=args.local_epoch, round=rnd
        )
        
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
    
    # Global aggregation
    global_weight = average_weights_weighted(local_weights, agg_weights)
    global_model.load_state_dict(global_weight)
    
    return aggregate_metrics(local_losses1, local_losses2, local_acc1, local_acc2)


def save_prototypes(args, rnd, local_protos):
    """Save prototypes for visualization"""
    if not getattr(args, 'tsne', False):
        return
    
    proto_dir = f'/tmp/protos/{args.train_rule}'
    os.makedirs(proto_dir, exist_ok=True)
    proto_file = f'{proto_dir}/dataset_{args.dataset}_noniid_{args.noniid_s}_users_{args.num_users}.pt'
    
    try:
        with open(proto_file, 'rb') as f:
            protos_dict = pickle.load(f)
    except FileNotFoundError:
        protos_dict = {}
    
    protos_dict[rnd] = local_protos
    
    with open(proto_file, 'wb') as f:
        pickle.dump(protos_dict, f)


def calculate_parameter_ratio(weights, protos):
    """Calculate ratio between model parameters and prototypes"""
    classifier_keys = ['fc2.weight', 'fc2.bias']
    w_params = sum(p.numel() for name, p in weights.items() if name not in classifier_keys)
    
    if isinstance(protos, dict):
        protos_params = sum(p.numel() for p in protos.values())
    elif isinstance(protos, torch.Tensor):
        protos_params = protos.numel()
    else:
        return 1
    
    return int(w_params / protos_params) if protos_params > 0 else 1


def update_global_classifier(args, global_weight_new, global_protos, local_weights, sizes_label):
    """Update global classifier using prototype-based aggregation"""
    if not getattr(args, 'afa', False):
        return
    
    new_classifier_weights = tools.aggregate_classifier_per_class(
        global_protos, local_weights, sizes_label, args, num_classes=args.num_classes
    )
    
    global_weight_new['fc2.weight'].data.copy_(new_classifier_weights['fc2.weight'])
    global_weight_new['fc2.bias'].data.copy_(new_classifier_weights['fc2.bias'])


def update_local_clients(args, local_clients, global_weight_new, global_protos, multiple, rnd):
    """Update all local clients with global information"""
    for idx in range(args.num_users):
        local_client = local_clients[idx]
        
        # Update global model
        local_client.update_global_model(gl_model_weight=global_weight_new)
        
        # Update base model conditionally
        if getattr(args, 'upfe', False):
            local_client.update_base_model(global_weight=global_weight_new)
        else:
            if (rnd + 1) % multiple != 0:
                local_client.update_base_model(global_weight=global_weight_new)
        
        # Update prototypes and classifier
        local_client.update_global_protos(global_protos=global_protos)
        
        if getattr(args, 'upcl', False):
            local_client.update_local_classifier(new_weight=global_weight_new)


def train_round_fedmate(args, global_model, local_clients, rnd, **kwargs):
    """Fedmate training round with prototype aggregation"""
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    
    if rnd > args.epochs:
        return 0, 0, 0, 0
    
    idx_users = select_clients(args, rnd)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1, local_acc2, agg_weights, sizes_label, local_protos = [], [], [], [], []
    multiple = 1
    
    # Local training phase
    for idx in idx_users:
        local_client = local_clients[idx]
        sizes_label.append(local_client.sizes_label)
        
        # Local training
        w, loss1, loss2, acc1, acc2, protos = local_client.local_training(
            local_epoch=args.local_epoch, round=rnd
        )
        
        # Calculate parameter ratio (only for first client)
        if idx == idx_users[0]:
            multiple = calculate_parameter_ratio(w, protos)
        
        # Store results
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
        agg_weights.append(local_client.agg_weight)
        local_protos.append(copy.deepcopy(protos))
    
    # Save prototypes for visualization
    save_prototypes(args, rnd, local_protos)
    
    # Global aggregation
    agg_weights = torch.stack(agg_weights).to(args.device)
    
    # Update global model and prototypes
    if rnd == 0:
        global_weight_new = average_weights_weighted(local_weights, agg_weights)
        global_model.load_state_dict(global_weight_new)
        
        if getattr(args, 'mum', False):
            global_protos = tools.protos_aggregation_MuM(args, local_protos, sizes_label, global_model)
        else:
            global_protos = tools.protos_aggregation_test(local_protos, sizes_label)
    else:
        # Aggregate prototypes first
        if getattr(args, 'mum', False):
            global_protos = tools.protos_aggregation_MuM(args, local_protos, sizes_label, global_model)
        else:
            global_protos = tools.protos_aggregation_test(local_protos, sizes_label)
        
        # Update global model
        global_weight_new = average_weights_weighted(local_weights, agg_weights)
        global_model.load_state_dict(global_weight_new)
        
        # Update classifier if enabled
        update_global_classifier(args, global_weight_new, global_protos, local_weights, sizes_label)
    
    # Update local clients
    update_local_clients(args, local_clients, global_weight_new, global_protos, multiple, rnd)
    
    return aggregate_metrics(local_losses1, local_losses2, local_acc1, local_acc2)