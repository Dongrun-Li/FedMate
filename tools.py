import copy
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import json
import random
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from models import Autoencoder
import torch.nn as nn


# ============================================================================
# Core Prototype Functions
# ============================================================================

def get_protos(protos):
    """Returns the average of feature embeddings per class"""
    protos_mean = {}
    for label, proto_list in protos.items():
        proto = torch.zeros_like(proto_list[0])
        for feat in proto_list:
            proto += feat
        protos_mean[label] = proto / len(proto_list)
    return protos_mean


def protos_aggregation_test(local_protos_list, local_sizes_list):
    """Standard prototype aggregation with sample size weighting"""
    agg_protos_label = {}
    agg_sizes_label = {}
    
    # Collect prototypes and sizes
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        local_sizes = local_sizes_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
                agg_sizes_label[label] = [local_sizes[label]]

    # Weighted aggregation
    for label, proto_list in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        proto = torch.zeros_like(proto_list[0])
        for i in range(len(proto_list)):
            proto += sizes_list[i] * proto_list[i]
        agg_protos_label[label] = proto / sum(sizes_list)
    
    return agg_protos_label


# ============================================================================
# Advanced Prototype Aggregation
# ============================================================================

def calculate_quality_weight(proto, proto_list, epsilon=1e-12):
    """Calculate prototype quality weight based on entropy and cosine similarity"""
    similarities = []
    
    for other_proto in proto_list:
        if not torch.equal(proto, other_proto):
            # Calculate cosine similarity
            similarity = F.cosine_similarity(proto.unsqueeze(0), other_proto.unsqueeze(0))
            similarities.append(similarity.item())

    if len(similarities) == 0:
        return 1.0

    total_similarity = sum(similarities)
    if total_similarity == 0:
        return 1.0
    
    # Normalize to probabilities
    probabilities = [sim / total_similarity for sim in similarities]
    
    # Calculate entropy
    entropy = -sum(p * math.log(p + epsilon, 2) for p in probabilities)
    
    # Return entropy weight (lower entropy = higher quality)
    return 1.0 / (entropy + epsilon) if entropy > 0 else 1.0


def protos_aggregation_weighted_by_quality(local_protos_list, local_sizes_list):
    """Prototype aggregation weighted by quality and sample size"""
    agg_protos_label = {}
    agg_sizes_label = {}
    
    # Collect prototypes
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        local_sizes = local_sizes_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
                agg_sizes_label[label] = [local_sizes[label]]
    
    alpha = 0.5  # Balance between quality and size
    epsilon = 1e-12
    
    # Quality-weighted aggregation
    for label, proto_list in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        weighted_proto = torch.zeros_like(proto_list[0])
        total_weight = 0.0
        
        for i in range(len(proto_list)):
            # Quality weight based on entropy
            quality_weight = calculate_quality_weight(proto_list[i], proto_list, epsilon)
            
            # Size weight
            size_weight = sizes_list[i] / sum(sizes_list)
            
            # Combined weight
            final_weight = alpha * quality_weight + (1 - alpha) * size_weight
            total_weight += final_weight
            weighted_proto += final_weight * proto_list[i]
        
        agg_protos_label[label] = weighted_proto / total_weight if total_weight > 0 else weighted_proto
    
    return agg_protos_label


def compute_kmeans_center_with_autoencoder(data_list, encoding_dim=500, device='cuda:0', dataset='cifar'):
    """Compute cluster center using autoencoder for dimensionality reduction"""
    # Filter out zero data
    non_zero_data = [data for data in data_list if not torch.all(data == 0)]
    
    if len(non_zero_data) == 0:
        return torch.zeros(encoding_dim).to(device), torch.zeros(len(data_list)).to(device)

    data_tensor = torch.stack([
        data if isinstance(data, torch.Tensor) else torch.tensor(data) 
        for data in non_zero_data
    ]).to(device)
    
    # Define autoencoder
    input_dim = 1024 if dataset == 'cifar100' else 128
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train autoencoder
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(data_tensor)
        loss = criterion(output, data_tensor)
        loss.backward()
        optimizer.step()
    
    # Get encoded data and compute center
    with torch.no_grad():
        encoded_data = model.encoder(data_tensor)
    
    centers_cuda = torch.mean(encoded_data, dim=0)
    
    # Calculate similarity-based weights
    similarity = F.cosine_similarity(encoded_data, centers_cuda.unsqueeze(0).repeat(encoded_data.size(0), 1))
    similarity = torch.exp(similarity * 60)  # Amplify differences
    
    # Set weights for all data (including zeros)
    weights = torch.zeros(len(data_list), device=device)
    non_zero_indices = [i for i, data in enumerate(data_list) if not torch.all(data == 0)]
    weights[non_zero_indices] = similarity
    
    # Normalize non-zero weights
    non_zero_weights = weights[non_zero_indices]
    if non_zero_weights.sum() > 0:
        weights[non_zero_indices] = non_zero_weights / non_zero_weights.sum()
    
    return centers_cuda, weights


def test_proto_with_classifier(global_model, local_protos_list, use_softmax=True):
    """Test prototypes with classifier and compute weights"""
    label_logits = {}
    
    # Count label occurrences
    label_counts = {}
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            label_counts[label] = label_counts.get(label, 0) + 1
    
    # Compute logits for each label
    for label in label_counts:
        logit_list = []
        
        for local_protos in local_protos_list:
            if label in local_protos:
                proto = local_protos[label]
                proto_tensor = torch.tensor(proto, dtype=torch.float32).unsqueeze(0)
                proto_tensor = proto_tensor.to(global_model.fc2.weight.device)
                
                logits = global_model.feature2logit(proto_tensor)
                label_logit = logits[0, label]
                logit_list.append(label_logit)
            else:
                logit_list.append(torch.tensor(0.0, device=global_model.fc2.weight.device))
        
        label_logits[label] = logit_list
    
    # Calculate weights
    normalized_label_weights = {}
    for label, logit_list in label_logits.items():
        logit_tensor = torch.stack(logit_list)
        weights = torch.ones_like(logit_tensor)
        weights[logit_tensor == 0] = 0
        
        non_zero_logit_tensor = logit_tensor[logit_tensor != 0]
        if non_zero_logit_tensor.numel() > 0:
            if use_softmax:
                softmax_weights = torch.softmax(non_zero_logit_tensor, dim=0)
                weights[logit_tensor != 0] = softmax_weights
            else:
                non_zero_logit_tensor = non_zero_logit_tensor - non_zero_logit_tensor.min() + 1e-6
                normalized_weights = non_zero_logit_tensor / torch.sum(non_zero_logit_tensor)
                weights[logit_tensor != 0] = normalized_weights
        
        normalized_label_weights[label] = weights
    
    # Aggregate prototypes using weights
    aggregated_protos = {}
    for label in normalized_label_weights:
        weighted_proto_sum = None
        weights = normalized_label_weights[label]
        
        proto_idx = 0
        for local_protos in local_protos_list:
            if label in local_protos:
                proto_tensor = torch.tensor(local_protos[label], dtype=torch.float32)
                weighted_proto = weights[proto_idx] * proto_tensor
                
                if weighted_proto_sum is None:
                    weighted_proto_sum = weighted_proto
                else:
                    weighted_proto_sum += weighted_proto
                proto_idx += 1
        
        aggregated_protos[label] = weighted_proto_sum
    
    return normalized_label_weights, aggregated_protos


def js_divergence(p, q):
    """Calculate Jensen-Shannon divergence"""
    m = [(p_i + q_i) * 0.5 for p_i, q_i in zip(p, q)]
    kl_pm = sum(p_i * math.log(p_i / m_i) if p_i != 0 else 0 for p_i, m_i in zip(p, m))
    kl_qm = sum(q_i * math.log(q_i / m_i) if q_i != 0 else 0 for q_i, m_i in zip(q, m))
    return 0.5 * (kl_pm + kl_qm)


def protos_aggregation_MuM(args, local_protos_list, local_sizes_list, global_model):
    """Multi-level prototype aggregation with quality assessment"""
    # First stage: sample size weighted aggregation
    first_stage_protos, first_stage_weights = _first_stage_aggregation(local_protos_list, local_sizes_list)
    
    # Second stage: autoencoder-based clustering weights
    second_stage_weights = _second_stage_clustering(local_protos_list, args)
    
    # Optional third stage: classifier-based weights
    if getattr(args, 'afa', False):
        third_stage_weights, _ = test_proto_with_classifier(global_model, local_protos_list)
        return _three_stage_fusion(local_protos_list, first_stage_weights, 
                                 second_stage_weights, third_stage_weights)
    else:
        return _two_stage_fusion(local_protos_list, first_stage_weights, second_stage_weights)


def _first_stage_aggregation(local_protos_list, local_sizes_list):
    """First stage: standard size-weighted aggregation"""
    agg_protos = protos_aggregation_test(local_protos_list, local_sizes_list)
    
    weights = {}
    for idx in range(len(local_protos_list)):
        local_sizes = local_sizes_list[idx]
        for label in local_protos_list[idx].keys():
            if label not in weights:
                weights[label] = {}
            # Calculate total size for this label across all clients
            total_size = sum(local_sizes_list[i].get(label, 0) for i in range(len(local_protos_list)))
            weights[label][idx] = local_sizes[label] / total_size if total_size > 0 else 0
    
    return agg_protos, weights


def _second_stage_clustering(local_protos_list, args):
    """Second stage: autoencoder-based clustering weights"""
    all_labels = set()
    for local_protos in local_protos_list:
        all_labels.update(local_protos.keys())
    
    label_data = {}
    for label in all_labels:
        data_list = []
        for local_protos in local_protos_list:
            if label in local_protos:
                data_list.append(copy.deepcopy(local_protos[label]))
            else:
                # Fill with zero vector for missing labels
                data_list.append(torch.zeros_like(next(iter(local_protos.values()))))
        label_data[label] = data_list
    
    weights = {}
    for label, data_list in label_data.items():
        _, label_weights = compute_kmeans_center_with_autoencoder(
            data_list, device=args.device, dataset=args.dataset
        )
        weights[label] = label_weights
    
    return weights


def _two_stage_fusion(local_protos_list, first_weights, second_weights):
    """Fuse first and second stage weights"""
    # Calculate JS divergence between weight distributions
    js_results = {}
    for label in first_weights:
        first_stage_list = list(first_weights[label].values())
        second_stage_list = second_weights[label].cpu().numpy().tolist()
        js_results[label] = js_divergence(first_stage_list, second_stage_list)
    
    # Compute fusion weights based on JS divergence
    fused_protos = {}
    for label in js_results:
        js_value = js_results[label]
        total_js = js_value + js_value
        first_weight = js_value / total_js
        second_weight = js_value / total_js
        
        # Aggregate prototypes using fused weights
        fused_proto = torch.zeros_like(next(iter(local_protos_list[0].values())))
        for idx, local_protos in enumerate(local_protos_list):
            if label in local_protos:
                combined_weight = (first_weight * first_weights[label].get(idx, 0) + 
                                 second_weight * second_weights[label][idx].item())
                fused_proto += combined_weight * local_protos[label]
        
        fused_protos[label] = fused_proto
    
    return fused_protos


def _three_stage_fusion(local_protos_list, first_weights, second_weights, third_weights):
    """Fuse all three stage weights using JS divergence"""
    fused_protos = {}
    
    for label in first_weights:
        # Get weight lists
        first_list = list(first_weights[label].values())
        second_list = second_weights[label].cpu().numpy().tolist()
        third_list = third_weights[label].cpu().numpy().tolist()
        
        # Calculate pairwise JS divergences
        js12 = js_divergence(first_list, second_list)
        js13 = js_divergence(first_list, third_list)
        js23 = js_divergence(second_list, third_list)
        
        # Compute fusion weights
        total = js12 + js13 + js23
        w1 = (js12 + js13) / total
        w2 = (js12 + js23) / total
        w3 = (js13 + js23) / total
        
        # Normalize
        total_w = w1 + w2 + w3
        w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w
        
        # Aggregate prototypes
        fused_proto = torch.zeros_like(next(iter(local_protos_list[0].values())))
        for idx, local_protos in enumerate(local_protos_list):
            if label in local_protos:
                combined_weight = (w1 * first_weights[label].get(idx, 0) + 
                                 w2 * second_weights[label][idx].item() + 
                                 w3 * third_weights[label][idx].item())
                fused_proto += combined_weight * local_protos[label]
        
        fused_protos[label] = fused_proto
    
    return fused_protos


# ============================================================================
# Model Weight Functions
# ============================================================================

def model_dist(w_1, w_2):
    """Calculate Euclidean distance between model weights"""
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1:
        dist = torch.norm(w_1[key].cpu() - w_2[key].cpu())
        dist_total += dist.cpu()
    return dist_total.cpu().item()


def average_weights_weighted(w, avg_weight):
    """Weighted average of model weights with distance-based adjustment"""
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(avg_weight)
    agg_w = weight / weight.sum(dim=0)

    # Initial weighted aggregation
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += agg_w[i] * w[i][key]
    
    # Calculate distances and adjust weights
    dist_list = [model_dist(w[i], w_avg) for i in range(len(w))]
    aggnew_w = [np.exp(-dist_list[i] / weight[i].cpu()) * agg_w[i].cpu() for i in range(len(w))]
    total = sum(aggnew_w)
    aggnew_w = [aggnew_w[i] / total for i in range(len(w))]

    # Final aggregation with adjusted weights
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += aggnew_w[i] * w[i][key]

    return w_avg


def aggregate_classifier_per_class(global_protos, local_weights, local_sizes_list, args, num_classes=10, lr=0.01, gamma=1.0):
    """Aggregate classifier weights per class with prototype-based training"""
    device = next(iter(local_weights[0].values())).device
    
    # Convert sizes to dict format if tensor
    if isinstance(local_sizes_list, torch.Tensor):
        sizes_dict_list = []
        for i in range(local_sizes_list.shape[0]):
            size_dict = {
                cls: float(local_sizes_list[i][cls].item())
                for cls in range(local_sizes_list.shape[1])
                if local_sizes_list[i][cls] > 0
            }
            sizes_dict_list.append(size_dict)
        local_sizes_list = sizes_dict_list

    # Extract classifier weights
    fc2_weight_list = [w['fc2.weight'] for w in local_weights]
    fc2_bias_list = [w['fc2.bias'] for w in local_weights]

    # Collect weights and sizes by label
    agg_weight_label = {}
    agg_bias_label = {}
    agg_sizes_label = {}

    for idx in range(len(local_weights)):
        local_sizes = local_sizes_list[idx]
        fc2_weight = fc2_weight_list[idx]
        fc2_bias = fc2_bias_list[idx]

        for label in range(num_classes):
            count = float(local_sizes[label])
            if label not in agg_weight_label:
                agg_weight_label[label] = []
                agg_bias_label[label] = []
                agg_sizes_label[label] = []
            agg_weight_label[label].append(fc2_weight[label])
            agg_bias_label[label].append(fc2_bias[label])
            agg_sizes_label[label].append(count)

    # Aggregate weights
    new_fc2_weight = torch.zeros_like(fc2_weight_list[0])
    new_fc2_bias = torch.zeros_like(fc2_bias_list[0])

    for label in agg_weight_label.keys():
        weight_list = agg_weight_label[label]
        bias_list = agg_bias_label[label]
        size_list = agg_sizes_label[label]
        total = sum(size_list)
        
        if total == 0:
            continue

        # Sample size weighted aggregation
        for i in range(len(weight_list)):
            w1 = size_list[i] / total
            new_fc2_weight[label] += w1 * weight_list[i]
            new_fc2_bias[label] += w1 * bias_list[i]
    
    # Optional classifier training with prototypes
    if getattr(args, 'mum', False):
        optimizer = torch.optim.SGD([new_fc2_weight, new_fc2_bias], lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for label in global_protos.keys():
            proto_tensor = global_protos[label].to(device).unsqueeze(0)
            logits = torch.matmul(proto_tensor, new_fc2_weight.T) + new_fc2_bias
            target = torch.tensor([label], device=device)
            loss = criterion(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {'fc2.weight': new_fc2_weight, 'fc2.bias': new_fc2_bias}