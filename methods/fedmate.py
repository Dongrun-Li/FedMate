# coding: utf-8
import os
import json
from copy import deepcopy
import numpy as np
import tools
import torch
from torch import nn
import torch.nn.functional as F
from models import Discriminator

class LocalUpdate_FedMate(object):
    def __init__(self, idx, args, train_set, test_set, model, h_l):
        self.idx = idx
        self.args = args
        self.num_classes = args.num_classes
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Model setup
        self.local_model = model
        self.last_model = deepcopy(model)
        self.global_model = deepcopy(model)
        self.w_local_keys = self.local_model.classifier_weight_keys
        
        # Training parameters
        self.local_ep_rep = 1
        self.lam = args.lam
        self.h_l = h_l
        
        # Data statistics
        self.probs_label = self._get_label_probs(self.train_data).to(self.device)
        self.sizes_label = self._get_label_sizes(self.train_data).to(self.device)
        self.datasize = torch.tensor(len(self.train_data.dataset)).to(self.device)
        self.agg_weight = torch.tensor(len(self.train_data.dataset)).to(self.device)
        
        # Prototype storage
        self.global_protos = {}
        self.g_protos = None

    def _get_label_probs(self, dataset):
        """Calculate label probability distribution"""
        py = torch.zeros(self.args.num_classes)
        total = len(dataset.dataset)
        for images, labels in dataset:
            for i in range(self.args.num_classes):
                py[i] += (i == labels).sum()
        return py / total

    def _get_label_sizes(self, dataset):
        """Calculate label counts"""
        probs = self._get_label_probs(dataset)
        return probs * len(dataset.dataset)

    def test_and_save_results(self, test_loader, rnd):
        """Test model and save per-class accuracy"""
        # Load existing results or initialize
        results = {}
        if os.path.exists(self.test_results_file):
            if os.stat(self.test_results_file).st_size > 0:
                with open(self.test_results_file, 'r') as f:
                    results = json.load(f)
        
        # Skip if round already exists
        if rnd in results:
            return results[rnd]

        # Calculate per-class accuracy
        self.local_model.eval()
        correct = [0] * 10
        total = [0] * 10
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.local_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    total[label] += 1
                    correct[label] += (predicted[i] == labels[i]).item()

        # Calculate accuracy for each class
        class_acc = {}
        for i in range(10):
            class_acc[i] = 100.0 * correct[i] / total[i] if total[i] > 0 else 0.0

        # Save results
        results[rnd] = class_acc
        with open(self.test_results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return class_acc

    def local_test(self, test_loader):
        """Test local model accuracy"""
        self.local_model.eval()
        correct = 0
        total = len(test_loader.dataset)
        loss_test = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels)
                loss_test.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        acc = 100.0 * correct / total
        return acc, sum(loss_test) / len(loss_test)

    def local_test_model(self, test_loader, model_):
        """Test specific model accuracy"""
        model_.eval()
        correct = 0
        total = len(test_loader.dataset)
        loss_test = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = model_(inputs)
                loss = self.criterion(outputs, labels)
                loss_test.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        acc = 100.0 * correct / total
        return acc, sum(loss_test) / len(loss_test)

    def update_global_model(self, gl_model_weight):
        """Update global model weights"""
        global_weight = self.global_model.state_dict()
        for k in global_weight.keys():
            global_weight[k] = gl_model_weight[k]
        self.global_model.load_state_dict(global_weight)

    def update_base_model(self, global_weight):
        """Update base model (non-classifier layers)"""
        local_weight = self.local_model.state_dict()
        for k in local_weight.keys():
            if k not in self.w_local_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def update_local_classifier(self, new_weight):
        """Update local classifier weights"""
        local_weight = self.local_model.state_dict()
        for k in local_weight.keys():
            if k in self.w_local_keys:
                local_weight[k] = new_weight[k]
        self.local_model.load_state_dict(local_weight)

    def update_global_protos(self, global_protos):
        """Update global prototypes"""
        self.global_protos = global_protos
        g_classes, g_protos = [], []
        
        for i in range(self.num_classes):
            g_classes.append(torch.tensor(i))
            if i in global_protos:
                g_protos.append(global_protos[i])
            else:
                # Use zero vector as default for missing classes
                g_protos.append(torch.zeros_like(list(global_protos.values())[0]))
        
        self.g_classes = torch.stack(g_classes).to(self.device)
        self.g_protos = torch.stack(g_protos).to(self.device)

    def get_local_protos(self):
        """Extract local prototypes from training data"""
        local_protos_list = {}
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features, _ = self.local_model(inputs)
            protos = features.clone().detach()
            
            for i in range(len(labels)):
                label = labels[i].item()
                if label in local_protos_list:
                    local_protos_list[label].append(protos[i, :])
                else:
                    local_protos_list[label] = [protos[i, :]]
        
        return tools.get_protos(local_protos_list)

    def _init_discriminators(self):
        """Initialize adversarial discriminators"""
        if not hasattr(self, 'adv_discriminator1'):
            self.adv_discriminator1 = Discriminator(input_dim=self.args.num_classes).to(self.device)
        if not hasattr(self, 'adv_discriminator2'):
            self.adv_discriminator2 = Discriminator(input_dim=self.args.num_classes).to(self.device)

    def _get_adv_weights(self, round_num):
        """Calculate adversarial loss weights based on round"""
        increase = self.args.increase
        in_adv1 = (increase // 10) % 10
        in_adv2 = increase % 10
        
        base_weight = max(0.1, min(0.2, round_num / self.args.epochs))
        inv_weight = max(0.1, min(0.2, 1 - round_num / self.args.epochs))
        
        adv_weight1 = base_weight if in_adv1 else inv_weight
        adv_weight2 = max(0.1, min(0.5, 1 - round_num / self.args.epochs)) if not in_adv2 else base_weight
        
        return adv_weight1, adv_weight2

    def _train_classifier_phase(self, model, local_protos1, global_protos, round_num, epoch_classifier):
        """Train classifier with adversarial training"""
        # Enable only classifier parameters
        for name, param in model.named_parameters():
            param.requires_grad = name in self.w_local_keys

        self._init_discriminators()
        adv_weight1, adv_weight2 = self._get_adv_weights(round_num)
        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=0.1, momentum=0.5, weight_decay=0.0005)
        optimizer_D1 = torch.optim.SGD(self.adv_discriminator1.parameters(), lr=0.001, momentum=0.5, weight_decay=0.0005)
        optimizer_D2 = torch.optim.SGD(self.adv_discriminator2.parameters(), lr=0.001, momentum=0.5, weight_decay=0.0005)
        
        iter_loss = []
        for ep in range(epoch_classifier):
            for images, labels in self.train_data:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                
                # Standard classification loss
                protos, output = model(images)
                loss_ce = self.criterion(output, labels)
                
                # Adversarial loss 1: Local vs Global prototypes
                adv_loss1 = self._compute_adv_loss1(local_protos1, global_protos, optimizer_D1)
                
                # Adversarial loss 2: WGAN-style consistency loss
                adv_loss2 = self._compute_adv_loss2(global_protos, optimizer_D2)
                
                # Prototype consistency loss
                ce_proto_loss = self._compute_proto_ce_loss(local_protos1)
                
                # Combine losses based on configuration
                total_loss = self._combine_losses(loss_ce, adv_loss1, adv_loss2, ce_proto_loss, 
                                                adv_weight1, adv_weight2)
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                iter_loss.append(total_loss.item())
        
        return sum(iter_loss) / len(iter_loss) if iter_loss else 0

    def _compute_adv_loss1(self, local_protos1, global_protos, optimizer_D1):
        """Compute first adversarial loss (local vs global prototypes)"""
        # Prepare data
        local_feats = [feat for feats in local_protos1.values() 
                      for feat in (feats if isinstance(feats, list) else [feats])]
        global_feats = list(global_protos.values())
        
        if not local_feats or not global_feats:
            return 0
        
        # Combine features and labels
        adv1_inputs = torch.stack(local_feats + global_feats).to(self.device)
        adv1_targets = torch.tensor([0] * len(local_feats) + [1] * len(global_feats)).to(self.device)
        
        # Update discriminator
        adv1_outputs_D = self.local_model.feature2logit(adv1_inputs).detach()
        loss_D1 = F.cross_entropy(self.adv_discriminator1(adv1_outputs_D), adv1_targets)
        optimizer_D1.zero_grad()
        loss_D1.backward()
        optimizer_D1.step()
        
        # Generator loss
        adv1_outputs_G = self.local_model.feature2logit(adv1_inputs)
        gen_targets = torch.tensor([1] * len(local_feats) + [0] * len(global_feats)).to(self.device)
        return F.cross_entropy(self.adv_discriminator1(adv1_outputs_G), gen_targets)

    def _compute_adv_loss2(self, global_protos, optimizer_D2):
        """Compute second adversarial loss (WGAN-style consistency)"""
        global_feats = list(global_protos.values())
        if not global_feats:
            return 0
            
        adv2_inputs = torch.stack(global_feats).to(self.device)
        
        # Train critic (discriminator)
        real_outputs = self.global_model.feature2logit(adv2_inputs).detach()
        fake_outputs = self.local_model.feature2logit(adv2_inputs).detach()
        
        real_scores = self.adv_discriminator2(real_outputs)
        fake_scores = self.adv_discriminator2(fake_outputs)
        
        # WGAN loss with gradient penalty
        loss_D = -(torch.mean(real_scores) - torch.mean(fake_scores))
        
        # Gradient penalty
        alpha = torch.rand(real_outputs.size(0), 1).to(self.device)
        interpolates = (alpha * real_outputs + (1 - alpha) * fake_outputs).requires_grad_(True)
        d_interpolates = self.adv_discriminator2(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                       grad_outputs=torch.ones_like(d_interpolates),
                                       create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        loss_D_total = loss_D + 10.0 * gradient_penalty
        
        optimizer_D2.zero_grad()
        loss_D_total.backward()
        optimizer_D2.step()
        
        # Generator loss
        fake_outputs_G = self.local_model.feature2logit(adv2_inputs)
        fake_scores_G = self.adv_discriminator2(fake_outputs_G)
        return -torch.mean(fake_scores_G)

    def _compute_proto_ce_loss(self, local_protos1):
        """Compute prototype consistency cross-entropy loss"""
        ce_proto_loss = 0
        count_ce = 0
        
        for lbl, feats in local_protos1.items():
            feat_list = feats if isinstance(feats, list) else [feats]
            for feat in feat_list:
                feat_input = feat.unsqueeze(0).to(self.device)
                logits = self.local_model.feature2logit(feat_input)
                target = torch.tensor([lbl]).to(self.device)
                ce_proto_loss += F.cross_entropy(logits, target)
                count_ce += 1
        
        return ce_proto_loss / count_ce if count_ce > 0 else 0

    def _combine_losses(self, loss_ce, adv_loss1, adv_loss2, ce_proto_loss, adv_weight1, adv_weight2):
        """Combine different loss components based on configuration"""
        if self.args.tat == 0:
            return loss_ce
        
        tat = self.args.tat
        adv_loss1_included = (tat // 100) % 10
        adv_loss2_included = (tat // 10) % 10
        ce_proto_loss_included = tat % 10
        
        total_loss = loss_ce
        if adv_loss1_included:
            total_loss += adv_weight1 * adv_loss1
        if adv_loss2_included:
            total_loss += adv_weight2 * adv_loss2
        if ce_proto_loss_included:
            total_loss += ce_proto_loss
            
        return total_loss

    def _train_feature_phase(self, model, local_protos1, global_protos, round_num, local_ep_rep):
        """Train feature extractor phase"""
        # Enable only feature extractor parameters
        for name, param in model.named_parameters():
            param.requires_grad = name not in self.w_local_keys

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=self.args.lr, momentum=0.5, weight_decay=0.0005)
        
        iter_loss = []
        for ep in range(local_ep_rep):
            for images, labels in self.train_data:
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                
                protos, output = model(images)
                loss0 = self.criterion(output, labels)
                
                # Prototype alignment loss
                loss1 = 0
                if round_num > 0:
                    protos_new = protos.clone().detach()
                    for i, yi in enumerate(labels):
                        yi = yi.item()
                        if yi in global_protos:
                            protos_new[i] = global_protos[yi].detach()
                        elif yi in local_protos1:
                            protos_new[i] = local_protos1[yi].detach()
                    
                    mse_loss = F.mse_loss(protos_new, protos)
                    cosine_loss = 1 - F.cosine_similarity(protos_new, protos).mean()
                    loss1 = mse_loss  # Can be adjusted: mse_loss + weight * cosine_loss
                
                loss = loss0 + (self.lam * loss1 if self.args.proto else 0)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        
        return sum(iter_loss) / len(iter_loss) if iter_loss else 0

    def local_training(self, local_epoch, round=0):
        """Main local training function"""
        model = self.local_model
        model.train()
        
        # Test and save per-class accuracy if enabled
        if self.args.class_acc:
            self.test_and_save_results(self.test_data, rnd=round)
        
        # Initial evaluation
        acc0, _ = self.local_test(self.test_data)
        self.last_model = deepcopy(model)
        
        # Get initial local prototypes
        local_protos1 = self.get_local_protos()
        
        # Training configuration
        epoch_classifier = self.args.clepoch
        local_ep_rep = local_epoch
        total_epochs = int(epoch_classifier + local_ep_rep)
        
        round_losses = []
        
        if total_epochs > 0:
            # Phase 1: Train classifier with adversarial training
            loss1 = self._train_classifier_phase(model, local_protos1, self.global_protos, 
                                               round, epoch_classifier)
            round_losses.append(loss1)
            
            # Intermediate evaluation
            acc1, _ = self.local_test(self.test_data)
            
            # Phase 2: Train feature extractor
            loss2 = self._train_feature_phase(model, local_protos1, self.global_protos, 
                                            round, local_ep_rep)
            round_losses.append(loss2)
        
        # Final evaluation and return results
        local_protos2 = self.get_local_protos()
        acc2, _ = self.local_test(self.test_data)
        
        round_loss1 = round_losses[0] if round_losses else 0
        round_loss2 = round_losses[-1] if round_losses else 0
        
        return model.state_dict(), round_loss1, round_loss2, acc0, acc2, local_protos2