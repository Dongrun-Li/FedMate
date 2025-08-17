# coding: utf-8
import os
import json
import torch
from torch import nn

class LocalUpdate_StandAlone(object):
    def __init__(self, idx, args, train_set, test_set, model, h_l):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.h_l = h_l
        self.test_results_file = getattr(args, 'test_results_file', 
            f'results/{args.dataset}_class_acc.json')

    def local_test(self, test_loader):
        """Test model accuracy"""
        self.local_model.eval()
        correct = 0
        total = len(test_loader.dataset)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.local_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        return 100.0 * correct / total

    def test_and_save_results(self, test_loader, rnd):
        """Test and save per-class accuracy"""
        # Load existing results
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
        correct = [0] * 10  # Assumes 10 classes
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
        class_acc = {i: 100.0 * correct[i] / total[i] if total[i] > 0 else 0.0 
                    for i in range(10)}

        # Save results
        results[rnd] = class_acc
        os.makedirs(os.path.dirname(self.test_results_file), exist_ok=True)
        with open(self.test_results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return class_acc

    def _train_epochs(self, optimizer, num_epochs):
        """Common training loop"""
        iter_loss = []
        
        if num_epochs > 0:
            # Train for specified epochs
            for ep in range(num_epochs):
                for images, labels in self.train_data:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.local_model.zero_grad()
                    _, output = self.local_model(images)
                    loss = self.criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
        else:
            # Train for specified iterations (less than 1 epoch)
            data_loader = iter(self.train_data)
            for it in range(self.args.local_iter):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                _, output = self.local_model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        
        return iter_loss

    def local_training(self, local_epoch, round):
        """Standalone local training (no federation)"""
        self.local_model.train()
        
        # Test and save per-class accuracy if enabled
        if getattr(self.args, 'class_acc', False):
            self.test_and_save_results(self.test_data, rnd=round)

        acc1 = self.local_test(self.test_data)
        
        # Setup optimizer for all parameters
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr,
                                   momentum=0.5, weight_decay=0.0005)
        
        # Train model
        iter_loss = self._train_epochs(optimizer, local_epoch)
        
        # Final evaluation
        acc2 = self.local_test(self.test_data)
        round_loss1 = iter_loss[0] if iter_loss else 0
        round_loss2 = iter_loss[-1] if iter_loss else 0
        
        return self.local_model.state_dict(), round_loss1, round_loss2, acc1, acc2