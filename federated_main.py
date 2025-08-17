import numpy as np
import torch
import torch.nn as nn
import time
import logging
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CifarCNN, CNN_FMNIST, Cifar100CNN, AnimalCNN
from options import args_parser


def setup_device(args):
    """Setup and return appropriate device"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def setup_random_seeds(seed=520):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_model(args, device):
    """Create and return appropriate model based on dataset"""
    model_configs = {
        ('cifar', 'cifar10', 'cinic', 'cinic_sep'): (CifarCNN, {'num_classes': args.num_classes}, 0.02),
        ('animal10', 'animal'): (AnimalCNN, {}, None),
        ('fmnist',): (CNN_FMNIST, {}, None),
        ('cifar100',): (Cifar100CNN, {'num_classes': args.num_classes}, None),
        ('mnist',): (CNN_FMNIST, {}, None),
        ('emnist',): (CNN_FMNIST, {'num_classes': 62}, None),
    }
    
    for dataset_names, (model_class, model_kwargs, lr) in model_configs.items():
        if args.dataset in dataset_names:
            if args.dataset == 'emnist':
                args.num_classes = 62
            if lr is not None:
                args.lr = lr
            return model_class(**model_kwargs).to(device)
    
    raise NotImplementedError(f"Model for dataset '{args.dataset}' not implemented")


def setup_logging(args):
    """Setup logging configuration"""
    log_filename = f"training_{args.dataset}_{args.train_rule}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_federated_model(args, global_model, train_round_func, logger):
    """Main training loop for federated learning"""
    # Initialize tracking variables
    train_losses = []
    local_accs1, local_accs2 = [], []
    local_clients = []
    
    logger.info(f"Starting federated training for {args.epochs} rounds")
    logger.info(f"Dataset: {args.dataset}, Algorithm: {args.train_rule}")
    logger.info(f"Clients: {args.num_users}, Local epochs: {args.local_ep}")
    
    # Training loop
    for round_num in range(args.epochs):
        start_time = time.time()
        
        # Perform one round of training
        loss1, loss2, local_acc1, local_acc2 = train_round_func(
            args, global_model, local_clients, round_num
        )
        
        # Track metrics
        train_losses.append(loss1)
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)
        
        # Calculate round time
        round_time = time.time() - start_time
        
        # Log progress
        logger.info(
            f"Round {round_num:3d}: Loss=({loss1:.4f}, {loss2:.4f}), "
            f"Acc=({local_acc1:.2f}%, {local_acc2:.2f}%), Time={round_time:.2f}s"
        )
        
        # Print to console for immediate feedback
        print(f"Round {round_num:3d}: Train Loss: {loss1:.4f}, {loss2:.4f}")
        print(f"Round {round_num:3d}: Local Accuracy: {local_acc1:.2f}%, {local_acc2:.2f}%")
    
    return {
        'train_losses': train_losses,
        'local_accs1': local_accs1,
        'local_accs2': local_accs2
    }


def save_results(results, args):
    """Save training results to file"""
    import json
    
    results_dict = {
        'args': vars(args),
        'metrics': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    filename = f"results_{args.dataset}_{args.train_rule}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {filename}")


def main():
    """Main function orchestrating the federated learning process"""
    # Parse arguments
    args = args_parser()
    
    # Setup environment
    device = setup_device(args)
    setup_random_seeds()
    logger = setup_logging(args)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, test_loader, global_test_loader, h_l = get_dataset(args)
    logger.info(f"Dataset loaded. Class balance metric: {h_l:.4f}")
    
    # Create model
    logger.info("Creating model...")
    global_model = create_model(args, device)
    logger.info(f"Model created: {global_model.__class__.__name__}")
    
    # Setup training components
    LocalUpdate = local_update(args.train_rule)
    train_round_func = one_round_training(args.train_rule)
    
    # Run training
    logger.info("Starting federated training...")
    training_start = time.time()
    
    results = train_federated_model(args, global_model, train_round_func, logger)
    
    training_time = time.time() - training_start
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save results
    save_results(results, args)
    
    # Print final summary
    final_acc1 = results['local_accs1'][-1] if results['local_accs1'] else 0
    final_acc2 = results['local_accs2'][-1] if results['local_accs2'] else 0
    final_loss = results['train_losses'][-1] if results['train_losses'] else 0
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.train_rule}")
    print(f"Rounds: {args.epochs}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {final_acc1:.2f}% / {final_acc2:.2f}%")
    print(f"Total Time: {training_time:.2f}s")
    print("="*60)


if __name__ == '__main__':
    main()