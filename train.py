import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from models import MusicCNN, ImprovedMusicCNN, ResNetMusic
from utils import AudioProcessor, MusicGenreDataset, create_data_splits
from utils import AudioAugmentation, SpecAugment
from utils.metrics import evaluate_model, plot_training_history, print_evaluation_results
from utils.losses import FocalLoss, WeightedFocalLoss, calculate_class_weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_dir, early_stopping_patience=10):
    """
    Train the model with validation

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping

    Returns:
        history: Dictionary with training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_dir / 'best_model.pth')
            print(f"✓ Saved best model with val_acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    return history


def main():
    parser = argparse.ArgumentParser(description='Train music genre classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--model', type=str, default='MusicCNN',
                        choices=['MusicCNN', 'ImprovedMusicCNN', 'ResNetMusic'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')

    # NEW: Advanced options
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights to handle imbalance')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss instead of cross-entropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--augmentation', type=str, default='none',
                        choices=['none', 'basic', 'specaugment'],
                        help='Data augmentation strategy')
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                        help='Probability of applying augmentation')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create audio processor
    print("\nInitializing audio processor...")
    audio_processor = AudioProcessor(sample_rate=22050, n_mels=128, duration=3.0)

    # Setup augmentation
    transform = None
    if args.augmentation == 'basic':
        print(f"Using basic augmentation (p={args.augmentation_prob})")
        transform = AudioAugmentation(
            time_mask_param=30,
            freq_mask_param=20,
            p=args.augmentation_prob
        )
    elif args.augmentation == 'specaugment':
        print(f"Using SpecAugment (p={args.augmentation_prob})")
        transform = SpecAugment(
            time_mask_param=40,
            freq_mask_param=30,
            num_masks=2,
            p=args.augmentation_prob
        )

    # Create dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    full_dataset = MusicGenreDataset(args.data_dir, audio_processor)

    # Split dataset
    train_dataset, val_dataset, test_dataset = create_data_splits(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Apply augmentation only to training set
    if transform is not None:
        train_dataset.dataset.transform = transform
        print("✓ Augmentation enabled for training set only")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    num_classes = len(full_dataset.genres)

    if args.model == 'MusicCNN':
        model = MusicCNN(num_classes=num_classes)
    elif args.model == 'ImprovedMusicCNN':
        model = ImprovedMusicCNN(num_classes=num_classes)
    elif args.model == 'ResNetMusic':
        model = ResNetMusic(num_classes=num_classes)

    model = model.to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Setup loss function
    print("\nSetting up loss function...")

    if args.use_focal_loss and args.use_class_weights:
        # Weighted Focal Loss
        print("Using Weighted Focal Loss")
        class_weights = calculate_class_weights(train_dataset, num_classes, device)
        criterion = WeightedFocalLoss(
            class_weights=class_weights,
            gamma=args.focal_gamma
        )
    elif args.use_focal_loss:
        # Focal Loss only
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
        criterion = FocalLoss(gamma=args.focal_gamma)
    elif args.use_class_weights:
        # Cross-Entropy with class weights
        print("Using Cross-Entropy with class weights")
        class_weights = calculate_class_weights(train_dataset, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Standard Cross-Entropy
        print("Using standard Cross-Entropy Loss")
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Save config
    config = {
        'model': args.model,
        'num_classes': num_classes,
        'genres': full_dataset.genres,
        'genre_to_idx': full_dataset.genre_to_idx,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'use_class_weights': args.use_class_weights,
        'use_focal_loss': args.use_focal_loss,
        'focal_gamma': args.focal_gamma,
        'augmentation': args.augmentation,
        'augmentation_prob': args.augmentation_prob
    }

    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train model
    print("\nStarting training...")
    print("=" * 80)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=save_dir,
        early_stopping_patience=args.early_stopping
    )

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, save_path=save_dir / 'training_history.png')

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics, y_true, y_pred = evaluate_model(
        model, test_loader, device, full_dataset.genres
    )

    print_evaluation_results(metrics, full_dataset.genres)

    # Save test results
    test_results = {
        'test_accuracy': metrics['accuracy'],
        'test_loss': metrics['loss'],
        'classification_report': metrics['classification_report']
    }

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {save_dir}")
    print("\n" + "=" * 80)
    print("Training Tips:")
    print("  • For class imbalance: use --use_class_weights")
    print("  • For small datasets: use --augmentation basic or specaugment")
    print("  • For hard examples: use --use_focal_loss")
    print("  • Combine techniques: --use_class_weights --use_focal_loss --augmentation specaugment")
    print("=" * 80)


if __name__ == '__main__':
    main()
