import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json

from models import MusicCNN, ImprovedMusicCNN, ResNetMusic
from utils import AudioProcessor, MusicGenreDataset
from utils.metrics import evaluate_model, plot_confusion_matrix, print_evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate music genre classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint_dir = Path(args.checkpoint).parent

    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create audio processor
    print("\nInitializing audio processor...")
    audio_processor = AudioProcessor(sample_rate=22050, n_mels=128, duration=3.0)

    # Create dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = MusicGenreDataset(
        args.data_dir,
        audio_processor,
        genre_to_idx=config['genre_to_idx']
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create model
    print(f"\nCreating model: {config['model']}")
    num_classes = config['num_classes']

    if config['model'] == 'MusicCNN':
        model = MusicCNN(num_classes=num_classes)
    elif config['model'] == 'ImprovedMusicCNN':
        model = ImprovedMusicCNN(num_classes=num_classes)
    elif config['model'] == 'ResNetMusic':
        model = ResNetMusic(num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    print("\nEvaluating model...")
    print("=" * 80)

    metrics, y_true, y_pred = evaluate_model(
        model, dataloader, device, config['genres']
    )

    # Print results
    print_evaluation_results(metrics, config['genres'])

    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        config['genres'],
        save_path=save_dir / 'confusion_matrix.png'
    )

    # Save results
    results = {
        'accuracy': metrics['accuracy'],
        'loss': metrics['loss'],
        'classification_report': metrics['classification_report']
    }

    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
