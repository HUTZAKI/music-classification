import torch
import argparse
from pathlib import Path
import json
import numpy as np

from models import MusicCNN, ImprovedMusicCNN, ResNetMusic
from utils import AudioProcessor


def predict_single_file(model, audio_path, audio_processor, device, genre_names, top_k=3):
    """
    Predict genre for a single audio file

    Args:
        model: Trained model
        audio_path: Path to audio file
        audio_processor: AudioProcessor instance
        device: Device to run on
        genre_names: List of genre names
        top_k: Number of top predictions to return

    Returns:
        Dictionary with predictions
    """
    model.eval()

    # Load and preprocess audio
    waveform = audio_processor.load_audio(audio_path)
    mel_spec = audio_processor.audio_to_melspec(waveform)

    # Add batch dimension
    mel_spec = mel_spec.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(mel_spec)
        probabilities = torch.softmax(outputs, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities[0], top_k)

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'genre': genre_names[idx],
            'probability': prob.item()
        })

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict music genre for audio file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to show')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    checkpoint_dir = Path(args.checkpoint).parent
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create audio processor
    audio_processor = AudioProcessor(sample_rate=22050, n_mels=128, duration=3.0)

    # Create model
    num_classes = config['num_classes']

    if config['model'] == 'MusicCNN':
        model = MusicCNN(num_classes=num_classes)
    elif config['model'] == 'ImprovedMusicCNN':
        model = ImprovedMusicCNN(num_classes=num_classes)
    elif config['model'] == 'ResNetMusic':
        model = ResNetMusic(num_classes=num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Predict
    print(f"\nPredicting genre for: {args.audio_path}")
    print("=" * 80)

    predictions = predict_single_file(
        model, args.audio_path, audio_processor,
        device, config['genres'], top_k=args.top_k
    )

    # Print results
    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 80)

    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['genre']:<15} {pred['probability']*100:>6.2f}%")

    print("=" * 80)


if __name__ == '__main__':
    main()
