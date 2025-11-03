import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch


def calculate_metrics(y_true, y_pred, genre_names=None):
    """
    Calculate classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        genre_names: List of genre names

    Returns:
        Dictionary containing metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    if genre_names is None:
        genre_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=genre_names,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

    return metrics


def plot_confusion_matrix(cm, genre_names, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        genre_names: List of genre names
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=genre_names,
        yticklabels=genre_names,
        cbar_kws={'label': 'Proportion'}
    )

    plt.title('Confusion Matrix (Normalized)', fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    plt.show()


def evaluate_model(model, dataloader, device, genre_names=None):
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run on
        genre_names: List of genre names

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, genre_names)
    metrics['loss'] = avg_loss

    return metrics, all_labels, all_preds


def print_evaluation_results(metrics, genre_names=None):
    """
    Print evaluation results in a formatted way

    Args:
        metrics: Dictionary from calculate_metrics
        genre_names: List of genre names
    """
    print("=" * 80)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print("=" * 80)

    if genre_names is None:
        genre_names = [f"Class {i}" for i in range(len(metrics['confusion_matrix']))]

    print("\nPer-Class Performance:")
    print("-" * 80)
    print(f"{'Genre':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)

    report = metrics['classification_report']
    for genre in genre_names:
        if genre in report:
            genre_metrics = report[genre]
            print(f"{genre:<15} {genre_metrics['precision']:<12.4f} "
                  f"{genre_metrics['recall']:<12.4f} "
                  f"{genre_metrics['f1-score']:<12.4f} "
                  f"{int(genre_metrics['support']):<10}")

    print("-" * 80)
    macro_avg = report['macro avg']
    print(f"{'Macro Avg':<15} {macro_avg['precision']:<12.4f} "
          f"{macro_avg['recall']:<12.4f} "
          f"{macro_avg['f1-score']:<12.4f}")

    weighted_avg = report['weighted avg']
    print(f"{'Weighted Avg':<15} {weighted_avg['precision']:<12.4f} "
          f"{weighted_avg['recall']:<12.4f} "
          f"{weighted_avg['f1-score']:<12.4f}")
    print("=" * 80)
