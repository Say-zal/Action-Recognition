import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from models.timesformer import load_timesformer_model
from training.dataset import get_dataloader
import os
import numpy as np

# Configure logging
logging.basicConfig(filename='evaluation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(data_dir, batch_size=8, model_path="timesformer_model_1.pth"):
    """
    Evaluate the model on the testing set.
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        model_path (str): Path to the trained model weights.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load data (returns train, val, test loaders, using only test loader)
    _, _, test_loader = get_dataloader(data_dir, batch_size=batch_size, clip_size=8, train_ratio=0.8, val_ratio=0.1)
    logging.info(f"Test loader size: {len(test_loader.dataset)} samples")

    # Load model and move to device
    try:
        processor, model = load_timesformer_model()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    except FileNotFoundError:
        logging.error(f"Model file {model_path} not found. Using pre-trained model without weights.")
        processor, model = load_timesformer_model(num_labels=25)
        model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(pixel_values=inputs).logits
            logging.info(f"Raw logits shape: {outputs.shape}")  # Debugging
            print(f"Raw logits shape: {outputs.shape}")
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)  # Use top-1 predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Debugging: Check predictions and ground truth
    logging.info(f"Test ground truth labels (y_true) range: {y_true.min()} to {y_true.max()}")
    logging.info(f"Test ground truth labels (y_true): {y_true}")
    logging.info(f"Test predicted labels (y_pred) range: {y_pred.min()} to {y_pred.max()}")
    logging.info(f"Test predicted labels (y_pred): {y_pred}")
    print(f"Test ground truth labels (y_true): {y_true}")
    print(f"Test predicted labels (y_pred): {y_pred}")

    # Top-1 Accuracy
    top1_acc = accuracy_score(y_true, y_pred)
    logging.info(f"Test Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Test Top-1 Accuracy: {top1_acc * 100:.2f}%")

    # Top-5 Accuracy
    y_probs_tensor = torch.tensor(y_probs)
    y_true_tensor = torch.tensor(y_true)
    top5_indices = torch.topk(y_probs_tensor, k=5, dim=1).indices  # Shape: [n_samples, 5]
    top5_correct = 0
    for i in range(len(y_true)):
        if y_true_tensor[i] in top5_indices[i]:
            top5_correct += 1
    top5_acc = top5_correct / len(y_true)
    logging.info(f"Test Top-5 Accuracy: {top5_acc * 100:.2f}% ({top5_correct}/{len(y_true)} samples)")
    print(f"Test Top-5 Accuracy: {top5_acc * 100:.2f}% ({top5_correct}/{len(y_true)} samples)")

    # Confusion Matrix with Enhanced Visualization
    class_names = [
        "brush_hair", "cartwheel", "catch", "chew", "climb", "climb_stairs", "draw_sword", 
        "eat", "fencing", "flic_flac", "golf", "handstand", "kiss", "pick", "pour", 
        "pullup", "pushup", "ride_bike", "shoot_bow", "shoot_gun", "situp", "smile", 
        "smoke", "throw", "wave"
    ]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", 
                annot_kws={"size": 8})
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("test_heatmap.png")
    plt.show()

    # Precision-Recall Curves (per class)
    plt.figure(figsize=(12, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        plt.plot(recall, precision,'o-', marker='.', label=class_names[i])
    plt.title("Test Precision-Recall Curves (One-vs-Rest)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("test_pr_curves.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model(data_dir="/user/HS402/zs00774/Downloads/HMDB_simp", batch_size=8)
