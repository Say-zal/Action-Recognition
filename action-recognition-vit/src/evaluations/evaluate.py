import torch
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from models.timesformer import load_timesformer_model
from training.dataset import get_dataloader
import os
# Configure logging
logging.basicConfig(filename='evaluation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(data_dir, batch_size=8):
    _, val_loader = get_dataloader(data_dir, batch_size)
    extractor, model = load_timesformer_model()
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(pixel_values=inputs).logits
            print(f"Raw logits shape: {outputs.shape}")  # Debugging
            _, preds = torch.max(outputs, dim=1)  # Use top-1 predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = all_labels
    y_pred = all_preds

    # Debugging: Check predictions and ground truth
    print(f"Ground truth labels (y_true): {y_true}")
    print(f"Predicted labels (y_pred): {y_pred}")

    # Top-1 Accuracy
    top1_acc = accuracy_score(y_true, y_pred)
    logging.info(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig("heatmap.png")
    plt.show()

    # Classification report
    class_names = [
        "brush_hair", "cartwheel", "catch", "chew", "climb", "climb_stairs", "draw_sword", 
        "eat", "fencing", "flic_flac", "golf", "handstand", "kiss", "pick", "pour", 
        "pullup", "pushup", "ride_bike", "shoot_bow", "shoot_gun", "situp", "smile", 
        "smoke", "throw", "wave"
    ]
    labels = list(range(len(class_names)))  # Generate labels [0, 1, ..., 24]
    report = classification_report(y_true, y_pred, target_names=class_names, labels=labels, zero_division=0)
    print(report)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("eval.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model(data_dir="/user/HS402/zs00774/Downloads/HMDB_simp", batch_size=8)
