import json
import matplotlib.pyplot as plt

def plot_metrics(metrics_file):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics["loss"], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig("visualization.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics("training_metrics.json")
