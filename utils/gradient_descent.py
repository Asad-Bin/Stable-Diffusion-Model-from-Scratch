import io
from PIL import Image
import matplotlib.pyplot as plt
import mlflow

def plot_and_log_grad_norms(run_id, weight_metrics, bias_metrics, client):

    def plot_metrics(metrics, title):
        plt.figure(figsize=(10,6))
        for metric in metrics:
            history = client.get_metric_history(run_id, metric)
            steps = [p.step for p in history]
            values = [p.value for p in history]
            label = metric.replace("grad_norm_weight_", "").replace("grad_norm_bias_", "")
            plt.plot(steps, values, label=label)
        plt.xlabel("Step")
        plt.ylabel("Gradient Norm (L2)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Convert bytes buffer to PIL Image
        img = Image.open(buf)

        # Log PIL image to mlflow
        mlflow.log_image(img, artifact_file=f"{title.replace(' ', '_').lower()}.png")

    plot_metrics(weight_metrics, "Gradient Norms - Weights (First Conv Layers)")
    plot_metrics(bias_metrics, "Gradient Norms - Biases (First Conv Layers)")
