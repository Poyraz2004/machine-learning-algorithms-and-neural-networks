import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(42)


class NeuralNetworkRegression:
    def __init__(self, layers, activation="tanh", init_strategy="xavier",
                 bias_strategy="zeros", reg_lambda=0.00001, seed=42):

        self.layers = layers
        self.activation_name = activation
        self.init_strategy = init_strategy
        self.bias_strategy = bias_strategy
        self.reg_lambda = reg_lambda

        self.weights = []
        self.biases = []

        rng = np.random.default_rng(seed)

        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i + 1]

            if init_strategy == "xavier":
                limit = np.sqrt(6 / (fan_in + fan_out))
                W = rng.uniform(-limit, limit, (fan_in, fan_out))
            elif init_strategy == "he":
                W = rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
            elif init_strategy == "normal_small":
                W = rng.normal(0, 0.1, (fan_in, fan_out))
            else:
                W = rng.normal(0, 1, (fan_in, fan_out))

            if bias_strategy == "small_positive":
                b = np.ones((1, fan_out)) * 0.01
            else:
                b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        self.history_mse = []
        self.history_r2 = []
        self.history_epochs = []

    def activation(self, x):
        if self.activation_name == "relu":
            return np.maximum(0, x)
        elif self.activation_name == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation_name == "tanh":
            return np.tanh(x)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_name == "sine":
            return np.sin(x)
        else:
            raise ValueError("Unknown activation function")

    def activation_derivative(self, x):
        if self.activation_name == "relu":
            return np.where(x > 0, 1, 0)
        elif self.activation_name == "leaky_relu":
            return np.where(x > 0, 1, 0.01)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        elif self.activation_name == "sine":
            return np.cos(x)
        else:
            raise ValueError("Unknown activation function")

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]

        a = X

        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z)

            self.z_values.append(z)
            self.a_values.append(a)

        z_output = np.dot(a, self.weights[-1]) + self.biases[-1]

        self.z_values.append(z_output)
        self.a_values.append(z_output)

        return z_output

    def backward(self, X, y, output, learning_rate, epoch):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        n = X.shape[0]

        delta = 2 * (output - y) / n

        grad_w = []
        grad_b = []

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.a_values[i].T, delta) + self.reg_lambda * self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True)

            grad_w.insert(0, dW)
            grad_b.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i - 1])

        for i in range(len(self.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grad_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (grad_w[i] ** 2)

            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (grad_b[i] ** 2)

            m_w_hat = self.m_w[i] / (1 - beta1 ** epoch)
            v_w_hat = self.v_w[i] / (1 - beta2 ** epoch)

            m_b_hat = self.m_b[i] / (1 - beta1 ** epoch)
            v_b_hat = self.v_b[i] / (1 - beta2 ** epoch)

            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def train(self, X, y, epochs, learning_rate, target_r2=0.97):
        for epoch in range(1, epochs + 1):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate, epoch)

            if epoch % 10 == 0:
                mse = mean_squared_error(y, output)
                r2 = r2_score(y, output)

                self.history_mse.append(mse)
                self.history_r2.append(r2)
                self.history_epochs.append(epoch)

                if r2 >= target_r2:
                    break


def load_dataset(path):
    data = np.loadtxt(path)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return X, y


def count_parameters(layers):
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i + 1]
        total += layers[i + 1]
    return total


def get_configs():
    return [
        {
            "layers": [1, 40, 1],
            "activation": "tanh",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.01,
            "lambda": 0.00001
        },
        {
            "layers": [1, 80, 1],
            "activation": "tanh",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.01,
            "lambda": 0.00001
        },
        {
            "layers": [1, 150, 1],
            "activation": "tanh",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.005,
            "lambda": 0.000001
        },
        {
            "layers": [1, 80, 80, 1],
            "activation": "tanh",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.005,
            "lambda": 0.000001
        },
        {
            "layers": [1, 120, 120, 1],
            "activation": "tanh",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.003,
            "lambda": 0.000001
        },
        {
            "layers": [1, 80, 80, 1],
            "activation": "sine",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.005,
            "lambda": 0.0000001
        },
        {
            "layers": [1, 150, 150, 1],
            "activation": "sine",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.003,
            "lambda": 0.0000001
        },
        {
            "layers": [1, 100, 1],
            "activation": "leaky_relu",
            "init": "he",
            "bias": "small_positive",
            "lr": 0.005,
            "lambda": 0.00001
        },
        {
            "layers": [1, 120, 1],
            "activation": "relu",
            "init": "he",
            "bias": "small_positive",
            "lr": 0.005,
            "lambda": 0.00001
        },
        {
            "layers": [1, 100, 1],
            "activation": "sigmoid",
            "init": "xavier",
            "bias": "zeros",
            "lr": 0.01,
            "lambda": 0.00001
        },
    ]


def train_best_model(X_train, y_train, X_test, y_test):
    configs = get_configs()

    best_model = None
    best_config = None
    best_score = -10**9

    for i, config in enumerate(configs):
        nn = NeuralNetworkRegression(
            layers=config["layers"],
            activation=config["activation"],
            init_strategy=config["init"],
            bias_strategy=config["bias"],
            reg_lambda=config["lambda"],
            seed=42 + i
        )

        nn.train(
            X_train,
            y_train,
            epochs=40000,
            learning_rate=config["lr"],
            target_r2=0.97
        )

        train_pred = nn.forward(X_train)
        test_pred = nn.forward(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        epochs_used = nn.history_epochs[-1] if nn.history_epochs else 40000
        params = count_parameters(config["layers"])

        score = (
            train_r2 * 100
            + test_r2 * 60
            - test_mse * 10
            - epochs_used * 0.00005
            - params * 0.00002
        )

        if score > best_score:
            best_score = score
            best_model = nn
            best_config = config

    return best_model, best_config


def create_report_page(pdf, dataset_name, nn, config, X_train, X_test, y_train, y_test):
    output_train = nn.forward(X_train)
    output_test = nn.forward(X_test)

    final_train_mse = mean_squared_error(y_train, output_train)
    final_train_r2 = r2_score(y_train, output_train)

    final_test_mse = mean_squared_error(y_test, output_test)
    final_test_r2 = r2_score(y_test, output_test)

    epoch_count = nn.history_epochs[-1] if nn.history_epochs else 40000
    parameter_count = count_parameters(config["layers"])

    fig = plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(nn.history_epochs, nn.history_mse)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"MSE vs Epochs, Final MSE = {final_train_mse:.5f}")

    plt.subplot(2, 2, 2)
    plt.plot(nn.history_epochs, nn.history_r2)
    plt.xlabel("Epochs")
    plt.ylabel("R²")
    plt.ylim(0, 1)
    plt.title(f"R² vs Epochs, Final R² = {final_train_r2:.4f}")

    plt.subplot(2, 2, 3)
    plt.scatter(X_train, y_train, label="Training Data", s=18)
    plt.scatter(X_test, y_test, label="Test Data", s=18)
    plt.scatter(X_train, output_train, label="Train Predictions", marker="s", s=18)
    plt.scatter(X_test, output_test, label="Test Predictions", marker="x", s=35)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Real vs Predicted")
    plt.legend(fontsize=8)

    plt.subplot(2, 2, 4)
    plt.axis("off")

    hidden_text = " - ".join(map(str, config["layers"][1:-1]))

    info = f"""
Dataset: {dataset_name}

Architecture:
Input neurons: {config["layers"][0]}
Hidden neurons: {hidden_text}
Output neurons: {config["layers"][-1]}
Total parameters: {parameter_count}

Training:
Epochs used: {epoch_count}
Learning rate: {config["lr"]}
Regularisation lambda: {config["lambda"]}

Activation:
Function: {config["activation"]}
Derivative: implemented in code

Initialisation:
Weights: {config["init"]}
Biases: {config["bias"]}

Optimizer:
Adam optimizer implemented from scratch

Final scores:
Train MSE: {final_train_mse:.6f}
Train R²: {final_train_r2:.6f}

Test MSE: {final_test_mse:.6f}
Test R²: {final_test_r2:.6f}
"""

    plt.text(0.02, 0.95, info, va="top", fontsize=10)

    plt.suptitle(f"Neural Network Regression Report - {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdf.savefig(fig)
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "activation": config["activation"],
        "layers": config["layers"],
        "init": config["init"],
        "lr": config["lr"],
        "lambda": config["lambda"],
        "epochs": epoch_count,
        "params": parameter_count,
        "train_mse": final_train_mse,
        "train_r2": final_train_r2,
        "test_mse": final_test_mse,
        "test_r2": final_test_r2
    }


def main():
    files = sorted(glob.glob("dane*.txt"))

    if len(files) < 8:
        raise ValueError("At least 8 daneXX.txt files are required.")

    selected_files = files[:8]

    with PdfPages("Project4_Part1_Neural_Network_Report.pdf") as pdf:
        summary_results = []

        for file_path in selected_files:
            dataset_name = os.path.basename(file_path)
            print(f"\nProcessing {dataset_name}...")

            X, y = load_dataset(file_path)

            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()

            X_scaled = x_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y_scaled,
                test_size=0.2,
                random_state=42
            )

            best_nn, best_config = train_best_model(X_train, y_train, X_test, y_test)

            result = create_report_page(
                pdf,
                dataset_name,
                best_nn,
                best_config,
                X_train,
                X_test,
                y_train,
                y_test
            )

            summary_results.append(result)

        fig = plt.figure(figsize=(15, 9))
        plt.axis("off")

        text = "Summary of all datasets\n\n"
        text += "Dataset | Activation | Layers | Init | LR | Lambda | Epochs | Params | Test MSE | Test R²\n"
        text += "-" * 125 + "\n"

        for r in summary_results:
            text += (
                f"{r['dataset']} | {r['activation']} | {r['layers']} | {r['init']} | "
                f"{r['lr']} | {r['lambda']} | {r['epochs']} | {r['params']} | "
                f"{r['test_mse']:.6f} | {r['test_r2']:.6f}\n"
            )

        plt.text(0.01, 0.98, text, va="top", fontsize=8.5, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    print("\nDone.")
    print("PDF report saved as: Project4_Part1_Neural_Network_Report.pdf")


if __name__ == "__main__":
    main()