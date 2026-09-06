import numpy as np
import matplotlib.pyplot as plt

SEED = 17
np.random.seed(SEED)

NUM_CLASSES = 4
POINTS_PER_CLASS = 50
INPUT_DIM = 2
TRAIN_RATIO = 0.8


cluster_centers = np.array([
    [-3.0, -1.0],
    [2.8, -2.2],
    [-2.2, 2.6],
    [3.2, 2.4]
])


cluster_std = [1.2, 1.4, 1.1, 1.3]

def generate_dataset():
    samples = []
    labels = []

    for class_id in range(NUM_CLASSES):
        current_points = np.random.normal(
            loc=0.0,
            scale=cluster_std[class_id],
            size=(POINTS_PER_CLASS, INPUT_DIM)
        )
        current_points += cluster_centers[class_id]

        current_labels = np.full(POINTS_PER_CLASS, class_id)

        samples.append(current_points)
        labels.append(current_labels)

    X_all = np.vstack(samples)
    y_all = np.concatenate(labels)

    permutation = np.random.permutation(X_all.shape[0])
    X_all = X_all[permutation]
    y_all = y_all[permutation]

    split_index = int(TRAIN_RATIO * len(X_all))

    X_train = X_all[:split_index]
    y_train = y_all[:split_index]
    X_test = X_all[split_index:]
    y_test = y_all[split_index:]

    return X_all, y_all, X_train, y_train, X_test, y_test


def plot_dataset(X, y):
    plt.figure(figsize=(8, 6))

    for c in range(NUM_CLASSES):
        mask = (y == c)
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            alpha=0.75,
            label=f"Class {c}"
        )

    plt.title("Synthetic 2D Classification Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

class BinaryPerceptron:
    def __init__(self, n_features, learning_rate=0.08, epochs=15):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = np.random.uniform(-0.5, 0.5, size=n_features)
        self.b = np.random.uniform(-0.5, 0.5)

    def step_function(self, value):
        return 1 if value >= 0 else 0

    def raw_score(self, x):
        return np.dot(self.w, x) + self.b

    def predict_single(self, x):
        return self.step_function(self.raw_score(x))

    def fit(self, X, y):
        for _ in range(self.epochs):
            order = np.random.permutation(len(X))
            X_epoch = X[order]
            y_epoch = y[order]

            for xi, yi in zip(X_epoch, y_epoch):
                prediction = self.predict_single(xi)
                error = yi - prediction

                self.w += self.learning_rate * error * xi
                self.b += self.learning_rate * error

    def predict(self, X):
        preds = [self.predict_single(xi) for xi in X]
        return np.array(preds)

    def decision_values(self, X):
        return np.array([self.raw_score(xi) for xi in X])


class OneVsRestPerceptron:
    def __init__(self, n_classes, n_features):
        self.n_classes = n_classes
        self.classifiers = [
            BinaryPerceptron(n_features) for _ in range(n_classes)
        ]

    def fit(self, X, y):
        for class_id in range(self.n_classes):
            binary_targets = (y == class_id).astype(int)
            self.classifiers[class_id].fit(X, binary_targets)

    def predict(self, X):
        class_scores = []

        for model in self.classifiers:
            class_scores.append(model.decision_values(X))

        class_scores = np.vstack(class_scores)
        return np.argmax(class_scores, axis=0)


class OneVsOnePerceptron:
    def __init__(self, n_classes, n_features):
        self.n_classes = n_classes
        self.models = {}

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                self.models[(i, j)] = BinaryPerceptron(n_features)

    def fit(self, X, y):
        for (i, j), model in self.models.items():
            pair_mask = (y == i) | (y == j)
            X_pair = X[pair_mask]
            y_pair = y[pair_mask]

            y_binary = (y_pair == i).astype(int)
            model.fit(X_pair, y_binary)

    def predict(self, X):
        final_predictions = []

        for sample in X:
            votes = np.zeros(self.n_classes)

            for (i, j), model in self.models.items():
                pred = model.predict_single(sample)

                if pred == 1:
                    votes[i] += 1
                else:
                    votes[j] += 1

            final_predictions.append(np.argmax(votes))

        return np.array(final_predictions)


class SoftmaxClassifier:
    def __init__(self, n_features, n_classes, learning_rate=0.05, epochs=2000):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

    def to_one_hot(self, y):
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(z_shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def fit(self, X, y):
        y_encoded = self.to_one_hot(y)
        n_samples = X.shape[0]

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            probabilities = self.softmax(logits)

            gradient = probabilities - y_encoded

            dW = (X.T @ gradient) / n_samples
            db = np.sum(gradient, axis=0, keepdims=True) / n_samples

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        logits = X @ self.W + self.b
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)


X, y, X_train, y_train, X_test, y_test = generate_dataset()
plot_dataset(X, y)

ovr_model = OneVsRestPerceptron(NUM_CLASSES, INPUT_DIM)
ovr_model.fit(X_train, y_train)
y_pred_ovr = ovr_model.predict(X_test)
ovr_acc = accuracy_score(y_test, y_pred_ovr)

ovo_model = OneVsOnePerceptron(NUM_CLASSES, INPUT_DIM)
ovo_model.fit(X_train, y_train)
y_pred_ovo = ovo_model.predict(X_test)
ovo_acc = accuracy_score(y_test, y_pred_ovo)

softmax_model = SoftmaxClassifier(INPUT_DIM, NUM_CLASSES)
softmax_model.fit(X_train, y_train)
y_pred_softmax = softmax_model.predict(X_test)
softmax_acc = accuracy_score(y_test, y_pred_softmax)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print()
print(f"One-vs-Rest Perceptron accuracy: {ovr_acc:.4f}")
print(f"One-vs-One Perceptron accuracy : {ovo_acc:.4f}")
print(f"Softmax Logistic Regression accuracy: {softmax_acc:.4f}")

model_names = [
    "OVR Perceptron",
    "OVO Perceptron",
    "Softmax Logistic"
]
model_scores = [ovr_acc, ovo_acc, softmax_acc]

plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, model_scores)
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title("Classifier Performance on Test Set")
plt.grid(axis="y", linestyle="--", alpha=0.6)

for bar, score in zip(bars, model_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.015,
        f"{score:.3f}",
        ha="center"
    )

plt.show()

print("\nComparison:")
print("The perceptron-based approaches can work reasonably well, but they depend heavily on linear separability between classes.")
print("Softmax logistic regression usually gives more stable multiclass predictions because it learns all classes jointly in one model.")