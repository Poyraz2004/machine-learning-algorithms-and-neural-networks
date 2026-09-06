import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline

# PROGRAM 1 - CLASSIFICATION


print("=== PROGRAM 1 ===")


X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_reg = LogisticRegression()
svm = SVC(probability=True)
rf = RandomForestClassifier()

log_reg.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)

voting = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('svm', svm),
        ('rf', rf)
    ],
    voting='soft'
)

voting.fit(X_train, y_train)

models = {
    "Logistic Regression": log_reg,
    "SVM": svm,
    "Random Forest": rf,
    "Voting Classifier": voting
}

for name, model in models.items():
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
    plt.title("Voting Classifier Decision Boundary")
    plt.show()

plot_decision_boundary(voting, X, y)



# PROGRAM 2 - REGRESSION

print("\n=== PROGRAM 2 ===")

def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y

datasets = ["dane1.txt", "dane2.txt"]

for file in datasets:
    print(f"\n--- {file} ---")

    X, y = load_data(file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)


    poly_model = make_pipeline(
        PolynomialFeatures(degree=5),
        LinearRegression()
    )
    poly_model.fit(X_train, y_train)


    y_pred_lin = lin_reg.predict(X_test)
    y_pred_poly = poly_model.predict(X_test)


    r2_lin = r2_score(y_test, y_pred_lin)
    r2_poly = r2_score(y_test, y_pred_poly)

    print(f"Linear R2: {r2_lin:.4f}")
    print(f"Polynomial R2: {r2_poly:.4f}")

    plt.scatter(X, y, s=10, label="Data")

    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    plt.plot(X_range, lin_reg.predict(X_range), label="Linear", linewidth=2)
    plt.plot(X_range, poly_model.predict(X_range), label="Polynomial", linewidth=2)

    plt.title(file)
    plt.legend()
    plt.show()