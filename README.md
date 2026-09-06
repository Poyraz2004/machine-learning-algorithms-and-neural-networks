# Machine Learning & Deep Learning Implementations

A comprehensive collection of foundational machine learning algorithms, statistical methods, and deep learning architectures developed during the Methods of Knowledge Engineering (MIW) coursework at Polish-Japanese Academy of Information Technology.

---

## Repository Structure

| Directory | Core Topics & Architectures | Key Technologies & Libraries |
| :--- | :--- | :--- |
| **[01. Markov Chains](./01-markov-chains)** | Stochastic processes, state transition matrices, probability distributions | Python, Random, Matplotlib |
| **[02. Classification](./02-classification-and-logistic-regression)** | Decision boundaries, linear separation, mathematical loss modeling | Python, NumPy, Matplotlib |
| **[03. Scikit-Learn Benchmarks](./03-scikit-learn-benchmarks)** | Classification & regression pipelines, SVM, Random Forest, Voting Classifier | scikit-learn, NumPy, Matplotlib |
| **[04. Neural Networks (Scratch)](./04-neural-networks-from-scratch)** | Multi-layer feedforward network from scratch, forward/backpropagation, loss optimization | Python, NumPy, scikit-learn, Matplotlib |
| **[05. CNN (Computer Vision)](./05-convolutional-neural-networks-cnn)** | CIFAR-10 image classification, Conv2D, MaxPooling, EarlyStopping, ReduceLROnPlateau | TensorFlow, Keras, Seaborn, scikit-learn |
| **[06. RNN / LSTM](./06-recurrent-neural-networks-rnn)** | Time-series forecasting (BTC), LSTM recurrent layers, financial data pipeline | TensorFlow, Keras, yfinance, Pandas, scikit-learn |

---

## Key Highlights

* **Mathematical Rigor:** Implemented neural network forward propagation, backpropagation, and gradient updates from scratch using pure NumPy.
* **Deep Learning & Computer Vision:** Built and evaluated a Convolutional Neural Network (CNN) on CIFAR-10, utilizing dynamic learning rate scheduling (`ReduceLROnPlateau`), early stopping, and confusion matrix diagnostics.
* **Time-Series Forecasting:** Implemented an LSTM network using TensorFlow/Keras to predict financial asset prices via live market data fetched with `yfinance`.
* **Supervised Machine Learning:** Evaluated support vector machines (SVM), ensemble learning (Random Forest, Voting Classifiers), and feature pipelines using `scikit-learn`.
