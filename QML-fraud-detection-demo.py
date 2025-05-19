import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# ---- Step 1: Create synthetic data (2 features, binary labels) ----
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=2, random_state=42)
X = MinMaxScaler().fit_transform(X)  # Normalize features to [0,1]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---- Step 2: Define quantum circuit ----
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def angle_embedding(x, wires):
    for i in range(len(x)):
        qml.RY(np.pi * x[i], wires=wires[i])

def variational_circuit(weights, wires):
    qml.templates.StronglyEntanglingLayers(weights, wires=wires)

@qml.qnode(dev)
def quantum_classifier(x, weights):
    angle_embedding(x, wires=range(n_qubits))
    variational_circuit(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# ---- Step 3: Define prediction + cost function ----
def predict(X, weights):
    return [1 if quantum_classifier(x, weights) < 0 else 0 for x in X]

def cost(weights, X, y):
    preds = np.array([quantum_classifier(x, weights) for x in X])
    return np.mean((preds - (1 - 2*y))**2)

# ---- Step 4: Training ----
np.random.seed(0)
weights = np.random.randn(3, n_qubits, 3, requires_grad=True)  # Shape depends on layers/qubits

opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 50

for i in range(steps):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
    if i % 10 == 0:
        print(f"Step {i} - Cost: {cost(weights, X_train, y_train)}")

# ---- Step 5: Evaluation ----
y_pred = predict(X_test, weights)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
