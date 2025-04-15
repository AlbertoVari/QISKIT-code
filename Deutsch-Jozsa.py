from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def deutsch_jozsa_balanced():
    n = 3  # number of input qubits
    qc = QuantumCircuit(n + 1, n)

    # Step 1: Prepare |1> in the last qubit
    qc.x(n)
    qc.h(n)

    # Step 2: Apply H to input qubits
    for i in range(n):
        qc.h(i)

    # Step 3: Oracle for f(x) = x1 (balanced)
    qc.cx(1, n)  # control on x[1], target is ancillary qubit

    # Step 4: Apply H again to input qubits
    for i in range(n):
        qc.h(i)

    # Step 5: Measure input qubits
    qc.measure(range(n), range(n))

    # Simulate
    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    # Plot
    fig = plt.figure(figsize=(8, 5))
    plot_histogram(counts, ax=fig.gca())
    plt.title("Deutsch–Jozsa Algorithm (Balanced Function f(x) = x₁)")
    plt.tight_layout()
    plt.savefig("deutsch_jozsa_histogram.png")
    print("✔ Histogram saved as 'deutsch_jozsa_histogram.png'")
    print("Counts:", counts)

if __name__ == "__main__":
    deutsch_jozsa_balanced()
