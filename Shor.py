from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from fractions import Fraction
from math import gcd
import matplotlib.pyplot as plt

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def qft_dagger(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            circuit.cp(-3.14159 / float(2 ** (j - m)), m, j)
        circuit.h(j)


from qiskit.circuit.library import UnitaryGate
import numpy as np

def create_classical_modmult_gate(a, N, n_qubits):
    """Returns a Gate that applies modular multiplication by a mod N"""
    size = 2 ** n_qubits
    U = np.identity(size)

    # Create a classical permutation unitary for mod multiplication
    for x in range(size):
        if x < N:
            new_x = (a * x) % N
            U[[x, new_x]] = U[[new_x, x]]  # Swap the rows for a mod mapping

    return UnitaryGate(U, label=f"×{a} mod {N}")


def controlled_modular_multiplier(qc, a, N, control, target_qubits):
    modmult = create_classical_modmult_gate(a, N, len(target_qubits))
    modmult_gate = modmult.control(1)
    qc.append(modmult_gate, [control] + list(target_qubits))  # ✅ FIXED


def modexp_by_2_mod_9(qc, control_qubits, target_qubits):
    # 2^(2^i) mod 9 for i = 0,1,2
    a_powers = [2, 4, 7]  # 2^1 mod 9 = 2, 2^2 = 4, 2^4 = 7 mod 9
    for i, a_i in enumerate(a_powers):
        controlled_modular_multiplier(qc, a_i, 9, control_qubits[i], target_qubits)

n_count = 4
def shors_order_finding_real_modexp(a, N, n_count):
    n_target = 4
    qc = QuantumCircuit(n_count + n_target, n_count)

    # Hadamard gates on counting register
    for q in range(n_count):
        qc.h(q)

    # Target register set to |1⟩
    qc.x(n_count)

    # Controlled modular exponentiation
    modexp_by_2_mod_9(qc, control_qubits=range(n_count), target_qubits=range(n_count, n_count + n_target))

    # Inverse QFT
    qft_dagger(qc, n_count)

    # Measurement
    for i in range(n_count):
        qc.measure(i, i)

    return qc

def get_order_from_counts(counts, a, N, n_count):
    print("\nMeasured results (with shots, phase, and estimated r):")
    for result_bin, shots in sorted(counts.items(), key=lambda x: -x[1]):
        result_dec = int(result_bin, 2)
        phase = result_dec / 2 ** n_count
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        print(f"→ {result_bin} ({shots} shots) → phase: {phase}, r: {r}")

        # Try to validate this r
        if r % 2 == 0 and pow(a, r, N) == 1:
            plus = pow(a, r // 2) + 1
            minus = pow(a, r // 2) - 1
            f1, f2 = gcd(plus, N), gcd(minus, N)
            if 1 < f1 < N and 1 < f2 < N:
                print(f"\n🎉 Valid non-trivial order found: r = {r}")
                print(f"Factors of {N}: {f1}, {f2}")
                return r

    print("\n⚠️ No valid order found. Try again or change base.")
    return None



def run_shor_real_modexp_demo(a=4, N=15):
    qc = shors_order_finding_real_modexp(a, N, n_count)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # Make sure to assign the figure
    fig = plt.figure(figsize=(8, 5))
    plot_histogram(counts, ax=fig.gca())
    plt.tight_layout()
    plt.savefig("shor_histogram.png")
    print("✔ Histogram saved as shor_histogram.png")

#    plot_histogram(counts)
#    plt.show()

    r = get_order_from_counts(counts, a, N, n_count)
    if r is not None and r % 2 == 0 and pow(a, r, N) == 1:
       plus = pow(a, r // 2) + 1
       minus = pow(a, r // 2) - 1
       f1, f2 = gcd(plus, N), gcd(minus, N)
       print(f"Factors of {N}: {f1}, {f2}")
    else:
       print("No valid r found or failed to compute factors.")

if __name__ == "__main__":
    run_shor_real_modexp_demo()
