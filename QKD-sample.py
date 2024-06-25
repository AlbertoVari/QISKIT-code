from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# Numero di qubit / bit
n = 10

# Passo 1: Alice genera i bit e le basi casualmente
alice_bits = np.random.randint(2, size=n)
alice_bases = np.random.randint(2, size=n)

# Passo 2: Creazione del circuito quantistico per inviare i qubit
def prepare_qubits(bits, bases):
    qc = QuantumCircuit(n)
    for i in range(n):
        if bits[i] == 1:
            qc.x(i)
        if bases[i] == 1:
            qc.h(i)
    return qc

# Alice prepara i qubit
qc = prepare_qubits(alice_bits, alice_bases)

# Passo 3: Bob sceglie le basi per misurare i qubit
bob_bases = np.random.randint(2, size=n)
def measure_qubits(qc, bases):
    for i in range(n):
        if bases[i] == 1:
            qc.h(i)
        qc.measure_all()

# Bob misura i qubit
measure_qubits(qc, bob_bases)

# Passo 4: Esecuzione del circuito
backend = Aer.get_backend('qasm_simulator')
results = execute(qc, backend, shots=1).result()
counts = results.get_counts()

# Estraiamo i risultati della misurazione di Bob
bob_results = list(counts.keys())[0][::-1]  # Invertiamo la stringa perché Qiskit inverte l'ordine dei bit

# Passo 5: Confronto delle basi e estrazione della chiave
alice_key = []
bob_key = []
for i in range(n):
    if alice_bases[i] == bob_bases[i]:
        alice_key.append(alice_bits[i])
        bob_key.append(int(bob_results[i]))

print("Alice's Key:", alice_key)
print("Bob's Key:  ", bob_key)
