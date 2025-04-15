from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

# Declare 2 qubits and 2 classical bits
# Step 1: Put all qubits in superposition
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
# Step 2: Oracle to mark |10⟩
circuit.x(qreg_q[0])
# Flip qubit 0 (because it's 0 in |10⟩)
circuit.h(qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[1])
circuit.x(qreg_q[0])
# Unflip
# Step 3: Diffusion operator (inversion about the mean)
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.h(qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[1])
circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
# Step 4: Measurement
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])