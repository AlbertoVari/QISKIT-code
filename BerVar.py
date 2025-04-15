from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

# Declare 4 input qubits + 1 output qubit
# q[0-3] = input, q[4] = output
# classical bits to store result
# Step 1: Initialize output qubit to |1⟩
circuit.x(qreg_q[4])
# Step 2: Apply Hadamard to all qubits
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.h(qreg_q[2])
circuit.h(qreg_q[3])
circuit.h(qreg_q[4])
circuit.barrier(qreg_q)
# Step 3: Oracle for a = 1101
# Apply CNOT from input qubit to output qubit where a_i = 1
circuit.cx(qreg_q[0], qreg_q[4])
# a₀ = 1
# skip q[1] since a₁ = 0
circuit.cx(qreg_q[2], qreg_q[4])
# a₂ = 1
circuit.cx(qreg_q[3], qreg_q[4])
# a₃ = 1
circuit.barrier(qreg_q)
# Step 4: Hadamard on input register again
circuit.h(qreg_q[0])
circuit.h(qreg_q[1])
circuit.h(qreg_q[2])
circuit.h(qreg_q[3])
circuit.barrier(qreg_q)
# Step 5: Measure input register
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.measure(qreg_q[2], creg_c[2])
circuit.measure(qreg_q[3], creg_c[3])