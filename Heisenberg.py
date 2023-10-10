from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import AerSimulator

# Crea un circuito quantistico
qc = QuantumCircuit(1)

# Applica la porta Hadamard
qc.h(0)

# Misura nella base sigma_z
qc.measure_all()
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result_z = simulator.run(compiled_circuit).result()
counts_z = result_z.get_counts()

# Rimuovi la misura e applica una seconda porta Hadamard per misurare nella base sigma_x
qc.remove_final_measurements()
qc.h(0)
qc.measure_all()
compiled_circuit_x = transpile(qc, simulator)
result_x = simulator.run(compiled_circuit_x).result()
counts_x = result_x.get_counts()

# Visualizza i risultati
print("Risultati nella base sigma_z:", counts_z)
print("Risultati nella base sigma_x:", counts_x)

