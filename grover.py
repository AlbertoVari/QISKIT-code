import os
import tempfile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.circuit.library.phase_oracle import PhaseOracle
import subprocess
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import numpy as np

# https://qiskit-community.github.io/qiskit-algorithms/tutorials/07_grover_examples.html

input_3sat_instance = '''
c example DIMACS-CNF 3-SAT
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
'''

fp = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
fp.write(input_3sat_instance)
file_name = fp.name
fp.close()
oracle = None
try:
    oracle = PhaseOracle.from_dimacs_file(file_name)
except ImportError as ex:
    print(ex)
finally:
    os.remove(file_name)

from qiskit_algorithms import AmplificationProblem

problem = None
if oracle is not None:
    problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)

from qiskit_algorithms import Grover
from qiskit.primitives import Sampler

grover = Grover(sampler=Sampler())
result = None
if problem is not None:
    result = grover.amplify(problem)
    print(result.assignment)

from qiskit.visualization import plot_histogram

if result is not None:
    hysto = result.circuit_results[0]
    plot_histogram(hysto,filename='hysto.png')

    # Define the source and destination paths
    source_path = "/home/italygourmet_co/quantum/hysto.png"

    destination_path = "/var/www/html/quantum/hysto.png"

    subprocess.call(["sudo", "cp", source_path, destination_path])
