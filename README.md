# QISKIT-code

The repository [QISKIT-code](https://github.com/AlbertoVari/QISKIT-code) contains several programs developed using Qiskit, each illustrating fundamental concepts in quantum computing. Here's a brief overview of each program:

1. **BB84-QKD**: This program demonstrates the BB84 quantum key distribution protocol, one of the first and most well-known quantum cryptography protocols. It showcases how two parties can securely share a cryptographic key using the principles of quantum mechanics.

2. **BlackHole-Evaporation**: Simulates the concept of black hole evaporation, possibly exploring Hawking radiation and the gradual loss of mass and energy from black holes over time. based preprint https://arxiv.org/pdf/2412.15180

3. **Entanglement**: Illustrates the phenomenon of quantum entanglement, where two or more qubits become linked such that the state of one instantaneously influences the state of the other, regardless of distance.

4. **Heisenberg.py**: Explores the Heisenberg uncertainty principle, highlighting the fundamental limits of simultaneously measuring certain pairs of physical properties, such as position and momentum.

5. **Interference**: Demonstrates quantum interference patterns resulting from the superposition of quantum states, a key aspect of phenomena like the double-slit experiment.

6. **ParticleCollision**: Simulates particle collision events, potentially modeling interactions at the quantum level similar to those observed in particle accelerators.

7. **QKD-sample.py**: Provides a sample implementation of a quantum key distribution protocol, emphasizing the security advantages of quantum communication methods.

8. **Superposition**: Demonstrates the principle of quantum superposition, where qubits can exist in multiple states simultaneously until measured.

9. **Teleport**: Implements quantum teleportation, a process by which the state of a qubit is transferred from one location to another without moving through the intervening space.

10. **WaveParticle**: Explores the wave-particle duality of quantum entities, showcasing how particles like electrons exhibit both wave-like and particle-like properties.

11. **grover.py**: Implements Grover's algorithm, a quantum search algorithm that provides a quadratic speedup for unstructured search problems compared to classical algorithms.

12. **quantum-qudit-crack.py**: Simulates a quantum brute-force password attack using Grover's algorithm, extended to qudits (d-dimensional quantum systems) instead of normal qubits. It operates in a hybrid quantum-classical way and also logs memory usage and execution time.

13. **Vqnn_Gradient_Attack.py** : Based on article "A Numerical Gradient Inversion Attack in Variational Quantum Neural-Networks" https://arxiv.org/pdf/2504.12806 the setting for running the code are

    Parameter  | Recommended Value | Why
    lr         | 0.005             | Smoother and stable descent
    h          | 0.0005            | Precise gradient estimation
    n_steps    | 500               | Allow time for convergence
    avg_window | 15                | Stronger low-pass filtering
    use_kalman | True              | Faster convergence

This Python script using PennyLane demonstrates numerical inversion of a VQNN input based solely on model gradients defining : 
a) A simple 2-qubit VQNN.
b) An attack that reconstructs hidden input data by matching gradients using a finite-difference method and adaptive filtering.
c) An optional Kalman filter refinement to speed up convergence.
--- Output:
Prints the true input, reconstructed input, and Mean Squared Error (MSE) between them — showing if the attack was successful.

Each program serves as a practical example of quantum computing principles, offering insights into both foundational concepts and advanced quantum algorithms. 

14. **QML-fraud-detection-demo.py** : The demo builds a hybrid quantum-classical classifier that embeds 2D input features into a quantum circuit using angle embedding, then uses a trainable variational circuit to classify fraud vs. non-fraud transactions based on measurement outcomes.  Install Requirements -> pip install pennylane scikit-learn matplotlib

15. **PQC-ML-KEM-Windows11.py** : This Python script demonstrates how to use the Windows CNG (Cryptography Next Generation) API via the bcrypt.dll library to: a) Open the ML-KEM algorithm provider. b) Generate a key pair. c) Export the public key.
because Post-Quantum Cryptography Comes to Windows Insiders and Linux https://techcommunity.microsoft.com/blog/microsoft-security-blog/post-quantum-cryptography-comes-to-windows-insiders-and-linux/4413803 using Cryptography API: Next Generation (CNG) https://learn.microsoft.com/en-us/windows/win32/seccng/cng-portal


## Related repositories

- Quantum code in QISKIT to run Grover's algorithm, also known as the quantum search algorithm, on IBM Q (https://github.com/AlbertoVari/QC-Grover)

- Post-Quantum Cryptography Basics (https://github.com/AlbertoVari/PQC-Learning-Module-WP1)

- Financial market prediction algorithms based also on QML (https://github.com/AlbertoVari/PredictionMeta4)

- Implementing Quantum Machine Learning in Qiskit for Genomic Sequence Classification (https://github.com/AlbertoVari/Deakin_Quantum-ML_genomic)

- Sample of NVIDIA SDK cuQuantum (https://github.com/AlbertoVari/cuQuantum/tree/main)

- Quantum machine Learning model running on NVIDIA Jetson Nano device connected to IBM Quantum services (https://github.com/AlbertoVari/SolidQML)

- Adder for two qubits with quantum logic gates running on IBM Quantum (https://github.com/AlbertoVari/Qadder)

- Paddle Quantum is a quantum computing framework developed by Baidu as an extension of the PaddlePaddle deep learning platform, designed for research and applications in quantum machine learning, quantum chemistry, and quantum optimization (https://github.com/AlbertoVari/Quantum)

- Quantum TSP contains code for an open source program solving the Travelling Salesman Problem with Quantum Computing (https://github.com/AlbertoVari/quantum_tsp)


