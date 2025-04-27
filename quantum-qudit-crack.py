"""
Quantum Brute-force Password Cracking with Qudits and Grover's Algorithm + Logging

This script simulates a quantum brute-force attack using Grover's algorithm
in a hybrid quantum-classical setting. It now includes logging of RAM usage and timing.

Why qudits?
- Qubits are 2-dimensional (|0>, |1>), representing binary systems.
- Qudits generalize qubits to d dimensions (e.g., d=12 for charset 'abcdef123456').
- This allows encoding of larger alphabets directly in fewer quantum units.

Requirements:
- pip install qutip tqdm matplotlib psutil
"""

import numpy as np
from qutip import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from hashlib import sha512
import time
import psutil
import os

# ------------------------ Configuration ------------------------
charset = '012345abcd'
password_length = 4
expected_password = '1234'
salt = b'AF\t\xb1\xceZ\xbb\x85'
iterations = 195765
expected_key = None

d = len(charset)

# Derive expected_key by simulating classical hash derivation
key = expected_password.encode() + salt
for _ in range(iterations):
    key = sha512(key).digest()
expected_key = key[:32]

# ------------------------ Mapping ------------------------
char_to_index = {ch: i for i, ch in enumerate(charset)}
index_to_char = {i: ch for i, ch in enumerate(charset)}
target_indices = [char_to_index[ch] for ch in expected_password]

print(f"[INFO] Charset: {charset} (d={d})")
print(f"[INFO] Target password: {expected_password} -> indices {target_indices}")

# ------------------------ Initial State ------------------------
print("[INFO] Preparing equal superposition state...")
basis_states = [basis(d, i) for i in range(d)]
superposition = sum(basis_states) / np.sqrt(d)
initial_state = tensor([superposition] * password_length)

# ------------------------ Oracle ------------------------
def oracle(state):
    target_state = tensor([basis(d, i) for i in target_indices])
    projector = target_state * target_state.dag()
    identity = tensor([qeye(d) for _ in range(password_length)])
    return (identity - 2 * projector) * state

# ------------------------ Diffusion ------------------------
def diffusion_operator():
    uniform_state = tensor([sum(basis_states) / np.sqrt(d) for _ in range(password_length)])
    projector = uniform_state * uniform_state.dag()
    identity = tensor([qeye(d) for _ in range(password_length)])
    return 2 * projector - identity

# ------------------------ Grover Iterations ------------------------
N = d ** password_length
# k = int(np.floor(np.pi / 4 * np.sqrt(N)))
# k = min(10, int(np.floor(np.pi / 4 * np.sqrt(N))))  # Cap for testing
k = int(np.floor(np.pi / 4 * np.sqrt(d ** password_length)))

print(f"[INFO] Total states: {N}, Grover iterations required: {k}")
state = initial_state
D = diffusion_operator()

# Logging start
start_time = time.time()
mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(f"[LOG] Memory before Grover: {mem_before:.2f} MB")
print("[INFO] Running Grover iterations...")

for _ in tqdm(range(k), desc="Grover"):
    state = oracle(state)
    state = D * state

# Logging end
end_time = time.time()
mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(f"[LOG] Memory after Grover: {mem_after:.2f} MB")
print(f"[LOG] Time taken: {end_time - start_time:.2f} seconds")

# ------------------------ Measurement ------------------------
probabilities = np.abs(state.full()) ** 2
probabilities = probabilities.flatten()

max_index = np.argmax(probabilities)
guess_indices = np.unravel_index(max_index, [d] * password_length)
guessed_password = ''.join(index_to_char[i] for i in guess_indices)

# ------------------------ Classical Verification ------------------------
print(f"\n[RESULT] Password guess from Grover: {guessed_password}")
guess_key = guessed_password.encode() + salt
for _ in range(iterations):
    guess_key = sha512(guess_key).digest()

if guess_key[:32] == expected_key:
    print(f"[✅] Password verified successfully via classical hash.")
else:
    print(f"[❌] Hash verification failed. Quantum guess may be incorrect.")

print(f"[RESULT] Probability: {probabilities[max_index]:.6f}")

# ------------------------ Plot ------------------------
labels, values = [], []
for i, prob in enumerate(probabilities):
    if prob > 0.001:
        indices = np.unravel_index(i, [d] * password_length)
        label = ''.join(index_to_char[j] for j in indices)
        labels.append(label)
        values.append(prob)

plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.xlabel("Password Guess")
plt.ylabel("Probability")
plt.title("Probability Distribution After Grover Iterations")
plt.tight_layout()
plt.show()
