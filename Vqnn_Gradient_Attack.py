import pennylane as qml
import numpy as np
import torch
import matplotlib.pyplot as plt

# (Page 2-3) Set up device for a 2-qubit VQNN
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# (Page 2-3) Define VQNN: Feature map (RX) + Ansatz (RY + CNOT)
def VQNN(x, theta):
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)
    for i in range(n_qubits):
        qml.RY(theta[i], wires=i)
    qml.CNOT(wires=[0, 1])

# (Page 3) Define Observable (PauliZ expectation)
obs = qml.PauliZ(0) @ qml.PauliZ(1)

# (Page 3) Define Cost function (expectation value)
def cost_fn(x, theta):
    @qml.qnode(dev)
    def circuit(x, theta):
        VQNN(x, theta)
        return qml.expval(obs)
    return circuit(x, theta)

# (Page 5) Gradient computation with parameter shift rule
param_shift_grad = qml.grad(cost_fn, argnum=1)

# (Page 6) Kalman filter update for convergence acceleration
def kalman_update(x_pred, x_meas, P_pred, R=1e-3, Q=1e-5):
    K = P_pred / (P_pred + R)  # Kalman Gain
    x_upd = x_pred + K * (x_meas - x_pred)
    P_upd = (1 - K) * P_pred + Q
    return x_upd, P_upd

# (Page 5-7) Enhanced Gradient inversion attack with adaptive smoothing and retries
def gradient_inversion_attack_v2(g_true, theta, x_guess, lr=0.01, h=1e-4, n_steps=500, init_avg_window=15, use_kalman=True):
    x_var = torch.tensor(x_guess, requires_grad=False, dtype=torch.float32)
    P = torch.ones_like(x_var) * 0.1  # Kalman initial uncertainty
    avg_window = init_avg_window

    prev_loss = None

    for step in range(n_steps):
        grad_approx = []
        for j in range(len(x_var)):
            losses = []
            for n in range(1, avg_window + 1):
                delta = n * h
                x_plus = x_var.clone()
                x_minus = x_var.clone()
                x_plus[j] += delta
                x_minus[j] -= delta

                g_plus = param_shift_grad(x_plus.detach().numpy(), theta)
                g_minus = param_shift_grad(x_minus.detach().numpy(), theta)

                loss_plus = np.mean((np.array(g_plus) - np.array(g_true)) ** 2)
                loss_minus = np.mean((np.array(g_minus) - np.array(g_true)) ** 2)

                numerical_grad = (loss_plus - loss_minus) / (2 * delta)
                losses.append(numerical_grad)

            grad_approx.append(np.mean(losses))

        x_pred = x_var - lr * torch.tensor(grad_approx)

        if use_kalman and prev_loss is not None:
            for j in range(len(x_var)):
                x_pred[j], P[j] = kalman_update(x_pred[j], x_var[j], P[j])

        x_var = x_pred

        pred_grad = param_shift_grad(x_var.detach().numpy(), theta)
        loss_now = np.mean((np.array(pred_grad) - np.array(g_true)) ** 2)

        if prev_loss is not None:
            if loss_now > prev_loss:
                avg_window = max(1, avg_window // 2)  # Adaptively shrink window size
                lr = lr * 0.5  # Reduce learning rate if stuck

        prev_loss = loss_now

        if loss_now < 1e-6:
            print(f"Converged at step {step}, loss: {loss_now}")
            break

    return x_var.detach().numpy(), loss_now

# (Page 7-9) Multi-retry attack strategy with tracking
def gradient_inversion_attack_auto(g_true, theta, retries=10, lr=0.01, h=1e-4, n_steps=500, init_avg_window=15, use_kalman=True):
    best_loss = float('inf')
    best_x = None
    losses = []

    for attempt in range(retries):
        x_guess = np.random.uniform(0, np.pi, size=n_qubits)
        x_reconstructed, loss = gradient_inversion_attack_v2(g_true, theta, x_guess, lr, h, n_steps, init_avg_window, use_kalman)
        print(f"Attempt {attempt}: MSE Loss = {loss}")
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_x = x_reconstructed

    return best_x, best_loss, losses

# Example usage
np.random.seed(42)
true_x = np.random.uniform(0, np.pi, size=n_qubits)
theta = np.random.uniform(0, np.pi, size=n_qubits)

# Calculate true gradient to simulate training gradient
g_true = param_shift_grad(true_x, theta)

# Launch enhanced multi-retry inversion attack
reconstructed_x, best_mse, losses_over_retries = gradient_inversion_attack_auto(g_true, theta)

# Print final results
print("\nTrue x:", true_x)
print("Reconstructed x:", reconstructed_x)
print("Best MSE:", best_mse)

# (New) Publication-style plot: MSE vs retries
plt.figure(figsize=(8, 5))
plt.plot(range(len(losses_over_retries)), losses_over_retries, marker='o')
plt.yscale('log')
plt.xlabel('Retry Attempt')
plt.ylabel('MSE (log scale)')
plt.title('Gradient Inversion Attack: MSE vs Retry Attempt')
plt.grid(True)
plt.show()
