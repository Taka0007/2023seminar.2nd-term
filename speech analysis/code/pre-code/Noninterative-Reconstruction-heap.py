import numpy as np
import heapq

def calculate_phase_gradient(S, a, b, lambda_L, M):
    """
    Calculate the phase gradient for STFT.

    :param S: STFT magnitude matrix.
    :param a, b: Hop sizes.
    :param lambda_L: Scaling parameter.
    :param M: Number of frequency bins.
    :return: Phase gradient matrix.
    """
    slog = np.log(S)
    Dt = np.diff(slog, axis=0, append=slog[-1:])
    Dω = np.diff(slog, axis=1, append=slog[:, -1:])

    φSC_ω = -lambda_L / (a * M) * Dt
    φSC_t = (a * M) / lambda_L * Dω + 2 * np.pi * a / M

    return np.stack([φSC_ω, φSC_t], axis=-1)

def integrate_phase_gradient(phase_gradient, tol=0.1):
    """
    Integrate the phase gradient using heap for path management.

    :param phase_gradient: Computed phase gradient.
    :param tol: Relative magnitude tolerance.
    :return: Reconstructed phase matrix.
    """
    magnitude = np.abs(phase_gradient)
    phase = np.zeros_like(magnitude)

    # Create a heap for managing integration paths
    heap = [(magnitude[i, j], i, j) for i in range(magnitude.shape[0]) for j in range(magnitude.shape[1])]
    heapq.heapify(heap)

    while heap:
        mag, i, j = heapq.heappop(heap)

        # Skip coefficients below tolerance
        if mag < tol:
            continue

        # Integrate phase gradient
        # This is a simplified version; in practice, you need to integrate along the paths efficiently.
        if i > 0:
            phase[i, j] += phase[i - 1, j] + phase_gradient[i - 1, j, 0]
        if j > 0:
            phase[i, j] += phase[i, j - 1] + phase_gradient[i, j - 1, 1]

    return phase

# Example usage
# S = np.array([[...]])  # STFT magnitude matrix
# a, b = 10, 10  # Example hop sizes
# lambda_L = 1  # Example lambda_L parameter
# M = 100  # Example number of frequency bins

# phase_gradient = calculate_phase_gradient(S, a, b, lambda_L, M)
# reconstructed_phase = integrate_phase_gradient(phase_gradient)
