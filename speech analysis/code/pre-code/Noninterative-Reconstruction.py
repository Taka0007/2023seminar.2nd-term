import numpy as np

def calculate_stft_phase_gradient(S, a, b, lambda_L, M):
    """
    Calculate the STFT phase gradient approximation.

    :param S: Magnitude of the STFT coefficients.
    :param a, b: Hop sizes in time and frequency.
    :param lambda_L: λL parameter.
    :param M: Number of frequency channels.
    :return: Scaled phase gradient matrix.
    """
    # Calculate logarithm of STFT magnitude
    slog = np.log(S)

    # Numerical differentiation along rows (time) and columns (frequency)
    Dt = np.diff(slog, axis=0, append=slog[-1:])
    Dω = np.diff(slog, axis=1, append=slog[:, -1:])

    # Calculate and scale the phase gradient
    φSC_ω = -lambda_L / (a * M) * Dt
    φSC_t = (a * M) / lambda_L * Dω + 2 * np.pi * a / M

    # Combine the components into a single matrix
    return np.stack([φSC_ω, φSC_t], axis=-1)


# Assume we have a STFT magnitude matrix S, and relevant parameters a, b, lambda_L, M
# S = np.array([[...]])  # STFT magnitude matrix
# a, b = 10, 10  # Example hop sizes
# lambda_L = 1  # Example λL parameter
# M = 100  # Example number of frequency channels

# phase_gradient = calculate_stft_phase_gradient(S, a, b, lambda_L, M)
