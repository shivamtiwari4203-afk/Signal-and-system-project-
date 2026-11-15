# Signal-and-system-project-
"""
Project: Discrete Fourier Transform (DFT) and Discrete-Time Fourier Transform (DTFT)
Subject: Signals and Systems

Requirements:  pip install numpy matplotlib
Run:           python dft_dtft_project.py

All graphs are saved in a folder called "output_graphs" next to this file,
and are also shown on screen. Use these PNG files in your report.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved -> {path}")


# ----------------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------------
def dft(x):
    """DFT from the definition: X[k] = sum_n x[n] e^{-j 2 pi k n / N}."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)      # N x N DFT matrix
    return W @ x


def idft(X):
    """Inverse DFT: x[n] = (1/N) sum_k X[k] e^{+j 2 pi k n / N}."""
    X = np.asarray(X, dtype=complex)
    N = len(X)
    k = np.arange(N)
    n = k.reshape(-1, 1)
    W = np.exp(2j * np.pi * n * k / N)
    return (W @ X) / N


def dtft(x, w, n0=0):
    """DTFT of a finite-length signal x[n], n = n0, n0+1, ...
    X(e^{jw}) = sum_n x[n] e^{-j w n}, evaluated on the frequency grid w."""
    x = np.asarray(x, dtype=complex)
    n = np.arange(n0, n0 + len(x))
    return np.exp(-1j * np.outer(w, n)) @ x


# ----------------------------------------------------------------------------
# Experiment 1: DFT of a two-tone signal (bin-aligned, so peaks are clean)
# ----------------------------------------------------------------------------
def experiment_1():
    N = 32
    n = np.arange(N)
    x = np.cos(2 * np.pi * 4 * n / N) + 0.5 * np.cos(2 * np.pi * 10 * n / N)
    X = dft(x)

    # Check our own DFT against NumPy's FFT
    print("Experiment 1: max |DFT - FFT| =", np.max(np.abs(X - np.fft.fft(x))))

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    ax[0].stem(n, x, basefmt=" ")
    ax[0].set(title="Input signal x[n] = cos(2π·4n/32) + 0.5 cos(2π·10n/32)",
              xlabel="n", ylabel="x[n]")
    ax[1].stem(n, np.abs(X), basefmt=" ")
    ax[1].set(title="DFT magnitude |X[k]|", xlabel="k", ylabel="|X[k]|")
    ax[2].stem(n, np.angle(X), basefmt=" ")
    ax[2].set(title="DFT phase ∠X[k]", xlabel="k", ylabel="radians")
    fig.tight_layout()
    save(fig, "fig1_signal_and_dft.png")


# ----------------------------------------------------------------------------
# Experiment 2: DFT samples lie on the DTFT curve
# ----------------------------------------------------------------------------
def experiment_2():
    N = 16
    n = np.arange(N)
    x = np.cos(2 * np.pi * 0.125 * n) + 0.5 * np.sin(2 * np.pi * 0.3 * n)

    w = np.linspace(0, 2 * np.pi, 2000)
    X_dtft = dtft(x, w)
    X_dft = dft(x)
    wk = 2 * np.pi * np.arange(N) / N

    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax[0].plot(w, np.abs(X_dtft), label="DTFT (continuous in ω)")
    ax[0].stem(wk, np.abs(X_dft), linefmt="r-", markerfmt="ro", basefmt=" ",
               label="DFT samples (N = 16)")
    ax[0].set(title="DFT = samples of the DTFT at ω_k = 2πk/N",
              ylabel="Magnitude")
    ax[0].legend()
    ax[1].plot(w, np.angle(X_dtft), ".", markersize=2, label="DTFT phase")
    ax[1].plot(wk, np.angle(X_dft), "ro", label="DFT phase")
    ax[1].set(xlabel="ω (rad/sample)", ylabel="Phase (rad)")
    ax[1].legend()
    ax[1].set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax[1].set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    fig.tight_layout()
    save(fig, "fig2_dft_vs_dtft.png")


# ----------------------------------------------------------------------------
# Experiment 3: Zero-padding gives a denser sampling of the same DTFT
# ----------------------------------------------------------------------------
def experiment_3():
    N = 16
    n = np.arange(N)
    x = np.cos(2 * np.pi * 0.125 * n) + 0.5 * np.sin(2 * np.pi * 0.3 * n)
    w = np.linspace(0, 2 * np.pi, 2000)
    X_dtft = np.abs(dtft(x, w))

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for a, L in zip(ax, [16, 32, 128]):
        xp = np.concatenate([x, np.zeros(L - N)])
        Xp = np.abs(dft(xp))
        wk = 2 * np.pi * np.arange(L) / L
        a.plot(w, X_dtft, color="gray", alpha=0.6, label="DTFT")
        a.stem(wk, Xp, basefmt=" ", label=f"{L}-point DFT")
        a.set(ylabel="|X|", title=f"{N} samples zero-padded to {L} points")
        a.legend(loc="upper right")
    ax[-1].set_xlabel("ω (rad/sample)")
    ax[-1].set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax[-1].set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    fig.tight_layout()
    save(fig, "fig3_zero_padding.png")


# ----------------------------------------------------------------------------
# Experiment 4: DTFT of x[n] = a^n u[n]  (numerical vs. closed form)
# ----------------------------------------------------------------------------
def experiment_4():
    a = 0.8
    n = np.arange(0, 200)               # 200 terms is enough since 0.8^200 ~ 0
    x = a ** n
    w = np.linspace(-np.pi, np.pi, 1500)

    X_num = dtft(x, w)
    X_ana = 1.0 / (1.0 - a * np.exp(-1j * w))     # closed form
    print("Experiment 4: max |numerical - analytical| =",
          np.max(np.abs(X_num - X_ana)))

    fig, ax = plt.subplots(3, 1, figsize=(8, 8.5))
    ax[0].stem(n[:25], x[:25], basefmt=" ")
    ax[0].set(title="x[n] = (0.8)^n u[n]  (first 25 samples)", xlabel="n",
              ylabel="x[n]")
    ax[1].plot(w, np.abs(X_ana), label="Analytical  1/(1 − a e^{−jω})")
    ax[1].plot(w, np.abs(X_num), "--", label="Numerical sum")
    ax[1].set(title="DTFT magnitude", xlabel="ω (rad/sample)", ylabel="|X(e^{jω})|")
    ax[1].legend()
    ax[2].plot(w, np.angle(X_ana), label="Analytical")
    ax[2].plot(w, np.angle(X_num), "--", label="Numerical sum")
    ax[2].set(title="DTFT phase", xlabel="ω (rad/sample)", ylabel="radians")
    ax[2].legend()
    for a_ in ax[1:]:
        a_.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        a_.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
    fig.tight_layout()
    save(fig, "fig4_dtft_exponential.png")


# ----------------------------------------------------------------------------
# Experiment 5: DTFT of a rectangular pulse (Dirichlet kernel)
# ----------------------------------------------------------------------------
def experiment_5():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    w = np.linspace(-np.pi, np.pi, 4001)
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    for L in (5, 10, 20):
        x = np.ones(L)
        X = dtft(x, w)
        ax[0].plot(w, np.abs(X), label=f"L = {L}")
    ax[0].set(title="DTFT magnitude of a rectangular pulse of length L",
              xlabel="ω (rad/sample)", ylabel="|X(e^{jω})|")
    ax[0].legend()

    L = 10
    X_num = np.abs(dtft(np.ones(L), w))
    X_ana = np.abs(np.sin(w_safe * L / 2) / np.sin(w_safe / 2))
    ax[1].plot(w, X_ana, label="|sin(ωL/2) / sin(ω/2)|")
    ax[1].plot(w, X_num, "--", label="Numerical")
    ax[1].set(title="Rectangular pulse, L = 10: formula vs. numerical",
              xlabel="ω (rad/sample)")
    ax[1].legend()
    for a_ in ax:
        a_.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        a_.set_xticklabels(["−π", "−π/2", "0", "π/2", "π"])
    fig.tight_layout()
    save(fig, "fig5_rect_pulse_dtft.png")


# ----------------------------------------------------------------------------
# Experiment 6: Inverse DFT reconstruction
# ----------------------------------------------------------------------------
def experiment_6():
    rng = np.random.default_rng(1)
    N = 32
    n = np.arange(N)
    x = np.cos(2 * np.pi * 4 * n / N) + 0.5 * np.cos(2 * np.pi * 10 * n / N) \
        + 0.2 * rng.standard_normal(N)
    X = dft(x)
    x_rec = idft(X)
    err = np.abs(x - x_rec)
    print("Experiment 6: max reconstruction error =", err.max())

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].stem(n, x, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="Original x[n]")
    ax[0].plot(n, x_rec.real, "r--x", label="Reconstructed (IDFT)")
    ax[0].set(title="Original vs. IDFT-reconstructed signal", xlabel="n",
              ylabel="x[n]")
    ax[0].legend()
    ax[1].semilogy(n, err + 1e-18, "k.-")
    ax[1].set(title="Reconstruction error |x[n] − x̂[n]|", xlabel="n",
              ylabel="error (log scale)")
    fig.tight_layout()
    save(fig, "fig6_idft_reconstruction.png")


# ----------------------------------------------------------------------------
# Experiment 7: Speed of direct DFT (O(N^2)) vs FFT (O(N log N))
# ----------------------------------------------------------------------------
def experiment_7():
    sizes = [64, 128, 256, 512, 1024, 2048]
    t_dft, t_fft = [], []
    for N in sizes:
        x = np.random.randn(N)
        t0 = time.perf_counter(); dft(x); t_dft.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for _ in range(50):
            np.fft.fft(x)
        t_fft.append((time.perf_counter() - t0) / 50)
    print("\nExperiment 7: timing (seconds)")
    print(f"{'N':>6} {'direct DFT':>12} {'FFT':>12}")
    for N, a, b in zip(sizes, t_dft, t_fft):
        print(f"{N:>6} {a:>12.6f} {b:>12.8f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(sizes, t_dft, "o-", label="Direct DFT  O(N²)")
    ax.loglog(sizes, t_fft, "s-", label="FFT  O(N log N)")
    ax.set(title="Computation time: DFT vs. FFT", xlabel="N (number of samples)",
           ylabel="time (seconds)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig7_dft_vs_fft_time.png")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()
    experiment_7()
    print("\nAll experiments finished. Graphs are in:", OUT_DIR)
    plt.show()
