from spinor_gpe.spinor import *
import cupy as cp

"""Module file that contains necessary functions for solving the spin-1 GPE using a symplectic method. These functions
use CuPy to wrap around CUDA for a fast and efficient evolution."""


def fourier_evo_1d(wfn: Wavefunction, k_grid: Grid, dt: float, q: float):
    """Solves the kinetic energy term in Fourier space.

    Parameters
    ----------
    wfn : Wavefunction
        The Wavefunction object.
    k_grid : Grid
        The k-space Grid object.
    dt : float
        Time step.
    q : float
        Quadratic Zeeman term.
    """

    wfn.psi_k[0] *= cp.exp(-0.25 * 1j * dt * (k_grid.squared + 2 * q))
    wfn.psi_k[1] *= cp.exp(-0.25 * 1j * dt * k_grid.squared)
    wfn.psi_k[2] *= cp.exp(-0.25 * 1j * dt * (k_grid.squared + 2 * q))


def _calc_spin_dens(wfn: Wavefunction, dt: float):
    """Calculates the spin vectors, cos & sin terms, and atomic density for
    use in interaction evolution.

    Parameters
    ----------
    wfn : Wavefunction
        The Wavefunction object.
    dt : float
        Time step.

    Returns
    ---------
    tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]
        Perpendicular spin vector, z-spin vector, cos term, sin term, density.
    """

    spin_perp = cp.sqrt(2.) * (cp.conj(wfn.psi[0]) * wfn.psi[1] + cp.conj(wfn.psi[1]) * wfn.psi[2])
    spin_z = cp.abs(wfn.psi[0]) ** 2 - cp.abs(wfn.psi[2]) ** 2
    F = cp.sqrt(cp.abs(spin_z) ** 2 + cp.abs(spin_perp) ** 2)  # Magnitude of spin vector

    cos_term = cp.cos(wfn.c2 * F * dt)
    sin_term = 1j * cp.sin(wfn.c2 * F * dt) / F
    sin_term = cp.nan_to_num(sin_term)  # Corrects division by 0

    density = cp.abs(wfn.psi[0]) ** 2 + cp.abs(wfn.psi[1]) ** 2 + cp.abs(wfn.psi[2]) ** 2

    return spin_perp, spin_z, cos_term, sin_term, density


def interaction_evo(wfn: Wavefunction, dt: float, p: float):
    """Computes the interaction flow

    Parameters
    ----------
    wfn : Wavefunction
        The Wavefunction object.
    dt : float
        Time step.
    p : float
        The linear Zeeman term.
    """

    F_perp, Fz, C, S, n = _calc_spin_dens(wfn, dt)

    new_wfn_plus = (C * wfn.psi[0] - S * (Fz * wfn.psi[0] + cp.conj(F_perp) / cp.sqrt(2) * wfn.psi[1])) \
                   * cp.exp(-1j * dt * (wfn.V - p + wfn.c0 * n))
    new_wfn_0 = (C * wfn.psi[1] - S / cp.sqrt(2) * (F_perp * wfn.psi[0] + cp.conj(F_perp) * wfn.psi[2])) \
                * cp.exp(-1j * dt * (wfn.V + wfn.c0 * n))
    new_wfn_minus = (C * wfn.psi[2] - S * (F_perp / cp.sqrt(2) * wfn.psi[1] - Fz * wfn.psi[2])) \
                    * cp.exp(-1j * dt * (wfn.V + p + wfn.c0 * n))

    wfn.update_wfn([new_wfn_plus, new_wfn_0, new_wfn_minus], k_space=False)


def forward_fft_1d(wfn: Wavefunction):
    """Performs a 1D fft on wavefunction components.

    Parameters
    ----------
    wfn : Wavefunction
        The Wavefunction object.
    """

    wfn.psi_k = [cp.fft.fft(psi) for psi in wfn.psi]


def backward_fft_1d(wfn: Wavefunction):
    """Performs an inverse 1D fft on wavefunction components.

    Parameters
    ----------
    wfn : Wavefunction
        The Wavefunction object.
    """

    wfn.psi = [cp.fft.ifft(psi_k) for psi_k in wfn.psi_k]
