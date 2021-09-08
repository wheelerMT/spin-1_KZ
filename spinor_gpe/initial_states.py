from abc import ABC, abstractmethod
from spinor_gpe.spinor import *


class InitialStateFactory:

    @staticmethod
    def set_initial_state(initial_state: str, grid: Grid, wfn: Wavefunction):

        if initial_state == 'polar':
            PolarInitialState.generate_initial_state(grid, wfn)


class InitialState(ABC):
    """Factory for choosing the initial state we require.
    Currently supported is fully polar or ferromagnetic.
    """

    @staticmethod
    @abstractmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction):
        """Generates the initial state given a grid and wavefunction object.

        Parameters
        ----------
        grid: Grid
            The Grid object associated with the wavefunction.
        wfn: Wavefunction
            The Wavefunction object.
        """
        pass


class PolarInitialState(InitialState):

    @staticmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction):
        if grid.dim == 1:
            # If 1D problem, generate 1D wfn with noise in outer components
            psiP1 = cp.zeros(grid.nx, dtype='complex64')  # (cp.random.normal(0, 0.5, grid.nx) + 1j * cp.random.normal(0, 0.5, grid.nx)) / cp.sqrt(grid.nx)
            psi0 = cp.ones(grid.nx, dtype='complex64')
            psiM1 = cp.zeros(grid.nx, dtype='complex64')  # (cp.random.normal(0, 0.5, grid.nx) + 1j * cp.random.normal(0, 0.5, grid.nx)) / cp.sqrt(grid.nx)
        else:
            # If 2D problem, generate 2D wfn
            psiP1 = cp.zeros((grid.nx, grid.ny), dtype='complex64')
            psi0 = cp.ones((grid.nx, grid.ny), dtype='complex64')
            psiM1 = cp.zeros((grid.nx, grid.ny), dtype='complex64')

        wfn.set_initial_state([psiP1, psi0, psiM1])
