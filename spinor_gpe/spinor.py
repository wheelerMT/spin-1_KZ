import cupy as cp


class Grid:
    """Generates a grid object for use with the wavefunction. Contains the properties
    of the grid as well as the 2D meshgrids themselves.
    """

    def __init__(self, nx: int, dx: float, ny: int = None, dy: float = None):
        """Instantiate a Grid object.
        Automatically generates grids using parameters provided. If no
        y parameters are provided, assumes 1D system.

        Parameters
        ----------
        nx : int
            Number of x grid points.
        dx : float
            Grid spacing in x-direction.
        ny : int
            Number of y grid points.
        dy : float
            Grid spacing in y-direction.
        """

        if ny is None:  # If 1D problem
            self.nx = nx
            self.dx = dx
            self.len_x = nx * dx

            # Generate 2D meshgrids:
            self.X = cp.arange(-nx // 2, nx // 2) * dx

            self.squared = self.X ** 2

            self.dim = 1  # Dimensionality of system

        else:  # 2D Problem
            self.nx, self.ny = nx, ny
            self.dx, self.dy = dx, dy
            self.len_x, self.len_y = nx * dx, ny * dy

            # Generate 2D meshgrids:
            self.X, self.Y = cp.meshgrid(cp.arange(-nx // 2, nx // 2) * dx, cp.arange(-ny // 2, ny // 2) * dy)

            self.squared = self.X ** 2 + self.Y ** 2

            self.dim = 2   # Dimensionality of system

    def fftshift(self):
        """Performs FFT shift on grids, depending on dimensionality.
        """

        if self.dim == 1:
            self.X = cp.fft.fftshift(self.X)
        elif self.dim == 2:
            self.X = cp.fft.fftshift(self.X)
            self.Y = cp.fft.fftshift(self.Y)


class Wavefunction:

    def __init__(self, c0: float, c2: float):

        self.psi = []
        self.psi_k = []
        self.c0 = c0
        self.c2 = c2
        self.V = 0

    def set_initial_state(self, wfn):
        self.psi = wfn

        # Check dimensionality and then do appropriate FFT
        if wfn[0].ndim == 1:
            self.psi_k = [cp.fft.fft(psi) for psi in wfn]
        elif wfn[0].ndim == 2:
            self.psi_k = [cp.fft.fft2(psi) for psi in wfn]

    def update_wfn(self, wfn, k_space: bool):
        if k_space:
            self.psi_k = wfn
        else:
            self.psi = wfn
