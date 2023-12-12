""" routines to numerically solve a quench in 1D transverse-field Ising model. """
import numpy as np
from functools import reduce, lru_cache
from typing import Callable, Sequence
# from pfapack import pfaffian

MLT = 1000000

def load_schedule(file : str) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Load schedule from file. Includes extra pi factors.

    Return
    ------
    fX, fZZ
    """
    # os.path.join(os.path.dirname(__file__), )
    schedule = np.loadtxt(file)
    fX = lambda s: np.pi * np.interp(s, schedule[:, 0], schedule[:, 1])
    fZZ = lambda s: np.pi * np.interp(s, schedule[:, 0], schedule[:, 2])
    return fX, fZZ


class IsingFFSolver:

    def __init__(self, N, J, fX, fZZ, boundary="obc", tau=1.0, snapshots=[0, 1], dt=1/64):
        r"""
        Simulate Ising model with Hamiltonian

        H(s) = sum_i (J_i fZZ(s) Z_i Z_{i+1} + fX(s) X_i)

        Quench schedule is specified by fX(s) and fZZ(s) (should contain all pi factors).
        The system is initialized in the ground state at s=0,
        and evolved in time following s(t) = t / tau.

        Parameters
        ----------
        N: int
            Chain length.

        J: Sequence[float] | float
            Nearet-neighbor couplings. If provided number of elements is smaller then N,
            they are periodically repeated to fill-in N coupings.

        fX: Callable[[float], float]
            s-dependent amplitude of transverse field.

        fZZ: Callable[[float], float]
            s-dependent amplitude of NN couplings.

        boundary: str
            Boundary conditions: 'obc' or 'pbc'.

        tau: float
            Total evolution time.

        snapshots: Sequence[float]
            Sorted list of numbers 0 <= s <= 1.
            Precalcuate evolution operators for specified valuess of s.
            To resolve bookkeepings, values are storred up to 6 digits.

        dt: float
            time-step of integrator used to calculate evolution operators.
        """

        if not isinstance(N, int) or N < 1:
            raise ValueError("N should be a positive int. ")
        if boundary not in ['obc', 'pbc']:
            raise ValueError("boundary should be 'pbc' for periodic or 'obc' for open boundary conditions. ")

        if not hasattr(J, '__iter__'):
            J = [J]
        J = J * np.ceil(N / len(J)).astype(int)
        J = J[:N]

        if not all(0 <= s <= 1 for s in snapshots):
            raise ValueError("All snapshots should be in 0 <= s <= 1. ")

        if not 0 < dt < 1:
            raise ValueError("dt should be in 0 <= dt <= 1. ")

        self.N = N
        self.boundary = boundary
        self.J = tuple(J)
        snapshots = set(np.rint(MLT * s).astype(int) for s in snapshots)
        snapshots.add(0)
        self.snapshots = tuple(sorted(snapshots))
        self.tau = tau
        self.dt = dt
        self.fX = fX
        self.fZZ = fZZ

        print("Diagonalizing initial Hamiltonian ...")
        if boundary == "obc":
            H = self.Hamiltonian_Ising(s=0, bnd=0)
            D0, O0, p = BdG_Hamiltonian(H)
            self.Eng0 = -sum(D0)
            self.O0 = O0
            self.p = p
        else:  # boundary == "pbc":
            Hp = self.Hamiltonian_Ising(s=0, bnd=-1)
            Hm = self.Hamiltonian_Ising(s=0, bnd=+1)
            Dp, Op, pp = BdG_Hamiltonian(Hp, parity=+1)
            Dm, Om, pm = BdG_Hamiltonian(Hm, parity=-1)
            if -sum(Dp) < -sum(Dm):
                self.O0 = Op
                self.p = pp
                self.Eng0 = -sum(Dp)
            else:
                self.O0 = Om
                self.p = pm
                self.Eng0 = -sum(Dm)
        self.C0 = C_from_O(self.O0)

        print("Calculating evolution operators ...")
        self.Us = []
        bnd = -self.p if boundary == "pbc" else 0
        for si, sf in zip((0,) + self.snapshots, self.snapshots):
            si, sf = si / MLT, sf / MLT
            self.Us.append(self.evolution_operator(sf, si, bnd=bnd))
        self.cmimj = lru_cache(maxsize=32)(self._cmimj)


    def Hamiltonian_Ising(self, s, bnd=0):
        """
        Hamiltonian H = J_i fZZ(s) Z_i Z_{i+1} + fX(s) X_i
        """

        HX = np.zeros((2 * self.N, 2 * self.N), dtype=np.float64)
        for n in range(self.N):  # 2 * HX / 1j
            HX[2 * n, 2 * n + 1] = 1
            HX[2 * n + 1, 2 * n] = -1

        HZZ = np.zeros((2 * self.N, 2 * self.N), dtype=np.float64)
        for n in range(self.N - 1):  # 2 * HZZ / 1j
            HZZ[2 * n + 1, 2 * n + 2] = self.J[n]
            HZZ[2 * n + 2, 2 * n + 1] = -self.J[n]

        n = self.N - 1
        HZZ[2 * n + 1, 0] = bnd * self.J[n]
        HZZ[0, 2 * n + 1] = -bnd * self.J[n]

        H = (1j / 2) * (self.fX(s) * HX + self.fZZ(s) * HZZ)
        return H


    def gate_X(self, s, dt):
        U = np.eye(2 * self.N, dtype=np.float64)
        x = 2 * self.fX(s) * dt
        cx = np.cos(x)
        sx = np.sin(x)
        for n in range(self.N):
            U[2*n+0, 2*n+0] = cx
            U[2*n+0, 2*n+1] = sx
            U[2*n+1, 2*n+0] = -sx
            U[2*n+1, 2*n+1] = cx

        return U


    def gate_ZZ(self, s, dt, bnd=0):
        U = np.eye(2 * self.N, dtype=np.float64)
        x = 2 * self.fZZ(s) * dt
        for n, Jn in enumerate(self.J[:-1]):
            cx = np.cos(Jn * x)
            sx = np.sin(Jn * x)
            U[2*n+1, 2*n+1] = cx
            U[2*n+1, 2*n+2] = sx
            U[2*n+2, 2*n+1] = -sx
            U[2*n+2, 2*n+2] = cx

        n = self.N - 1
        Jn = self.J[n]
        cx = np.cos(bnd * Jn * x)
        sx = np.sin(bnd * Jn * x)
        U[2*n+1, 2*n+1] = cx
        U[2*n+1, 0] = sx
        U[0, 2*n+1] = -sx
        U[0, 0] = cx

        return U


    def evolution_operator(self, sf, si, bnd=0):
        U = np.eye(2 * self.N)
        if abs(sf - si) < 1e-8:
            return U

        s2 = 0.41449077179437573714

        ds = self.dt / self.tau
        steps = int(np.ceil((sf - si) / ds))
        ds = (sf - si) / steps
        dt = ds * self.tau
        s = si
        for _ in range(steps):
            s = s + s2 * ds / 2
            UX2 = self.gate_X(s, s2 * dt / 2)
            UZZ = self.gate_ZZ(s, s2 * dt, bnd)
            U =  UX2 @ UZZ @ UX2 @ U
            s = s + s2 * ds / 2

            s = s + s2 * ds / 2
            UX2 = self.gate_X(s, s2 * dt / 2)
            UZZ = self.gate_ZZ(s, s2 * dt, bnd)
            U =  UX2 @ UZZ @ UX2 @ U
            s = s + s2 * ds / 2

            s = s + (1 - 4 * s2) * ds / 2
            UX2 = self.gate_X(s, (1 - 4 * s2) * dt / 2)
            UZZ = self.gate_ZZ(s, (1 - 4 * s2) * dt, bnd)
            U =  UX2 @ UZZ @ UX2 @ U
            s = s + (1 - 4 * s2) * ds / 2

            s = s + s2 * ds / 2
            UX2 = self.gate_X(s, s2 * dt / 2)
            UZZ = self.gate_ZZ(s, s2 * dt, bnd)
            U =  UX2 @ UZZ @ UX2 @ U
            s = s + s2 * ds / 2

            s = s + s2 * ds / 2
            UX2 = self.gate_X(s, s2 * dt / 2)
            UZZ = self.gate_ZZ(s, s2 * dt, bnd)
            U =  UX2 @ UZZ @ UX2 @ U
            s = s + s2 * ds / 2
        assert abs(s - sf) < 1e-10
        return U


    def cm(self, si, sj):
        """ Correlation matrix <a(si) a(sj)> of Majorana fermions. """
        mi = np.rint(si * MLT).astype(int)
        mj = np.rint(sj * MLT).astype(int)

        try:
            indi = self.snapshots.index(mi)
        except ValueError as ex:
            raise ValueError(f" initialized snapshots do not include si = {si}") from ex
        try:
            indj = self.snapshots.index(mj)
        except ValueError as ex:
            raise ValueError(f" initialized snapshots do not include sj = {sj}") from ex

        return self.cmimj(indi, indj)


    def _cmimj(self, indi, indj):
        Ui = reduce(np.matmul, self.Us[indi: None : -1])
        Uj = reduce(np.matmul, self.Us[indj: None : -1])
        return  Ui @ self.C0 @ Uj.T


    def measure_ZZnn(self, i, s) -> float:
        """
        Calculate equal-time nearest neighbour <Z_i(s) Z_{i+1}(s)>
        """
        if i < 0 or i > self.N-2:
            raise ValueError(f"Specified position outside of chain of length {self.N}.")

        C = self.cm(s, s)
        return -C[2*i+1, 2*i+2].imag


    def measure_X(self, i, s) -> float:
        """
        Calculate <X_i(s)>
        """
        if i < 0 or i > self.N-1:
            raise ValueError(f"Specified position outside of chain of length {self.N}.")
        C = self.cm(s, s)
        return -C[2*i, 2*i+1].imag


    def measure_ZZ(self, i1, s1, i2, s2) -> float:
        """
        Calculate abs(<Z_i1(s1) Z_i2(s2)>)
        """
        if any(i < 0 or i >= self.N for i in [i1, i2]):
            raise ValueError(f"Specified position outside of chain of length {self.N}.")

        if self.boundary == 'pbc' and any(s1 != s2):
            raise ValueError("For PBC non-equal-time ZZ correlation is currently not supported")

        rs =  [slice(0, 2*i + 1) for i in [i1, i2]]
        ss = [s1, s2]
        Cblock = [[self.cm(t1, t2)[r1, r2] for r2, t2 in zip(rs, ss)] for r1, t1 in zip(rs, ss)]
        tmp = np.block(Cblock)
        tmp = np.triu(tmp, 1)
        tmp = tmp - tmp.T
        return np.sqrt(abs(np.linalg.det(tmp)))   #  pfaffian(tmp) ** 2 = det(tmp)


    def measure_ZZZZ(self, i1, s1, i2, s2, i3, s3, i4, s4) -> float:
        """
        Calculate abs(<Z_i1(s1) Z_i2(s2) Z_i3(s3) Z_i4(s4)>)
        """
        if any(i < 0 or i >= self.N for i in [i1, i2, i3, i4]):
            raise ValueError(f"Specified position outside of chain of length {self.N}.")

        if self.boundary == 'pbc' and any(s != s1 for s in [s2, s3, s4]):
            raise ValueError("For PBC non-equal-time ZZZZ correlation is currently not supported")

        rs =  [slice(0, 2*i + 1) for i in [i1, i2, i3, i4]]
        ss = [s1, s2, s3, s4]
        Cblock = [[self.cm(t1, t2)[r1, r2] for r2, t2 in zip(rs, ss)] for r1, t1 in zip(rs, ss)]
        tmp = np.block(Cblock)
        tmp = np.triu(tmp, 1)
        tmp = tmp - tmp.T
        return np.sqrt(abs(np.linalg.det(tmp)))   #  pfaffian(tmp) ** 2 = det(tmp)


    def measure_XX(self, i1, s1, i2, s2) -> float:
        """
        Calculate abs(<X_i1(s1) X_i2(s2)>)
        """
        if any(i < 0 or i >= self.N for i in [i1, i2]):
            raise ValueError(f"Specified position outside of chain of length {self.N}.")

        rs =  [slice(2*i, 2*i + 2) for i in [i1, i2]]
        ss = [s1, s2]
        Cblock = [[self.cm(t1, t2)[r1, r2] for r2, t2 in zip(rs, ss)] for r1, t1 in zip(rs, ss)]
        tmp = np.block(Cblock)
        tmp = np.triu(tmp, 1)
        tmp = tmp - tmp.T
        return np.sqrt(abs(np.linalg.det(tmp)))   #  pfaffian(tmp) ** 2 = det(tmp)


    def measure_XXXX(self, i1, s1, i2, s2, i3, s3, i4, s4) -> float:
        """
        Calculate abs(<X_i1(s1) X_i2(s2) X_i3(s3) X_i4(s4)>)
        """
        if any(i < 0 or i >= self.N for i in [i1, i2, i3, i4]):
            raise ValueError(f"Specified position outside of chain of length {self.N}.")

        rs =  [slice(2*i, 2*i + 2) for i in [i1, i2, i3, i4]]
        ss = [s1, s2, s3, s4]
        Cblock = [[self.cm(t1, t2)[r1, r2] for r2, t2 in zip(rs, ss)] for r1, t1 in zip(rs, ss)]
        tmp = np.block(Cblock)
        tmp = np.triu(tmp, 1)
        tmp = tmp - tmp.T
        return np.sqrt(abs(np.linalg.det(tmp)))   #  pfaffian(tmp) ** 2 = det(tmp)


    def measure_entropy(self, i1, i2, s) -> float:
        """
        Calculate von Neuman entropy of a block [i1, i2) at time-snapshot s.

        Use base-2 log.

        If i1 >= i2, return 0.
        """
        if any(i < 0 or i > self.N for i in [i1, i2]):
            raise ValueError(f"Specified position outside of chain of length {self.N}.")
        if i2 <= i1:
            return 0.
        r = slice(2*i1, 2*i2)

        Cblock = self.cm(s, s)[r, r]
        eps, _ = np.linalg.eigh(Cblock)

        tol = 1e-13

        ind = (tol < eps) * (eps < 2 - tol)
        eps = eps[ind] / 2
        return -sum(np.log2(x) * x for x in eps)

def BdG_Hamiltonian(H, parity=None):
    """
    Diagonalize Hamiltonian matrix.
    Here assume no degeneracy in the spectrum, and no need to control the parity.
    """
    N = len(H) // 2
    D, V = np.linalg.eigh(H)
    V2 = V[:,  N : 2 * N]  # modes coresponding to positive energy
    D2 = 2 * D[N : 2 * N]  # positive energies
    O = np.zeros((2*N, 2*N))
    O[:, slice(1, 2 * N, 2)]  =   np.sqrt(2) * np.real(V2)
    O[:, slice(0, 2 * N, 2)]  = - np.sqrt(2) * np.imag(V2)

    Hd = np.zeros((2*N, 2*N))  # diagonal
    for n in range(N):  # 2 * Hd / 1j
        Hd[2 * n, 2 * n + 1] = D2[n]
        Hd[2 * n + 1, 2 * n] = -D2[n]
    Hd = (-1j / 2) * Hd

    assert np.max(abs(O @ O.T - np.eye(2 * N))) < 1e-13
    assert np.max(abs(Hd - O.T @ H @ O)) < 1e-13

    C = C_from_O(O)
    Canty = -np.imag(C[slice(0, None, 2), slice(1, None, 2)])
    p = np.linalg.det(Canty)
    if abs(abs(p) - 1) > 1e-8:
        raise ValueError("Cannot fix parity. Resolving ground-state degeneracy is currently not supported.")
    p = 1 if abs(p - 1) < 1e-8 else -1

    if parity is None:
        return D2, O, p

    if p != parity:
        D2[0] *= -1
        O[:, 1] *= -1

    return D2, O, parity


def C_from_O(O):
    N = len(O) // 2
    CD = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    for n in range(N):
        CD[2 * n, 2 * n] = 1
        CD[2 * n + 1, 2 * n + 1] = 1
        CD[2 * n, 2 * n + 1] = -1j
        CD[2 * n + 1, 2 * n] = 1j
    C = O @ CD @ O.T
    C = (C + C.T.conj()) / 2
    return C


# def Cparity(C):
#     N = len(C) // 2
#     Canty = (C - C.T) / 2
#     return (1j ** N) * pfaffian(Canty)

