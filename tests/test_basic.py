""" test yastn.block """
import os
import pytest
import numpy as np
from IsingFF import IsingFFSolver, load_schedule


def path(str):
    return os.path.join(os.path.dirname(__file__), str)


def test_obc():
    fX, fZZ = load_schedule(path("qa_schedule_230429.txt"))
    solver = IsingFFSolver(N=64,
                       J=-1,
                       fX=fX,
                       fZZ=fZZ,
                       boundary='obc',
                       tau=7.0,
                       snapshots=[0, 0.5, 1])

    tol = 1e-4  # reference data from MPS
    assert pytest.approx(solver.measure_ZZnn(i=0, s=1), rel=tol) == 0.937845223
    assert pytest.approx(solver.measure_ZZnn(i=1, s=1), rel=tol) == 0.843519390
    assert pytest.approx(solver.measure_ZZnn(i=2, s=1), rel=tol) == 0.790716906
    assert pytest.approx(solver.measure_ZZnn(i=31, s=1), rel=tol) ==0.765074310
    assert pytest.approx(solver.measure_ZZnn(i=60, s=1), rel=tol) == 0.790716906
    assert pytest.approx(solver.measure_ZZnn(i=61, s=1), rel=tol) == 0.843519390
    assert pytest.approx(solver.measure_ZZnn(i=62, s=1), rel=tol) == 0.937845223

    assert pytest.approx(solver.measure_entropy(i1=0, i2=31, s=0.5), rel=tol) == 0.992476

    assert pytest.approx(solver.measure_XX(i1=30, s1=0.5, i2=40, s2=0.5), rel=tol) == 0.0317710189
    assert pytest.approx(solver.measure_ZZ(i1=30, s1=1.0, i2=32, s2=1.0), rel=tol) == 0.5392132741


if __name__ == "__main__":
    test_obc()