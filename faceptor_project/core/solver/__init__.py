from .solver import Solver
from .post_solver import PostSolver

def solver_entry(C):
    return globals()[C.config['common']['solver']['type']](C)