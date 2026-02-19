import numpy as np

class SORsolver:
    '''
    pace equation using deta^2 = 0 using Sussive over relaxation (SOR) method.
    Boundary condition is set to 0, and initial guess is set to 1.
    '''

    def __init__(self, N = 100, omega = 1.5, tol = 1e-6, max_iter = 10000):
        self.N = N
        self.omega = omega
        self.tol = tol
        self.max_iter = max_iter

        self.c = np.zeros((N, N)) # initial guess is set to 0, and boundary condition is set to 0
        self._apply_boundary_condition() 
    def _apply_boundary_condition(self):
        #Bottom boundary (j =0) 
        self.c[0, :] = 0
        #Top boundary (j = N-1) 
        self.c[-1, :] = 0
        #Left boundary (i = 0)
        self.c[:, 0] = 0
        #Right boundary (i = N-1)
        self.c[:, -1] = 0
    def solve(self, verbose = False):
        for it in range(self.max_iter):
            delta = 0
            for i in range(1, self.N-1):
                for j in range(1, self.N-1):

                    left = (i-1) % self.N
                    right = (i+1) % self.N
                    
                    old_value = self.c[i, j]
                    gs_update = 0.25 * (
                        self.c[left, j] + 
                        self.c[right, j] + 
                        self.c[i, j-1] + 
                        self.c[i, j+1]
                    )
                    new_value = (self.omega * gs_update) + ((1 - self.omega) * old_value)

                    self.c[i, j] = new_value
                    delta = max(delta, abs(new_value - old_value))
            if verbose:
                print(f"Iteration {it+1}, max change: {delta:.6e}")
            if delta < self.tol:
                if verbose:
                    print(f"Convergence achieved after {it+1} iterations.")
                break
        return self.c