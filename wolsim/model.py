"""Forwards simulation of the agent based model with wolbachia.
"""

import copy
import random
import numpy as np
from pde import FieldCollection, PDEBase, ScalarField, UnitGrid


class AgentBasedModel:
    """Agent based model of viral infection.
    
    This class supports two variants of the model.
    """
    def __init__(self, n_x, n_y, model='signalling'):
        """
        Parameters
        ----------
        n_x : int
            Number of grid cells in x direction
        n_y : int
            Number of grid cells in y direction
        model : {'signalling', 'genetic'}
            Which variant of the model to run
        """
        self.n_x = n_x
        self.n_y = n_y

        if model == 'signalling':
            self.num_virus = 1
            self.model = model
        elif model == 'genetic':
            self.num_virus = 2
            self.model = model
        else:
            raise ValueError('Model variant not recognized.')

        self.virus_grid = {}
        for i in range(self.num_virus):
            self.virus_grid[i] = np.zeros((n_x, n_y))

        self.conc_grid = np.zeros((n_x, n_y))
        self.wolbachia_grid = np.zeros((n_x, n_y))

        self.virus_diffuse_rate = {}
        self.virus_carrying_capacity = {}
        self.virus_growth_rate = {}

        self.virus_grid_history = []
        self.concentration_history = []

    def set_virus_parameters(self, diffuse_rate, growth_rate, carrying_capacity, virus_num=0):
        """Set the parameters controlling virus diffusion and growth.
        
        Parameters
        ----------
        diffuse_rate : float
            Rate of diffusion to neighboring cells
        growth_rate : float
            Rate of growth
        carrying_capacity : float
            Carrying capacity
        virus_num : int, optional (0)
            Which type of the virus in the genetic model (0=unmodified, 1=modified)
        """
        self.virus_diffuse_rate[virus_num] = diffuse_rate
        self.virus_growth_rate[virus_num] = growth_rate
        self.virus_carrying_capacity[virus_num] = carrying_capacity

    def initialize_wolbachia(self, p=0.3):
        """Initialize the wolbachia grid.

        Wolbachia are placed randomly, with probability p, in the 15x15 top left of the grid.

        Parameters
        ----------
        p : float
            probability of there being wolbachia in a cell
        """
        self.wolbachia_grid[:15, :15] = np.random.choice((0, 1), size=((15, 15)), p=[1-p, p])
        
        self.update_concentration()
        self.concentration_history.append(copy.deepcopy(self.conc_grid))

    def initialize_wolbachia_barrier(self):
        self.wolbachia_grid[7:13, :] = np.ones(self.wolbachia_grid[7:13, :].shape)

        self.update_concentration()
        self.concentration_history.append(copy.deepcopy(self.conc_grid))

    def initialize_virus_barrier(self):
        self.virus_grid[0][2, 10] = 1
        self.virus_grid_history.append(copy.deepcopy(self.virus_grid))

    def initialize_virus(self, virus_num=0):
        """Initialize the virus grid.
        
        Parameters
        ----------
        virus_num : int, optional (0)
            Which type of the virus in the genetic model (0=unmodified, 1=modified)
        """
        self.virus_grid[virus_num][15, 15] = 1
        self.virus_grid_history.append(copy.deepcopy(self.virus_grid))

    def update_concentration(self):
        """Solve a PDE to update concentration based on wolbachia locations.
        """
        class PDE(PDEBase):
            def __init__(self, outer, D=1.0, gamma=1.0, w=1.0, g=1.0, bc="auto_periodic_neumann"):
                super().__init__()
                self.D = D
                self.gamma = gamma
                self.w = w
                self.g = g
                self.bc = bc
                self.outer = outer

            def get_state(self, s):
                s.label = 'concentration'
                return FieldCollection([s,])

            def evolution_rate(self, state, t=0):
                s, = state
                eta = ScalarField(s.grid, data=self.outer.wolbachia_grid)
                v = ScalarField(s.grid, data=self.outer.virus_grid[0])
                ds_dt = self.D * s.laplace(self.bc) - self.gamma * s + self.w * eta - self.g * s * v
                return FieldCollection([ds_dt,])
       
        def create_pde(outer):
            return PDE(outer)

        eq = create_pde(self)
        grid = UnitGrid([self.n_x, self.n_y])
        s = ScalarField(grid, 0)
        state = eq.get_state(s)
        sol = eq.solve(state, t_range=10, dt=1e-1, tracker=None)
        self.conc_grid = sol.data[0, :, :]
    

    def _diffuse_virus(self, k, i, j):
        """Move virus to neighboring cells.

        Parameters
        ----------
        k : int
            Which virus variant (in genetic model)
        i : int
            x starting location
        j : int
            y starting location
        """
        v = self.virus_grid[k][i, j]

        for _ in range(int(v)):
            num_diffuses = np.random.poisson(self.virus_diffuse_rate[k])

            coords = (i, j)

            for _ in range(num_diffuses):
                dir = random.choice(['left', 'right', 'up', 'down'])

                self.virus_grid[k][*coords] -= 1

                if dir == 'left':
                    new_coords = (i-1, j)
                
                elif dir == 'right':
                    new_coords = (i+1, j)
                
                elif dir == 'up':
                    new_coords = (i, j+1)
                
                elif dir == 'down':
                    new_coords = (i, j-1)

                x_new, y_new = new_coords
                if 0 <= x_new < self.n_x and 0 <= y_new < self.n_y:
                    self.virus_grid[k][*new_coords] += 1
                    coords = new_coords
                else:
                    # It left the grid, stop moving it
                    break

    def _grow_virus(self, k, i, j):
        """Grow the virus within its current cell.
        
        Parameters
        ----------
        k : int
            Which virus variant (in genetic model)
        i : int
            x location
        j : int
            y location
        """
        v = self.virus_grid[k][i, j]

        total_virus = sum([self.virus_grid[k][i, j] for k in range(self.num_virus)])

        if self.model == 'signalling':
            growth = self.virus_growth_rate[k]*(np.exp(-20*self.conc_grid[i, j])) \
                    * v * (1 - total_virus / self.virus_carrying_capacity[k])
        
        else:
            growth = self.virus_growth_rate[k] \
                    * v * (1 - total_virus / self.virus_carrying_capacity[k])
                

        if growth > 0:
            if self.num_virus == 1:
                self.virus_grid[k][i, j] = v + np.random.poisson(growth)
            else:
                if self.wolbachia_grid[i, j] == 1:
                    self.virus_grid[1][i, j] = self.virus_grid[1][i, j] + np.random.poisson(growth)
                else:
                    self.virus_grid[k][i, j] = self.virus_grid[k][i, j] + np.random.poisson(growth)
        
        if growth < 0:
            self.virus_grid[k][i, j] = max(0, v - np.random.poisson(-growth))

    def update_virus(self, k):
        """One iteration of virus agent based simulation.
        """
        for i in range(self.n_x):
            for j in range(self.n_y):        
                v = self.virus_grid[k][i, j]
                if v > 0:
                    self._grow_virus(k, i, j)
                    self._diffuse_virus(k, i, j)

    def run_time_step(self):
        """One iteration of simulation.
        """
        for k in range(self.num_virus):
            self.update_virus(k)

        if self.num_virus == 1:
            self.update_concentration()
            self.concentration_history.append(copy.deepcopy(self.conc_grid))

        self.virus_grid_history.append(copy.deepcopy(self.virus_grid))
        
