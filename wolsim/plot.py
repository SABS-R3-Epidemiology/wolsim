"""Visualization and animation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors


class SimulationAnimation:

    def __init__(self, model):
        """
        Parameters
        ----------
        model : AgentBasedModel
            Model for which to generate animation
        """
        self.model = model

    def plot(self):
        """Plot wolbachia, compound concentration, and virus over time.
        """
        fig = plt.figure(figsize=(12, 5))

        ##### Panel 1: Wolbachia #####
        ax = fig.add_subplot(1, 3, 1)
        img = ax.imshow(self.model.wolbachia_grid, cmap='Greens', )
        legend = ax.figure.colorbar(img, ax=ax, shrink=0.5)

        ax.set_xticks(np.arange(self.model.n_x)[::5])
        ax.set_yticks(np.arange(self.model.n_y)[::5])

        ax.set_xticks(np.arange(-.5, self.model.n_x), minor=True)
        ax.set_yticks(np.arange(-.5, self.model.n_y), minor=True)

        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        ax.set_xlabel('X Grid')
        ax.set_ylabel('Y Grid')
        ax.set_title('Wolbachia present')

        ##### Panel 2: Compound concentration (animated) #####
        axc = fig.add_subplot(1, 3, 2)
        first_conc = self.model.concentration_history[0]

        highest_conc = np.max([np.max(a) for a in [self.model.concentration_history[i] for i in range(len(self.model.concentration_history))]])
        imgc = axc.imshow(first_conc, cmap='Blues', norm=matplotlib.colors.LogNorm(vmin=None, vmax=highest_conc))
        legend = axc.figure.colorbar(imgc, ax=axc, shrink=0.5)
        axc.set_xticks(np.arange(self.model.n_x)[::5])
        axc.set_yticks(np.arange(self.model.n_y)[::5])

        axc.set_xticks(np.arange(-.5, self.model.n_x), minor=True)
        axc.set_yticks(np.arange(-.5, self.model.n_y), minor=True)

        axc.grid(which='minor', color='k', linestyle='-', linewidth=1)

        axc.set_xlabel('X Grid')
        axc.set_ylabel('Y Grid')
        axc.set_title('Inhibitor concentration')

        ##### Panel 3: Virus (animated) #####
        ax = fig.add_subplot(1, 3, 3)

        first_config = self.model.virus_grid_history[0][0]
        highest_virus = np.max([np.max(a) for a in [self.model.virus_grid_history[i][0] for i in range(len(self.model.virus_grid_history))]])
        img = ax.imshow(first_config, cmap='Reds',  norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=highest_virus,))

        def init():
            img.set_data(first_config)
            imgc.set_data(first_conc)
            return img, imgc

        def animate(i):
            a = self.model.virus_grid_history[i][0]
            img.set_array(a)

            c = self.model.concentration_history[i]
            imgc.set_data(c)

            return img, imgc 

        ax.set_xticks(np.arange(self.model.n_x)[::5])
        ax.set_yticks(np.arange(self.model.n_y)[::5])

        ax.set_xticks(np.arange(-.5, self.model.n_x), minor=True)
        ax.set_yticks(np.arange(-.5, self.model.n_y), minor=True)

        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        ax.set_xlabel('X Grid')
        ax.set_ylabel('Y Grid')

        legend = ax.figure.colorbar(img, ax=ax, shrink=0.5)
        ax.set_title('Viral abundance')

        fig.set_tight_layout(True)

        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            init_func=init,
            frames=len(self.model.virus_grid_history),
            interval=30)
        
        return ani
    
    def plot_twovirus(self):
        """Plot wolbachia, and both types of virus.
        """
        fig = plt.figure(figsize=(12, 5))

        ##### Panel 1: wolbachia #####
        ax = fig.add_subplot(1, 3, 1)

        img = ax.imshow(self.model.wolbachia_grid, cmap='Greens', )
        legend = ax.figure.colorbar(img, ax=ax, shrink=0.5)

        ax.set_xticks(np.arange(self.model.n_x)[::5])
        ax.set_yticks(np.arange(self.model.n_y)[::5])

        ax.set_xticks(np.arange(-.5, self.model.n_x), minor=True)
        ax.set_yticks(np.arange(-.5, self.model.n_y), minor=True)

        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        ax.set_xlabel('X Grid')
        ax.set_ylabel('Y Grid')
        ax.set_title('Wolbachia present')

        ##### Panel 2: unmodified virus (animated) #####
        axc = fig.add_subplot(1, 3, 2)

        first_config_v0 = self.model.virus_grid_history[0][0]
        highest_virus = np.max([np.max(a) for a in [self.model.virus_grid_history[i][0] for i in range(len(self.model.virus_grid_history))]])
        imgc = axc.imshow(first_config_v0, cmap='Reds', norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=highest_virus+1))
        legend = axc.figure.colorbar(imgc, ax=axc, shrink=0.5)

        axc.set_xticks(np.arange(self.model.n_x)[::5])
        axc.set_yticks(np.arange(self.model.n_y)[::5])

        axc.set_xticks(np.arange(-.5, self.model.n_x), minor=True)
        axc.set_yticks(np.arange(-.5, self.model.n_y), minor=True)

        axc.grid(which='minor', color='k', linestyle='-', linewidth=1)

        axc.set_xlabel('X Grid')
        axc.set_ylabel('Y Grid')
        axc.set_title('Unmodified viral abundance')

        ##### Panel 3: modified virus (animated) #####
        ax = fig.add_subplot(1, 3, 3)
        first_config = self.model.virus_grid_history[0][0]
        highest_virus = np.max([np.max(a) for a in [self.model.virus_grid_history[i][1] for i in range(len(self.model.virus_grid_history))]])
        img = ax.imshow(first_config, cmap='Reds',  norm=matplotlib.colors.LogNorm(vmin=1e-1, vmax=highest_virus+1))

        def init():
            img.set_data(first_config)
            imgc.set_data(first_config_v0)

            return img, imgc

        def animate(i):
            a = self.model.virus_grid_history[i][1]
            img.set_array(a)

            c = self.model.virus_grid_history[i][0]
            imgc.set_data(c)

            return img, imgc 

        ax.set_xticks(np.arange(self.model.n_x)[::5])
        ax.set_yticks(np.arange(self.model.n_y)[::5])

        ax.set_xticks(np.arange(-.5, self.model.n_x), minor=True)
        ax.set_yticks(np.arange(-.5, self.model.n_y), minor=True)

        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

        ax.set_xlabel('X Grid')
        ax.set_ylabel('Y Grid')

        legend = ax.figure.colorbar(img, ax=ax, shrink=0.5)
        ax.set_title('Modified viral abundance')
        fig.set_tight_layout(True)

        ani = animation.FuncAnimation(
            fig=fig,
            func=animate,
            init_func=init,
            frames=len(self.model.virus_grid_history),
            interval=30)
        
        return ani
    

