"""Test the code in the module plot.py.
"""

import numpy as np
import unittest
from unittest.mock import patch
import wolsim


class TestModel(unittest.TestCase):
    def test_init(self):
        model = wolsim.AgentBasedModel(20, 20, 'signalling')
        plotter = wolsim.SimulationAnimation(model)

        self.assertTrue(plotter.model == model)

    def test_plot(self):
        model = wolsim.AgentBasedModel(20, 20, 'signalling')
        model.set_virus_parameters(1.5, 1.6, 1.7)
        model.initialize_wolbachia()
        model.initialize_virus()
        model.run_time_step()
        
        plotter = wolsim.SimulationAnimation(model)
        plotter.plot()

    def test_plot_twovirus(self):
        model = wolsim.AgentBasedModel(20, 20, 'genetic')
        model.set_virus_parameters(1.5, 1.6, 1.7)
        model.set_virus_parameters(1.5, 1.6, 1.7, 1)
        model.initialize_wolbachia()
        model.initialize_virus()
        model.run_time_step()
        
        plotter = wolsim.SimulationAnimation(model)
        plotter.plot_twovirus()



if __name__ == '__main__':
    unittest.main()
