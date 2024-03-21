"""Test the code in the module model.py.
"""

import numpy as np
import unittest
from unittest.mock import patch
import wolsim


class TestModel(unittest.TestCase):

    def test_init(self):
        model = wolsim.AgentBasedModel(10, 10, 'signalling')
        self.assertEqual(model.num_virus, 1)
        self.assertEqual(model.model, 'signalling')

        model = wolsim.AgentBasedModel(10, 10, 'genetic')
        self.assertEqual(model.num_virus, 2)
        self.assertEqual(model.model, 'genetic')

        with self.assertRaises(ValueError):
            wolsim.AgentBasedModel(10, 10, 'a')

    def test_set_virus_parameters(self):
        model = wolsim.AgentBasedModel(10, 10, 'genetic')
        model.set_virus_parameters(1.5, 1.6, 1.7)

        self.assertEqual(model.virus_diffuse_rate[0], 1.5)
        self.assertEqual(model.virus_growth_rate[0], 1.6)
        self.assertEqual(model.virus_carrying_capacity[0], 1.7)

        model.set_virus_parameters(2.5, 2.6, 2.7, 1)

        self.assertEqual(model.virus_diffuse_rate[1], 2.5)
        self.assertEqual(model.virus_growth_rate[1], 2.6)
        self.assertEqual(model.virus_carrying_capacity[1], 2.7)    

    def test_initialize_wolbachia(self):
        model = wolsim.AgentBasedModel(20, 20, 'genetic')
        model.initialize_wolbachia(0.5)

        self.assertTrue(set(np.unique(model.wolbachia_grid)) == {0, 1})

    def test_initialize_virus(self):
        model = wolsim.AgentBasedModel(20, 20, 'genetic')
        model.initialize_virus(0)

        self.assertTrue(np.min(model.virus_grid[0] >= 0))

    def test_update_concentration(self):
        model = wolsim.AgentBasedModel(20, 20, 'genetic')
        model.initialize_wolbachia(0.5)
        model.update_concentration()
        
        self.assertTrue(np.min(model.conc_grid) >= 0)

    def test_update_virus(self):
        model = wolsim.AgentBasedModel(20, 20, 'genetic')
        model.set_virus_parameters(1.5, 1.6, 1.7)
        model.initialize_wolbachia()
        model.initialize_virus()
        model.update_virus(0)

        self.assertTrue(np.min(model.virus_grid[0] >= 0))
        self.assertTrue(set(np.unique(model.wolbachia_grid)).issubset(set(range(1000))))

    def test_run_time_step(self):
        model = wolsim.AgentBasedModel(20, 20, 'genetic')
        model.set_virus_parameters(1.5, 1.6, 1.7)
        model.initialize_wolbachia()
        model.initialize_virus()

        model.run_time_step()
        self.assertTrue(isinstance(model.virus_grid_history[-1][0], np.ndarray))

        model = wolsim.AgentBasedModel(20, 20, 'signalling')
        model.set_virus_parameters(1.5, 1.6, 1.7)
        model.initialize_wolbachia()
        model.initialize_virus()

        model.run_time_step()
        self.assertTrue(isinstance(model.virus_grid_history[-1][0], np.ndarray))    
        self.assertTrue(isinstance(model.concentration_history[-1], np.ndarray))    


if __name__ == '__main__':
    unittest.main()
