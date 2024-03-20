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


if __name__ == '__main__':
    unittest.main()
