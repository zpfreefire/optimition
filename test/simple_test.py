import unittest
from Algorithms.flower_pollination_algorithm import FPA

class Test(unittest.TestCase):
    def test(self):
        Fpa=FPA()
        Fpa.run()
        self.assertAlmostEqual(Fpa.best_solution[-1],Fpa.func.get_optimum()[1],3)