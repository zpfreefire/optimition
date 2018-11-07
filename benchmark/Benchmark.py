import numpy as np
from benchmark.Base_benchmark import Benchmarks


class Ackley(Benchmarks):
    def __init__(self, min_values=[-32.768] * 2, max_values=[32.768] * 2, dimension=2):
        super(Ackley, self).__init__(min_values, max_values, dimension)

    def get_optimum(self):
        return [0] * self.dimension, 0

    @staticmethod
    def eval(array, a=20, b=0.2, c=2 * np.pi):
        d = len(array)

        sum1 = 0
        sum2 = 0
        for i in range(d):
            xi = array[i]
            sum1 = sum1 + xi ** 2
            sum2 = sum2 + np.cos(c * xi)
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)

        fitness = term1 + term2 + a + np.exp(1)

        return fitness


class Bukin6(Benchmarks):
    def __init__(self, min_values=[-15, -3], max_values=[-5, 3],dimension=2):
        super(Bukin6, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [-10, 1], 0

    @staticmethod
    def eval(array):
        x1 = array[0]
        x2 = array[1]
        term1 = 100 * np.sqrt(abs(x2 - 0.01 * x1 ** 2))
        term2 = 0.01 * abs(x1 + 10)

        fitness = term1 + term2
        return fitness

class Camel3(Benchmarks):
    def __init__(self, min_values=[-5, -5], max_values=[5, 5],dimension=2):
        super(Camel3, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [0, 0], 0

    @staticmethod
    def eval(array):
        term1 = 2 * array[0] ** 2
        term2 = -1.05 * array[0] ** 4
        term3 = array[0] ** 6 / 6
        term4 = array[0] * array[1]
        term5 = array[1] ** 2

        fitness = term1 + term2 + term3 + term4 + term5
        return fitness

class Camel6(Benchmarks):
    def __init__(self, min_values=[-3, -2], max_values=[3, 2],dimension=2):
        super(Camel6, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [0.0898, -0.7126], -1.0316284229280819

    @staticmethod
    def eval(array):
        term1 = (4 - 2.1 * array[0] ** 2 + (array[0] ** 4) / 3) * array[0] ** 2
        term2 = array[0] * array[1]
        term3 = (-4 + 4 * array[1] ** 2) * array[1] ** 2

        fitness = term1 + term2 + term3
        return fitness

class Crossit(Benchmarks):
    def __init__(self, min_values=[-10, -10], max_values=[10, 10],dimension=2):
        super(Crossit, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [1.3491, 1.3491], -2.0626118504479614

    @staticmethod
    def eval(array):
        term1 = np.sin(array[0]) * np.sin(array[1])
        term2 = np.exp(abs(100 - np.sqrt(array[0] ** 2 + array[1] ** 2) / np.pi))
        fitness = -0.0001 * (abs(term1 * term2) + 1) ** 0.1
        return fitness

class Easom(Benchmarks):
    def __init__(self, min_values=[-100, -100], max_values=[100, 100],dimension=2):
        super(Easom, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [np.pi, np.pi], -1

    @staticmethod
    def eval(array):
        term1 = -np.cos(array[0]) * np.cos(array[1])
        term2 = np.exp(-(array[0] - np.pi) ** 2 - (array[1] - np.pi) ** 2)
        fitness = term1 * term2
        return fitness

class Eggholder(Benchmarks):
    def __init__(self, min_values=[-512, -512], max_values=[512, 512],dimension=2):
        super(Eggholder, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [512, 404.2319], -959.6406627106155

    @staticmethod
    def eval(array):
        fitness = - (array[1] + 47) * np.sin(np.sqrt(abs(array[1] + (array[0] / 2) + 47))) - array[0] * np.sin(
            np.sqrt(abs(array[0] - (array[1] + 47))))
        return fitness

class Griewank(Benchmarks):
    def __init__(self, min_values=[-600, -600], max_values=[600, 600],dimension=2):
        super(Griewank, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [0.0] * self.dimension, 0.0

    @staticmethod
    def eval(array):
        d = len(array)
        sum = 0
        prod = 1

        for i in range(d):
            xi = array[i]
            sum += xi ** 2 / 4000
            prod *= np.cos(xi / np.sqrt(i + 1))
        fitness = sum - prod + 1
        return fitness

class Holdertable(Benchmarks):
    def __init__(self, min_values=[-10, -10], max_values=[10, 10],dimension=2):
        super(Holdertable, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [8.05502,9.66459], -19.208502567767606

    @staticmethod
    def eval(array):
        term1 = np.sin(array[0]) * np.cos(array[1])
        term2 = np.exp(abs(1 - np.sqrt(array[0] ** 2 + array[1] ** 2) / np.pi))
        fitness = -abs(term1 * term2)
        return fitness

class Levy13(Benchmarks):
    def __init__(self, min_values=[-10, -10], max_values=[10, 10],dimension=2):
        super(Levy13, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [1,1], 1.3497838043956716e-31

    @staticmethod
    def eval(array):
        term1 = (np.sin(3 * np.pi * array[0]) ** 2)
        term2 = (array[0] - 1) ** 2 * (1 + (np.sin(3 * np.pi * array[1])) ** 2)
        term3 = (array[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * array[1])) ** 2)

        fitness = term1 + term2 + term3
        return fitness

class Michalewicz(Benchmarks):
    def __init__(self, min_values=[0, 0], max_values=[np.pi, np.pi],dimension=2):
        super(Michalewicz, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [2.20,1.57], -1.801140718473825

    @staticmethod
    def eval(array):
        sum = 0
        m = 10
        for (i, x) in enumerate(array, start=1):
            sum = sum + np.sin(x) * np.sin((i * (x ** 2)) / np.pi) ** (2 * m)
        fitness = -sum
        return fitness

class Rastrigin(Benchmarks):
    def __init__(self, min_values=[-5.12, -5.12], max_values=[5.12, 5.12],dimension=2):
        super(Rastrigin, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [0.0] * self.dimension, 0.0

    @staticmethod
    def eval(array):
        sum = 0
        for x in array:
            sum = sum + x ** 2 - 10 * np.cos(2 * np.pi * x)
        fitness = 10.0 * len(array) + sum
        return fitness

class Levy(Benchmarks):
    def __init__(self, min_values=[-10, -10], max_values=[10, 10], dimension=2):
        super(Levy, self).__init__(min_values, max_values, dimension)

    def get_optimum(self):
        return [1, 1], 0

    @staticmethod
    def eval(array):
        d = len(array)
        w = []
        for i in range(d):
            w.append(1 + (array[i] - 1) / 4)

        term1 = (np.sin(np.pi * w[0])) ** 2
        term3 = (w[d - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[d - 1])) ** 2)

        sum = 0
        for i in range(d - 1):
            wi = w[i]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            sum += new

        fitness = term1 + sum + term3
        return fitness

class Rosenbrock(Benchmarks):
    def __init__(self, min_values=[-5, -5], max_values=[10, 10],dimension=2):
        super(Rosenbrock, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [1] * self.dimension, 0

    @staticmethod
    def eval(array):
        d = len(array)
        sum = 0
        for i in range(d - 1):
            xi = array[i]
            xnext = array[i + 1]
            new = 100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2
            sum += new
        fitness = sum
        return fitness

class Schaffer2(Benchmarks):
    def __init__(self, min_values=[-100, -100], max_values=[100, 100],dimension=2):
        super(Schaffer2, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [0,0], 0.0

    @staticmethod
    def eval(array):
        term1 = (np.sin(array[0] ** 2 - array[1] ** 2)) ** 2 - 0.5
        term2 = (1 + 0.001 * (array[0] ** 2 + array[1] ** 2)) ** 2

        fitness = 0.5 + term1 / term2
        return fitness

class Schwefel(Benchmarks):
    def __init__(self, min_values=[-500, -500], max_values=[500, 500],dimension=2):
        super(Schwefel, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [420.9687] * self.dimension, 2.545567497236334e-05

    @staticmethod
    def eval(array):
        sum = 0
        for x in array:
            sum = sum + x * np.sin(np.sqrt(np.abs(x)))
        fitness = 418.9829 * len(array) - sum
        return fitness

class Shubert(Benchmarks):
    def __init__(self, min_values=[-5.12, -5.12], max_values=[5.12, 5.12],dimension=2):
        super(Shubert, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [] , -186.7309

    @staticmethod
    def eval(array):
        sum1 = 0
        sum2 = 0
        for i in range(1,6):
            new1 = i * np.cos((i + 1) * array[0] + i)
            new2 = i * np.cos((i + 1) * array[1] + i)
            sum1 += new1
            sum2 += new2
        fitness = sum1 * sum2
        return fitness

class Sphere(Benchmarks):
    def __init__(self, min_values=[-5.12, -5.12], max_values=[5.12, 5.12],dimension=2):
        super(Sphere, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [0.0]*self.dimension , 0.0

    @staticmethod
    def eval(array):
        fitness = 0
        for i in range(len(array)):
            fitness = fitness + array[i] ** 2
        return fitness

class Stybtang(Benchmarks):
    def __init__(self, min_values=[-5, -5], max_values=[5, 5],dimension=2):
        super(Stybtang, self).__init__(min_values, max_values,dimension)

    def get_optimum(self):
        return [-2.903534,-2.903534] , -78.3323314075428

    @staticmethod
    def eval(array):
        d = len(array)
        sum = 0
        for i in range(d):
            xi = array[i]
            new = xi ** 4 - 16 * xi ** 2 + 5 * xi
            sum += new

        fitness = sum / 2

        return fitness