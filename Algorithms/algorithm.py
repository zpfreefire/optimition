import abc
from benchmark.Benchmark import *
import logging
import csv
import datetime
import os
import random

logging.basicConfig(format='%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class Algorithm(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self.func = kwargs.pop('func', Ackley())
        self.population = kwargs.pop('population', 50)
        self.iterations = kwargs.pop('iterations', 200)
        self.precision = kwargs.pop('precision', 0.0001)
        self.eval_counter = 0
        self.best_solution = []
        self.current_solution = []


    @abc.abstractmethod
    def initial_position(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    # @staticmethod
    # def RouletteWheelSelection(self, weights):
    #     accumulation = np.cumsum(weights)
    #     p = random.random() * accumulation[-1]
    #     chosen_index = -1
    #     for index in range(len(accumulation)):
    #         if (accumulation[index] > p):
    #             chosen_index = index
    #             break
    #     choice = chosen_index
    #     return choice

    def target_function(self, position):
        self.eval_counter += 1
        return self.func.eval(position)

    def stop_condition(self, i, optimum_now):
        if i >= self.iterations:
            return True
        if abs(optimum_now - self.func.get_optimum()[-1]) <= self.precision:
            return True
        return False

    def best_output(self):
        logging.info("coordinate: %s,values: %s" % (str(self.best_solution[0]), str(self.best_solution[1])))


    def save(self):
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = "../statistics/%s/%s" % (self.__class__.__name__, self.func.__class__.__name__)
        directory = os.path.exists(path)
        if not directory:
            os.makedirs(path)

        filename = ("%s/%s.csv" % (path, time))
        print(filename)
        f = open(filename, "w", newline="")
        writer = csv.writer(f)
        for row in self.current_solution:
            writer.writerow(row)
        f.close()

