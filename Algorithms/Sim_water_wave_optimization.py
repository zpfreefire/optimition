from Algorithms.algorithm import Algorithm
import numpy as np
import random


class SimWWO(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.pop('alpha', 1.0026)
        self.Kmax = kwargs.pop('Kmax', 6)
        self.lamb = kwargs.pop('lamb', 0.5)
        self.para = kwargs.pop('para', 0.001)
        # self.beta = kwargs.pop('beat', 0.001)
        self.WaveLen = []

    def initial_position(self):
        Wave_position = np.array(np.zeros((self.population, self.func.dimension)))
        Func_Value = []
        for i in range(0, self.population):
            for j in range(0, self.func.dimension):
                Wave_position[i, j] = random.uniform(self.func.min_values[j], self.func.max_values[j])
            self.WaveLen.append(self.lamb)
            Func_Value.append(self.target_function(Wave_position[i]))
        Func_Value = np.array(Func_Value)
        Wave_position = np.c_[Wave_position, Func_Value]
        return Wave_position

    def Propagation(self, popu, index):
        Popu = []
        for i in range(self.func.dimension):
            temp = popu[i] + random.uniform(-1, 1) * self.WaveLen[index] * (
                    self.func.max_values[i] - self.func.min_values[i])
            while temp < self.func.min_values[i] or temp > self.func.max_values[i]:
                temp = popu[i] + random.uniform(-1, 1) * self.WaveLen[index] * (
                        self.func.max_values[i] - self.func.min_values[i])
            Popu.append(temp)
        return Popu

    def Breaking(self, popu, beta):
        Popu = []
        for i in range(self.func.dimension):
            temp = popu[i] + np.random.normal(0, 1) * beta * (self.func.max_values[i] - self.func.min_values[i])
            Popu.append(temp)
        return Popu

    def getMinFitness(self, population):
        minFitness = self.target_function(population[0][0:population.shape[1] - 1])
        for i in range(self.population):
            if (self.target_function(population[i][0:population.shape[1] - 1]) < minFitness):
                minFitness = self.target_function(population[i][0:population.shape[1] - 1])
        return minFitness

    def getMaxFitness(self, popu):
        maxFitness = self.target_function(popu[0][0:popu.shape[1] - 1])
        for i in range(self.population):
            if (self.target_function(popu[i][0:popu.shape[1] - 1]) > maxFitness):
                maxFitness = self.target_function(popu[i][0:popu.shape[1] - 1])
        return maxFitness

    def getBest(self, popu):
        minFitness = self.target_function(popu[0][0:popu.shape[1] - 1])
        flag = 0
        for i in range(self.population):
            if (self.target_function(popu[i][0:popu.shape[1] - 1]) < minFitness):
                minFitness = self.target_function(popu[i][0:popu.shape[1] - 1])
                flag = i
        BestWave = popu[flag][0:popu.shape[1] - 1]
        return BestWave

    def Update_Popu(self, popu):
        for i in range(popu.shape[0]):
            popu[i][popu.shape[1]-1] = self.target_function(popu[i][0:popu.shape[1] - 1])
        return popu

    def run(self):
        count = 0
        Wave_position = self.initial_position()
        BestValue = self.getBest(Wave_position)
        best_ind = self.target_function(BestValue)
        while not self.stop_condition(count, best_ind):
            print("Iteration = ", count, " of ", self.iterations, " f(x) = ", best_ind)

            for i in range(self.population):
                Temp = self.Propagation(Wave_position[i], i)
                BestValue = self.getBest(Wave_position)
                if self.target_function(Temp) < self.target_function(Wave_position[i]):
                    if self.target_function(Temp) < self.target_function(BestValue):
                        beta = 0.25 - 0.249 * count / self.iterations
                        BestValue = self.Breaking(Wave_position[i], beta)
                    Wave_position[i][0:Wave_position.shape[1] - 1] = Temp
            best_ind = self.target_function(BestValue)
            self.current_solution.append(best_ind)
            count += 1
            Wave_position = self.Update_Popu(Wave_position)
            Wave_position = Wave_position[np.lexsort(Wave_position.T)]
            self.population = 50 - int(44 * count / self.iterations)
            Wave_position = Wave_position[:self.population][:]
            min=self.getMinFitness(Wave_position)
            max=self.getMaxFitness(Wave_position)
            for i in range(self.population):
                self.WaveLen[i] = self.WaveLen[i] * self.alpha ** (
                        -(self.target_function(Wave_position[i][0:Wave_position.shape[1] - 1]) - min + self.para)
                        / (max - min + self.para))
            self.current_solution.append(best_ind)
        print("最优坐标：", BestValue)
        Best = []
        for i in range(self.func.dimension):
            Best.append(BestValue[i])
        Best.append(best_ind)
        self.best_solution = Best
        return Best


if __name__ == "__main__":
    from benchmark.Benchmark import *

    cs = SimWWO(func=Schaffer2())
    cs.run()
