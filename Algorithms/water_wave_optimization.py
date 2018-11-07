from Algorithms.algorithm import Algorithm
import numpy as np
import random


class WWO(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hmax = kwargs.pop('hmax', 6)
        self.alpha = kwargs.pop('alpha', 1.0026)
        #self.beta = kwargs.pop('beta', 0.001)
        self.lamb = kwargs.pop('lamb', 0.5)
        self.para = kwargs.pop('para', 0.0001)
        self.WaveLen = []
        self.WaveHeight = []

    def initial_position(self):
        Wave_position = np.zeros((self.population, self.func.dimension))
        for i in range(0, self.population):
            for j in range(0, self.func.dimension):
                Wave_position[i, j] = random.uniform(self.func.min_values[j], self.func.max_values[j])
            self.WaveLen.append(self.lamb)
            self.WaveHeight.append(self.hmax)
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

    def Refraction(self, population, popu, index):
        Popu = []
        temp = self.getBest(population)
        for i in range(self.func.dimension):
            flag = np.random.normal((popu[i] + temp[i]) / 2, np.abs(temp[i] - popu[i]) / 2)
            Popu.append(flag)
        self.WaveHeight[index] = self.hmax
        if self.target_function(Popu) == 0:
            pass
        else:
            self.WaveLen[index] = self.WaveLen[index] * self.target_function(popu) / self.target_function(Popu)
        return Popu

    def Breaking(self, popu,beta):
        Popu = []
        for i in range(self.func.dimension):
            temp = popu[i] + np.random.normal(0, 1) * beta * (self.func.max_values[i] - self.func.min_values[i])
            Popu.append(temp)
        return Popu

    def getMinFitness(self, population):
        minFitness = self.target_function(population[0])
        for i in range(self.population):
            if (self.target_function(population[i]) < minFitness):
                minFitness = self.target_function(population[i])
        return minFitness

    def getMaxFitness(self, popu):
        maxFitness = self.target_function(popu[0])
        for i in range(self.population):
            if (self.target_function(popu[i]) > maxFitness):
                maxFitness = self.target_function(popu[i])
        return maxFitness

    def getBest(self, popu):
        minFitness = self.target_function(popu[0])
        flag = 0
        for i in range(self.population):
            if (self.target_function(popu[i]) < minFitness):
                minFitness = self.target_function(popu[i])
                flag = i
        BestWave = popu[flag]
        return BestWave

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
                        beta=0.25-0.249*count/self.iterations
                        BestValue = self.Breaking(Temp,beta)
                    Wave_position[i] = Temp
                else:
                    self.WaveHeight[i] -= 1
                    if self.WaveHeight[i] == 0:
                        Wave_position[i] = self.Refraction(Wave_position, Wave_position[i], i)
            best_ind = self.target_function(BestValue)
            min=self.getMinFitness(Wave_position)
            max=self.getMaxFitness(Wave_position)
            for i in range(self.population):
                self.WaveLen[i] = self.WaveLen[i] * self.alpha ** (
                        -(self.target_function(Wave_position[i]) - min + self.para)
                        / (max - min + self.para))
            self.current_solution.append(best_ind)
            count += 1
        self.best_solution = best_ind
        return best_ind


