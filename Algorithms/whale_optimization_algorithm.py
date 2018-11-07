from Algorithms.algorithm import Algorithm
import numpy as np
import random


class WOA(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b = kwargs.pop('b', 1)

    def initial_position(self):
        Whale_position = np.zeros((self.population, self.func.dimension))
        for i in range(0, self.population):
            for j in range(0, self.func.dimension):
                Whale_position[i, j] = random.uniform(self.func.min_values[j], self.func.max_values[j])
            #Whale_position[i, -1] = self.target_function(Whale_position[i, 0:Whale_position.shape[1] - 1])
        return Whale_position

    def TwoPlotDistance(self, popu, BestWhale):
        distance = 0.0
        for i in range(self.func.dimension):
            distance += (popu[i] - BestWhale[i]) ** 2
        return np.sqrt(distance)

    def EncirclePrey(self, popu, BestWhale, A, C):
        Popu = []
        for i in range(self.func.dimension):
            D = np.abs(C * BestWhale[i] - popu[i])
            X = BestWhale[i] - A * D
            Popu.append(X)
        return Popu

    def SpiralUpdate(self, popu, BestWhale, L):
        # Popu=np.zeros((1,self.func.dimension))
        Popu = []
        Length = self.TwoPlotDistance(popu, BestWhale)
        for i in range(self.func.dimension):
            X = Length * np.exp(self.b * L) * np.cos(2 * np.pi * L) + BestWhale[i]
            Popu.append(X)
        return Popu

    def SearchPrey(self, popu1, popu2, A, C):
        # Popu=np.zeros((1,self.func.dimension))
        Popu = []
        for i in range(self.func.dimension):
            D = np.abs(C * popu2[i] - popu1[i])
            X = popu2[i] - A * D
            Popu.append(X)
        return Popu

    def getBest(self, popu):
        BestWahle = []
        minFitness = self.target_function(popu[0])
        flag = 0
        for i in range(self.population):
            if (self.target_function(popu[i]) < minFitness):
                minFitness = self.target_function(popu[i])
                flag = i
        Best = popu[flag]
        for i in range(self.func.dimension):
            BestWahle.append(Best[i])
        return BestWahle

    def run(self):
        count = 0
        Whale_position = self.initial_position()
        BestValue = self.getBest(Whale_position)
        best_ind = self.target_function(BestValue)
        while not self.stop_condition(count, best_ind):
            print("Iteration = ", count, " of ", self.iterations, " f(x) = ", best_ind)

            for i in range(self.population):
                a = 2 - 2 * count / self.iterations
                L = random.uniform(-1, 1)
                p = random.uniform(0, 1)
                C = 2 * random.uniform(0, 1)
                A = 2 * a * random.uniform(0, 1) - a
                if p < 0.5:
                    if np.abs(A) < 1:
                        flag = self.EncirclePrey(Whale_position[i], BestValue, A, C)
                        Whale_position[i] = flag
                    else:
                        index = random.randint(0, self.population - 1)
                        flag = self.SearchPrey(Whale_position[i], Whale_position[index], A, C)
                        Whale_position[i] = flag
                else:
                    flag = self.SpiralUpdate(Whale_position[i], BestValue, L)
                    Whale_position[i] = flag
            for i in range(self.population):
                for j in range(self.func.dimension):
                    if Whale_position[i][j] < self.func.min_values[j]:
                        Whale_position[i][j] = self.func.min_values[j]
                    if Whale_position[i][j] > self.func.max_values[j]:
                        Whale_position[i][j] = self.func.max_values[j]
            if self.target_function(self.getBest(Whale_position))<self.target_function(BestValue):
                best_ind=self.target_function(self.getBest(Whale_position))
                BestValue=self.getBest(Whale_position)
            self.current_solution.append(best_ind)
            count += 1
        Best = []
        for i in range(self.func.dimension):
            Best.append(BestValue[i])
        Best.append(best_ind)
        self.best_solution = Best
        return Best


