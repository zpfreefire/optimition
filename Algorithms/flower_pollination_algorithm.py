from Algorithms.algorithm import Algorithm
import numpy as np
import random
from scipy.stats import levy

class FPA(Algorithm):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.P=kwargs.pop('P',0.8)

    def initial_position(self):
        Flower_position = np.zeros((self.population, self.func.dimension))
        for i in range(0, self.population):
            for j in range(0, self.func.dimension):
                Flower_position[i, j] = random.uniform(self.func.min_values[j], self.func.max_values[j])
        return Flower_position

    def getBest(self, population):
        minFitness = self.target_function(population[0])
        flag = 0
        for i in range(len(population)):
            if (self.target_function(population[i]) < minFitness):
                minFitness = self.target_function(population[i])
                flag = i
        Best = population[flag]
        return Best

    def Global_Pollination(self,popu,BestValue):
        pop = []
        step = []
        for i in range(self.func.dimension):
            step.append(levy.pdf(1.5))
        for i in range(self.func.dimension):
            temp = popu[i] + step[i] * (BestValue[i] - popu[i])
            pop.append(temp)
        return pop

    def Local_Pollination(self,popu,Population):
        pop = []
        alpha = np.random.uniform(0, 1)
        index_one = np.random.randint(0, self.population - 1)
        index_two = np.random.randint(0, self.population - 1)
        while index_one == index_two:
            index_one = np.random.randint(0, self.population - 1)
            index_two = np.random.randint(0, self.population - 1)
        for i in range(self.func.dimension):
            temp = popu[i] + alpha * (Population[index_two][i] - Population[index_one][i])
            pop.append(temp)
        return pop

    def run(self):
        count=0
        Flower=self.initial_position()
        BestValue=self.getBest(Flower)
        best_ind=self.target_function(BestValue)
        while not self.stop_condition(count,best_ind):
            print("Iteration = ", count, " of ", self.iterations, " f(x) = ", best_ind)

            for i in range(self.population):
                if np.random.rand()<self.P:
                    temp=self.Global_Pollination(Flower[i],BestValue)
                else:
                    temp=self.Local_Pollination(Flower[i],Flower)
                for j in range(self.func.dimension):
                    if temp[j]<self.func.min_values[j]:
                        temp[j]=self.func.min_values[j]
                    if temp[j]>self.func.max_values[j]:
                        temp[j]=self.func.max_values[j]
                if self.target_function(temp)<self.target_function(Flower[i]):
                    Flower[i]=temp
            BestValue=self.getBest(Flower)
            best_ind=self.target_function(BestValue)
            self.current_solution.append(best_ind)
            count+=1
        Best=[]
        for i in range(self.func.dimension):
            Best.append(BestValue[i])
        Best.append(best_ind)
        self.best_solution=Best
        return Best

