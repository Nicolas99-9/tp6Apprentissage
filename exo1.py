#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

class EpsilonGreedy:


    def __init__(self,epsilon):
        self.epsilon = epsilon

    #number of arms
    def initialize(self,n_arms):
        self.n_arms = n_arms
        self.arms = [0 for i  in range(n_arms)]
        self.etapes = {}
        for i in range(n_arms):
            self.etapes[i] = []

    def select_arms(self):
        if random.random() > self.epsilon:
            return np.argmax(self.arms)#exploite
        return random.randint(0,self.n_arms-1) #explore
    
    def update(self,chosen_arm,reward):
        self.etapes[chosen_arm].append(reward)
        mn = sum(self.etapes[chosen_arm])/float(len(self.etapes[chosen_arm]))
        self.arms[chosen_arm] = mn
        


class BernoulliArm:
    def __init__(self,p):
        self.p = p 
    def draw(self):
        return 0.0 if random.random() > self.p else 1.0

def test_algorithm(algo,means,num_sims,horizon):
    arms = [BernoulliArm(mu) for mu in means]
    rewards = []
    for sim in range(num_sims):
        algo.initialize(len(arms))
        for t in range(horizon):
            chosen_arm = algo.select_arms()
            reward = arms[chosen_arm].draw()
            algo.update(chosen_arm,reward)
            rewards.append(reward)
    return np.array(rewards).reshape((num_sims,horizon))



# means  :
# num_sims : number of times the algo is launch
#horizon : number of iteration in the epsilon greedy function
test = [0.1,0.1,0.1,0.1,0.9]

result = []
#tester la valeur de epsilon pour faire varier exploration/ exploitation
for i in range(100):
    print(i)
    count = i*0.01
    algo = EpsilonGreedy(count)
    results = test_algorithm(algo,test,100,250)
    result.append(np.mean([np.mean(w) for w in results]))

print(result)
print("Epsilon max  : ", float(np.argmax(result))*0.01,result[np.argmax(result)])


'''
0 correspond à une reward de 0
epsilon 0.3 ==> 0.395 ,  valeur max

'''

