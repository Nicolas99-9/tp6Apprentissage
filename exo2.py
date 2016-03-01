#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

class EpsilonGreedy:


    def __init__(self,epsilon,to):
        self.epsilon = epsilon
        self.to = to

    #number of arms
    def initialize(self,n_arms):
        self.n_arms = n_arms
        self.arms = [0 for i  in range(n_arms)]
        self.etapes = {}
        for i in range(n_arms):
            self.etapes[i] = []

    def select_arms(self,too):
        if random.random() > self.epsilon:
            return np.argmax(self.arms)#exploite
        else:
            self.to = (1.0/(too+pow(10,-7)))
            cout = 0.0
            for i in range(self.n_arms):
                cout += np.exp((self.arms[i]/self.to))
            test = [(np.exp((self.arms[i]/self.to))/cout) for i in range(self.n_arms)]
            return np.random.choice(range(self.n_arms),p=  test)
    
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
            chosen_arm = algo.select_arms(horizon)
            reward = arms[chosen_arm].draw()
            algo.update(chosen_arm,reward)
            rewards.append(reward)
    return np.array(rewards).reshape((num_sims,horizon))  #tableau indiquant si on a ete recompense



# means  :
# num_sims : number of times the algo is launch
#horizon : number of iteration in the epsilon greedy function
test = [0.1,0.1,0.1,0.1,0.9]

result = []
random.shuffle(test)
#tester la valeur de epsilon pour faire varier exploration/ exploitation
#mettre 100 pour varier de 0.01  à 1.0
for i in range(25):
    print(i)
    count = i*0.01
    algo = EpsilonGreedy(count,2)
    results = test_algorithm(algo,test,100,250)
    result.append(np.mean([np.mean(w) for w in results]))

print(result)
print("Epsilon max  : ", float(np.argmax(result))*0.01,result[np.argmax(result)])


'''
0 correspond à une reward de 0
epsilon 0.18 ==> 0.71 ,  valeur max


[0.098159999999999983, 0.23960000000000001, 0.37944000000000011, 0.50212000000000001, 0.53603999999999996, 0.54455999999999993, 0.62844000000000011, 0.63619999999999999, 0.64603999999999995, 0.65827999999999987, 0.70928000000000002, 0.69628000000000001, 0.72140000000000004, 0.71028000000000002, 0.71611999999999998, 0.69940000000000013, 0.71239999999999992, 0.71132000000000006, 0.71700000000000008, 0.70780000000000021, 0.71192000000000022, 0.7006, 0.70587999999999995, 0.69255999999999995, 0.6935199999999998]
('Epsilon max  : ', 0.12, 0.72140000000000004)

'''

