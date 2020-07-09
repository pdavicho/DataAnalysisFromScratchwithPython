# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:10:15 2019

@author: pablo
"""

# ASSOCIATION RULE LEARNING
import pandas as pd
dataset = pd.read_csv('C:/Users/pablo/Desktop/IE345_DeepLearning/DataAnalysisFromScratchwithPython_Peters Morgan/Datasets/Market_Basket_Optimisation.csv', header=None)
dataset.head(10)

transaction = []
for i in range(0, 7501):
    transaction.append([str(dataset.values[i, j]) for j in range(0, 20) ])

from apyori import apriori

rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift=3, min_length=2)

results = list(rules)
results_list = []
for i in range (0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSupport: \t' + str(results[i][1]))
    print(results_list)