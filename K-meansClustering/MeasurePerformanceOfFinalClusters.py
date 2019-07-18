#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:29:53 2018

@author: robertjohnson
"""
import math
import copy
import numpy as np

def nCr(n,r):
    # if n = r then we can only pick 1.
    if n == r: return 1
    # if n is less than r then we are unable to pick r values from n
    if n < r: return 0
    # else define factorial and perform nCr.
    f = math.factorial
    return f(n) / (f(r) * f(n-r))

def workOutMetrics(currentStartingPoint, closestPoint, listOfItems, k, howManyDifferentItems):
    itemsInEachGroup = []
    # Copy the list of items
    listOfItemsCopy = copy.deepcopy(listOfItems)
    # create a list to append all groups
    for i in range(k): itemsInEachGroup.append([])
    # add the type of an item to the cluster it is most close to
    for item in range(len(closestPoint)):
        itemsInEachGroup[closestPoint[item]].append(listOfItemsCopy[item][2])
    # Count how many datasets are in each cluster
    howManyInEachGroup = np.unique(closestPoint, return_counts=True)[1]
    TPandFP = 0
    # get nCr for each type of item in the dataset and work out TPandFP
    for group in howManyInEachGroup: TPandFP += nCr(group,2)
    # create a list of lists for each cluster of size, howManyDifferentItems in this case it is 4.
    seperateItemsInEachGroup = np.zeros((k,howManyDifferentItems))
    TP = 0
    # for each cluster
    for i in range (len(seperateItemsInEachGroup)):
        # count how many of each type of item is in the cluster
        temp = np.unique(itemsInEachGroup[i], return_counts=True)
        # for each item type in the cluster
        for j in range (len(temp[0])):
            # put the amount of a certain type into the specified array location.
            seperateItemsInEachGroup[i][temp[0][j]] = temp[1][j]
            # perform nCr on the size of each item within each cluster.
            TP += nCr(temp[1][j],2)
    FP = TPandFP - TP
    FN = 0
    # for each cluster
    for i in range(len(seperateItemsInEachGroup)):
        # for each type of item in each cluster
        for j in range(len(seperateItemsInEachGroup[i])):
            # for every other cluster.
            for k in range(i+1,len(seperateItemsInEachGroup)):
                # if the type of item in the current cluster is 0 then break.
                if(seperateItemsInEachGroup[i][j]==0): break
                # else multiply it.
                FN += seperateItemsInEachGroup[i][j]*seperateItemsInEachGroup[k][j]
    TNandFN = 0
    # each cluster
    for i in range(len(howManyInEachGroup)):
        # for each cluster after the one in the outer loop
        for j in range(i+1,len(howManyInEachGroup)):
            # multiply the sizes of them.
            TNandFN += howManyInEachGroup[i]*howManyInEachGroup[j]
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F = (2*P*R)/(P+R)
    # return precision, recall and f-score
    return F,P,R,TP,FN,FP