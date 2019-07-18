#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:27:37 2018

@author: robertjohnson
"""
import math
import numpy as np

#Method to work out the distance from a list of item to each cluster using euclidian distance
def euclidianDistance(listOfItems,currentStartingPoint):
    # create an empty array to store the nearest cluster to each point.
    closestPoint = []
    # for each item in the list of items
    for item in listOfItems:
        # work out the distance between this point and the first cluster
        minValue = math.sqrt(np.sum(np.subtract(currentStartingPoint[0],item[0])**2))
        minIndex = 0
        # for each cluster
        for second in range(1,len(currentStartingPoint)):
            # work out the distance
            currentDistance = math.sqrt(np.sum(np.subtract(currentStartingPoint[second],item[0])**2))
            # if the distance is less than the min found so far update.
            if (currentDistance < minValue):
                minValue = currentDistance
                minIndex = second
        # append the nearest cluster.
        closestPoint.append(minIndex)
    return closestPoint

#Method to work out the distance from a list of item to each cluster using manhatten distance
def manhattenDistance(listOfItems,currentStartingPoint):
    # create an empty array to store the nearest cluster to each point.
    closestPoint = []
    # for each item in the list of items
    for item in listOfItems:
        # work out the distance between this point and the first cluster
        minValue = np.sum(np.absolute(np.subtract(currentStartingPoint[0],item[0])))
        minIndex = 0
        # for each cluster
        for second in range(1,len(currentStartingPoint)):
            # work out the distance
            currentDistance = np.sum(np.absolute(np.subtract(currentStartingPoint[second],item[0])))
            # if the distance is less than the min found so far update.
            if (currentDistance < minValue):
                minValue = currentDistance
                minIndex = second
        # append the nearest cluster.
        closestPoint.append(minIndex)
    return closestPoint

#Method to work out the cosine similarity from a list of item to each cluster
def cosineDistance(listOfItems,currentStartingPoint):
    # create an empty array to store the nearest cluster to each point.
    closestPoint = []
    # for each item in the list of items
    for item in listOfItems:
        # work out the cosine similarity between this point and the first cluster
        minValue = np.dot(currentStartingPoint[0], item[0]) / (np.sqrt(np.dot(currentStartingPoint[0],currentStartingPoint[0])) * np.sqrt(np.dot(item[0],item[0])))
        minIndex = 0
        # for each cluster
        for second in range(1,len(currentStartingPoint)):
            # work out the similarity
            currentDistance = np.dot(currentStartingPoint[second], item[0]) / (np.sqrt(np.dot(currentStartingPoint[second],currentStartingPoint[second])) * np.sqrt(np.dot(item[0],item[0])))
            # if it is more similar then update
            if (currentDistance > minValue):
                minValue = currentDistance
                minIndex = second
            #append it.
        closestPoint.append(minIndex)
    return closestPoint

