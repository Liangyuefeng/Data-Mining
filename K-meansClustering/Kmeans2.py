# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:15:50 2018

@author: Robert
"""


import numpy as np
import pathlib
import DistanceMeasures as DM
import MeasurePerformanceOfFinalClusters as MP
import graphing as GR

# readFile to work out the features and items in the dataset
# perams
# x: file name
# itemNum: the number representing the type of item
# listofitems: Items currently scanned
# listOfItemsOnlyFeatures: Items currently scanned only features
def readFile(x,itemNum,listOfItems,listOfItemsOnlyFeatures):
    # start reading the file
    file_object  = open(x, "r")
    y = file_object.readlines()
    # stop reading the file.
    file_object.close()
    # for each line in the document
    for val in y:
        # split it based on spaces
        a = val.split(" ")
        # create a new list
        c = []
        # append each feature to the list
        for val2 in range(1,len(a)):
            c.append(float(a[val2]))
        # create two numpy arrays
        c = np.array(c,dtype=np.float)
        l2 = np.array(c,dtype=np.float)
        # normalise the l2 array
        norm = np.sqrt(np.sum(l2[:] * l2[:]))
        l2[:] = l2[:] / norm
        # create an array of the feature vector, the name of the item and the item number
        temp = [c,a[0],itemNum]
        # create an array of the feature vector normalised, the name of the item and the item number
        l2temp = [l2,a[0],itemNum]
        # add the item to list of items
        listOfItemsOnlyFeatures.append(l2temp)
        listOfItems.append(temp)
    # return both lists
    return listOfItems, listOfItemsOnlyFeatures


# method to print the percentage 
# Used in Bio-Inspired Group project made by Robert Johnson and Michael Wright
# for the aims of displaying a percentage completion.
# perams
# iteration: the current iteration
# total: how many iterations there are
def print_percentage(iteration: int, total: int):
    fraction = (iteration) / total
    cubes = int(fraction * 40)
    to_print = ("{0:%i}" % 40).format("-" * cubes)
    print("Progress:▕%s▏(%.2f%%)\r" % (to_print, fraction * 100), end="")
    return


#run basic K-means movement of clusters
# perams
# listOfItems: list of items and feature vectors
# currentStartingPoint: starting clusters
# whatDistanceMeasure: the type of measuring to use.
# itemnumbers: the number of items chosen initially
# howManyDifferentItems: the amount of different types of item.
def runKMeansOnInput(listOfItems, currentStartingPoint, whatDistanceMeasure, itemnumbers, howManyDifferentItems):
    # array of distance methods
    arrayOfMethods = [DM.euclidianDistance,DM.euclidianDistance,DM.manhattenDistance,
                      DM.manhattenDistance,DM.cosineDistance,DM.cosineDistance]
    # for 100 iterations
    for p in range (100):
        # find out the distances from each point to each center using the distance method decided
        closestPoint = arrayOfMethods[whatDistanceMeasure](listOfItems, currentStartingPoint)
        # create a copy of the current clusters
        previousRound = currentStartingPoint[:]
        # set the current starting points to 0
        currentStartingPoint = np.zeros((k,300))
        # count how many items make up the current average of all items in the cluster
        countOfHowMany = np.zeros(k)
        # for each item
        for i in range(len(closestPoint)):
            # add its features to the cluster it is nearest to 
            currentStartingPoint[closestPoint[i]] = np.add(currentStartingPoint[closestPoint[i]],listOfItems[i][0])
            countOfHowMany[closestPoint[i]] += 1
        # for each cluster
        for i in range(k):
            # average the cluster's features
            currentStartingPoint[i] = currentStartingPoint[i]/countOfHowMany[i]
        # if this rounds centers are the same as the previous round then we have converged so break.
        if(np.array_equal(previousRound,currentStartingPoint)):
            break
    # return the current runs F-score, precision, recall, tp, fn, fp.
    F,P,R,TP,FN,FP = MP.workOutMetrics(currentStartingPoint, closestPoint, listOfItems, k, howManyDifferentItems)
    return F,R,P,TP,FP,FN



# K-means++ selection of clusters
# perams
# k : the number of clusters
# howManyPermutations : the amount of different starting points we would like to test
# listOfItems : list of datapoints
# listOfItemsNorm : list of datapoints normalised
# whatDistanceMeasure : the distance measure to use
# howManyDifferentItems : the amount of different types of item.
def runKMeansBasedOnKPlusPlus(k, howManyPermutations, listOfItems, listOfItemsNorm, whatDistanceMeasure, howManyDifferentItems):
    # if the distance measure is 0,2 or 4 then set the measure method to be un-normalised
    if whatDistanceMeasure == 0 or whatDistanceMeasure == 2 or whatDistanceMeasure == 4: measureMethod = listOfItems
    # else set it to the normalised
    else: measureMethod = listOfItemsNorm
    # create an empty list of f-scores, recalls, precisions, tps, fps, fns and items
    fscores = []
    recalls = []  
    precisions = []
    tps = []
    fps = []
    fns = []
    items = []
    # for the amount of iterations required
    for iterations in range (howManyPermutations):
        # create an empty list to store the current starting items
        itemsPicked = []
        firstPick = []
        # create an empty list to store probability to choose each item within the dataset as starting points
        temp = np.zeros(len(measureMethod))
        # fill the array with 1's
        temp.fill(1)
        whichItemToPick = []
        # create an array that stores the index to each item
        for i in range(len(measureMethod)):
            whichItemToPick.append(i)
        # get the total sum of the probabilitys
        sumOfArray = np.sum(temp)
        # divide it as probabilitys must add up to 1.
        temp = temp/sumOfArray
        # create a new array to store the names of the items
        nameOfItems = []
        # find k different cluster centers
        for amountOfK in range(k):
            # choose a random item from the list of index's
            numberOfItem = np.random.choice(whichItemToPick, 1, p=temp)[0]
            # add the number chosen
            itemsPicked.append(numberOfItem)
            # add the number chosen to the list of first pick centers
            firstPick.append(measureMethod[numberOfItem][0])
            # if amount of k = k-1 break
            if amountOfK == k-1: break
            # add the name of the item
            nameOfItems.append(measureMethod[numberOfItem][1])
            # for all items in the dataset
            for item in range(len(measureMethod)):
                # work out the euclidian distance
                minValue = np.linalg.norm(firstPick[0]-measureMethod[item][0])
                # for each additional center find out the minimal distance
                for second in range(1,len(firstPick)):
                    currentDistance = np.linalg.norm(firstPick[second]-measureMethod[item][0])
                    if (currentDistance < minValue):
                        minValue = currentDistance
                # update the minimal distance
                temp[item] = minValue
            # sum up all minimal distances
            sumOfArray = np.sum(temp)
            # divide all distances by the sum of distances to create a probability distribution.
            temp = temp/sumOfArray
        # run k-means and add statistics to list of statistics.
        fscore,precision,recall,tp,fp,fn = runKMeansOnInput(measureMethod, firstPick, whatDistanceMeasure, itemsPicked, howManyDifferentItems)
        fscores.append(fscore)
        recalls.append(recall)
        precisions.append(precision)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        items.append(itemsPicked)
    return fscores,recalls,precisions,tps,fps,fns,items


# K-means selection of clusters
# perams
# k : the number of clusters
# howManyPermutations : the amount of different starting points we would like to test
# listOfItems : list of datapoints
# listOfItemsNorm : list of datapoints normalised
# whatDistanceMeasure : the distance measure to use
# howManyDifferentItems : the amount of different types of item.
def runKMeans(k, howManyPermutations, listOfItems, listOfItemsNorm, whatDistanceMeasure, howManyDifferentItems):
    # if the distance measure is 0,2 or 4 then set the measure method to be un-normalised
    if whatDistanceMeasure == 0 or whatDistanceMeasure == 2 or whatDistanceMeasure == 4: measureMethod = listOfItems
    # else set it to the normalised
    else: measureMethod = listOfItemsNorm
    # create an empty list of f-scores, recalls, precisions, tps, fps, fns and items
    fscores = []
    recalls = []  
    precisions = []
    tps = []
    fps = []
    fns = []
    items = []
    # for the amount of iterations desired
    for iterations in range (howManyPermutations):
        # create an empty list to store the current starting items
        itemsPicked = []
        firstPick = []
        # create an empty list to store probability to choose each item within the dataset as starting points
        temp = np.zeros(len(measureMethod))
        # fill the array with 1's
        temp.fill(1)
        whichItemToPick = []
        # create an array that stores the index to each item
        for i in range(len(measureMethod)):
            whichItemToPick.append(i)
        # get the total sum of the probabilitys
        temp = temp/len(measureMethod)
        # create a new array to store the names of the items
        nameOfItems = []
        # find k different cluster centers
        for amountOfK in range(k):
            # choose a random item from the list of index's
            numberOfItem = np.random.choice(whichItemToPick, 1, p=temp)[0]
            # add the number chosen
            itemsPicked.append(numberOfItem)
            # add the number chosen to the list of first pick centers
            firstPick.append(measureMethod[numberOfItem][0])
            # if amount of k = k-1 break
            if amountOfK == k-1: break
            # add the name of the item
            nameOfItems.append(measureMethod[numberOfItem][1])
            amountOfItems = len(measureMethod)
            # for each item
            for item in range(len(measureMethod)):
                # set the probability to choose the item to 1
                temp[item] = 1
                # for each center if the center is the same then remove the probability to choose the current item
                for second in range(len(firstPick)):
                    if (np.array_equal(firstPick[second],measureMethod[item][0]) == True):
                        temp[item] = 0
                        amountOfItems -= 1
                # normalise the probabilitys.
            temp = temp/amountOfItems
        # run k-means and add statistics to list of statistics.
        fscore,precision,recall,tp,fp,fn = runKMeansOnInput(measureMethod, firstPick, whatDistanceMeasure, itemsPicked, howManyDifferentItems)
        fscores.append(fscore)
        recalls.append(recall)
        precisions.append(precision)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        items.append(itemsPicked)    
    return fscores,recalls,precisions,tps,fps,fns,items




# create an empty list of items
listOfItems = []
# create an empty list of normalised items
listOfItemsOnlyFeaturesL2 = []
# state number of different items.
howManyDifferentItems = 4
# read all the different types of items in to list of items and list of normalised item vectors
listOfItems, listOfItemsOnlyFeaturesL2 = readFile("animals", 0 ,listOfItems,listOfItemsOnlyFeaturesL2)
listOfItems, listOfItemsOnlyFeaturesL2 = readFile("countries", 1 ,listOfItems,listOfItemsOnlyFeaturesL2)
listOfItems, listOfItemsOnlyFeaturesL2 = readFile("fruits", 2 ,listOfItems,listOfItemsOnlyFeaturesL2)
listOfItems, listOfItemsOnlyFeaturesL2 = readFile("veggies", 3 ,listOfItems,listOfItemsOnlyFeaturesL2)
# convert each array into a numpy array
listOfItems = np.array(listOfItems)
listOfItemsOnlyFeaturesL2 = np.array(listOfItemsOnlyFeaturesL2)
# how many runs we would like to carry out
howManyPermutations = int(input("How many runs of each setting combination would you like to carry out?\n"))
wouldYouLikeToSeeGraphs = int(input("Would you like to see graphs for each run in program (1 = yes, 0 = no)?\n(The program will still output these graphs to file)\n"))
# the different methods being carried out.
methods = ["Euclidean","Euclidean with normalisation","Manhatten","Manhatten with normalisation","Cosine similarity","Cosine similarity with normalisation"]
# create a folder to store results
pathlib.Path("overallResults/").mkdir(parents=True, exist_ok=True) 
# clear the file for all the avg's
file = open("overallResults/allAvgs.txt","w") 
file.write("")
file.close() 
# for each method
for whatDistanceMeasure in range(6):
    # create lists to store avg's sd's max's for both k-means and k-means++
    listOfFScoresAvg = []
    listOfPrecisionAvg = []
    listOfRecallAvg = []
    listOfFScoresSD = []
    listOfPrecisionSD = []
    listOfRecallSD = []
    listOfFScoresmax = []
    listOfPrecisionmax = []
    listOfRecallmax = []
    listOfFScoresAvg2 = []
    listOfPrecisionAvg2 = []
    listOfRecallAvg2 = []
    listOfFScoresSD2 = []
    listOfPrecisionSD2 = []
    listOfRecallSD2 = []
    listOfFScoresmax2 = []
    listOfPrecisionmax2 = []
    listOfRecallmax2 = []
    # for each k value
    for k in range(1,11):
        # run normal k-means and store the results
        fscores,recalls,precisions,tps,fps,fns,items = runKMeans(k, howManyPermutations, listOfItems, listOfItemsOnlyFeaturesL2, whatDistanceMeasure, howManyDifferentItems)
        argMax = np.argmax(fscores)
        listOfFScoresmax.append(fscores[argMax])
        listOfPrecisionmax.append(precisions[argMax])
        listOfRecallmax.append(recalls[argMax])  
        averageFscore = np.average(fscores)
        averageRecalls = np.average(recalls)
        averagePrecisions = np.average(precisions)
        listOfFScoresSD.append(np.std(fscores))
        listOfRecallSD.append(np.std(recalls))
        listOfPrecisionSD.append(np.std(precisions))
        listOfFScoresAvg.append(averageFscore)
        listOfPrecisionAvg.append(averagePrecisions)
        listOfRecallAvg.append(averageRecalls)
        # run k-means++ and store the results
        fscores2,recalls2,precisions2,tps2,fps2,fns2,items2 = runKMeansBasedOnKPlusPlus(k, howManyPermutations, listOfItems, listOfItemsOnlyFeaturesL2, whatDistanceMeasure, howManyDifferentItems)
        argMax2 = np.argmax(fscores2)
        listOfFScoresmax2.append(fscores2[argMax2])
        listOfPrecisionmax2.append(precisions2[argMax2])
        listOfRecallmax2.append(recalls2[argMax2])
        averageFscore2 = np.average(fscores2)
        averageRecalls2 = np.average(recalls2)
        averagePrecisions2 = np.average(precisions2)
        listOfFScoresSD2.append(np.std(fscores2))
        listOfRecallSD2.append(np.std(recalls2))
        listOfPrecisionSD2.append(np.std(precisions2))
        listOfFScoresAvg2.append(averageFscore2)
        listOfPrecisionAvg2.append(averagePrecisions2)
        listOfRecallAvg2.append(averageRecalls2)
        print_percentage((10*whatDistanceMeasure)+k,10*6)
    # output the graph's and metrics.
    currentMethod = methods[whatDistanceMeasure].replace(" ", "_")
    pathlib.Path("graphskmeans/"+currentMethod+"/").mkdir(parents=True, exist_ok=True)
    GR.draw_graphIncSDFscore(listOfFScoresAvg, listOfPrecisionAvg, listOfRecallAvg, listOfFScoresSD, "Avg performance %s k-means++ (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans/"+currentMethod+"/fscore.png",wouldYouLikeToSeeGraphs)
    GR.draw_graphIncSDPrecision(listOfFScoresAvg, listOfPrecisionAvg, listOfRecallAvg, listOfPrecisionSD, "Avg performance using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans/"+currentMethod+"/presision.png")
    GR.draw_graphIncSDRecall(listOfFScoresAvg, listOfPrecisionAvg, listOfRecallAvg, listOfRecallSD, "Avg performance using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans/"+currentMethod+"/recall.png")
    GR.draw_graphMeans(listOfFScoresAvg, listOfPrecisionAvg, listOfRecallAvg, "Avg performance using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans/"+currentMethod+"/means.png")
    GR.draw_graphSD(listOfFScoresSD,listOfPrecisionSD,listOfRecallSD, "SD using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans/"+currentMethod+"/SD.png")
    GR.draw_graphMax(listOfFScoresmax, listOfPrecisionmax, listOfRecallmax, "Max using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans/"+currentMethod+"/max.png")
    file = open("overallResults/allAvgs.txt","a") 
    file.write("graphskmeans/"+currentMethod+"/" + " # " + str(listOfFScoresAvg) + " # " + str(listOfPrecisionAvg) + " # " + str(listOfRecallAvg) + " # " + str(listOfFScoresSD) + " # " + str(listOfPrecisionSD) + " # " + str(listOfRecallSD) + " # " + str(listOfFScoresmax) + " # " + str(listOfPrecisionmax) + " # " + str(listOfRecallmax)+"\n")
    pathlib.Path("graphskmeans++/"+currentMethod+"/").mkdir(parents=True, exist_ok=True) 
    GR.draw_graphIncSDFscore(listOfFScoresAvg2, listOfPrecisionAvg2, listOfRecallAvg2, listOfFScoresSD2, "Avg performance %s k-means (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans++/"+currentMethod+"/fscore.png",wouldYouLikeToSeeGraphs)
    GR.draw_graphIncSDPrecision(listOfFScoresAvg2, listOfPrecisionAvg2, listOfRecallAvg2, listOfPrecisionSD2, "Avg performance using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans++/"+currentMethod+"/presision.png")
    GR.draw_graphIncSDRecall(listOfFScoresAvg2, listOfPrecisionAvg2, listOfRecallAvg2, listOfRecallSD2, "Avg performance using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans++/"+currentMethod+"/recall.png")
    GR.draw_graphMeans(listOfFScoresAvg2, listOfPrecisionAvg2, listOfRecallAvg2, "Avg performance using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans++/"+currentMethod+"/means.png")
    GR.draw_graphSD(listOfFScoresSD2,listOfPrecisionSD2,listOfRecallSD2, "SD using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans++/"+currentMethod+"/SD.png")
    GR.draw_graphMax(listOfFScoresmax2, listOfPrecisionmax2, listOfRecallmax2, "Max using %s (%d permutations)"%(methods[whatDistanceMeasure],howManyPermutations),"graphskmeans++/"+currentMethod+"/max.png")
    file.write("graphskmeans++/"+currentMethod+"/" + " # " + str(listOfFScoresAvg2) + " # " + str(listOfPrecisionAvg2) + " # " + str(listOfRecallAvg2) + " # " + str(listOfFScoresSD2) + " # " + str(listOfPrecisionSD2) + " # " + str(listOfRecallSD2) + " # " + str(listOfFScoresmax2) + " # " + str(listOfPrecisionmax2) + " # " + str(listOfRecallmax2)+"\n")
    file.close() 