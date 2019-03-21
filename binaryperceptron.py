# utf-8
import numpy as np
from random import shuffle
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def testAndTrain(iteration, trainingData, testingData, shuffleVal):
    global onevsTwo, twoVsThree, oneVsThree
    print("\n\n" + "Binary classifier" + "\n\n")
    # Declare variables to store the results.
    test1In1vs2TP = test1In1vs2FP = 0
    test1In3vs1TP = test1In3vs1FP = 0

    test2In1vs2TP = test2In1vs2FP = 0
    test2In2vs3TP = test2In2vs3FP = 0

    test3In2vs3TP = test3In2vs3FP = 0
    test3In3vs1TP = test3In3vs1FP = 0

    # create a binary perceptron to compare class 1 and class 2.
    onevsTwo = perceptTrain(iteration, trainingData, 1, 2, shuffleVal)
    # create a binary perceptron to compare class 2 and class 3.
    twoVsThree = perceptTrain(iteration, trainingData, 2, 3, shuffleVal)
    # create a binary perceptron to compare class 1 and class 3.
    oneVsThree = perceptTrain(iteration, trainingData, 1, 3, shuffleVal)

    # test each data and compute the TP or FP
    for features, Class in testingData:
        # If the record we are processing is either class 1 or class 2
        if Class == 1 or Class == 2:
            # test,if the activation is above 0 then the record is predicted in class 1
            if (np.dot(onevsTwo[0], features) + onevsTwo[1]) > 0:
                # if the actual class was 1 then we predicted correctly
                if Class == 1:
                    test1In1vs2TP += 1
                # else we predicted falsely
                else:
                    test1In1vs2FP += 1
                # else its the opposite class (class 2)
            else:
                # if it was the other class we predicted correctly
                if Class == 2:
                    test2In1vs2TP += 1
                # else we predicted falsely
                else:
                    test2In1vs2FP += 1

            # If the record we are processing is either class 2 or class 3
        if Class == 2 or Class == 3:
            # test the network if the activation is above 0 then the record is predicted in class 2
            if (np.dot(twoVsThree[0], features) + twoVsThree[1]) > 0:
                # if the actual class was 2 then we predicted correctly
                if Class == 2:
                    test2In2vs3TP += 1
                # else we predicted falsely
                else:
                    test2In2vs3FP += 1
                # else if the actual class was 3 then we predicted correctly
            else:
                if Class == 3:
                    test3In2vs3TP += 1
                # else we predicted falsely
                else:
                    test3In2vs3FP += 1

            # if the class we are testing is either 3 or 1
        if Class == 3 or Class == 1:
            # test the network if the activation is above 0 then the record is predicted in class 1
            if (np.dot(oneVsThree[0], features) + oneVsThree[1]) > 0:
                # if the class was actually 1 then we predicted correctly
                if Class == 1:
                    test1In3vs1TP += 1
                # else we predicted falsely
                else:
                    test1In3vs1FP += 1
            # else if the actual class was 3 then we predicted correctly
            else:
                if Class == 3:
                    test3In3vs1TP += 1
                # else we predicted falsely.
                else:
                    test3In3vs1FP += 1

    # print the accuracy for network 1 vs 2.
    print(printTestAccuracy(test1In1vs2TP, test1In1vs2FP, test2In1vs2FP, test2In1vs2TP, "1", "2"))
    # print the weights for network 1 vs 2
    print("\tWeights of 1 vs 2 = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f\n\n" % (
        onevsTwo[0][0], onevsTwo[0][1], onevsTwo[0][2], onevsTwo[0][3], onevsTwo[1]))

    # print the accuracy for network 2 vs 3.
    print(printTestAccuracy(test2In2vs3TP, test2In2vs3FP, test3In2vs3FP, test3In2vs3TP, "2", "3"))
    # print the weights for network 2 vs 3
    print("\tWeights of 2 vs 3 = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f\n\n" % (
        twoVsThree[0][0], twoVsThree[0][1], twoVsThree[0][2], twoVsThree[0][3], twoVsThree[1]))

    # print the accuracy for network 1 vs 3.
    print(printTestAccuracy(test3In3vs1TP, test3In3vs1FP, test1In3vs1FP, test1In3vs1TP, "3", "1"))
    # print the weights for network 1 vs 3
    print("\tWeights of 1 vs 3 = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f\n\n" % (
        oneVsThree[0][0], oneVsThree[0][1], oneVsThree[0][2], oneVsThree[0][3], oneVsThree[1]))

    # print and draw graph for accuracy on training data
    accuracy1 = onevsTwo[2]
    accuracy2 = twoVsThree[2]
    accuracy3 = oneVsThree[2]

    print("Accuracy during training class 1 vs 2")
    graph(accuracy1, iteration, "1 vs 2")
    print(accuracy1)
    print("Accuracy during training class 2 vs 3")
    graph(accuracy2, iteration, "2 vs 3")
    print(accuracy2)
    print("Accuracy during training class 3 vs 1")
    graph(accuracy3, iteration, "3 vs 1")
    print(accuracy3)

    return


def perceptTrain(maxIter, Data, ClassN, ClassN2, shuffleVal):
    # initialise weights to 0
    accuracy = []
    weights = np.array([0, 0, 0, 0])
    # initialise bias to 0
    bias = 0
    # run system for maxIter times
    for i in range(0, maxIter):
        mistakes = 0
        # shuffle the data
        if shuffleVal == 1:
            shuffle(Data)
        # for each data split on features and class
        for features, classN in Data:
            # if the data class is either of the two class we want to distinguish
            if classN == ClassN or classN == ClassN2:
                # set the tempclass to be 1 if it is the first class
                # else set it to -1.
                tempSetNum = 1 if classN == ClassN else -1
                # perform matrix multiplication on features and weights, add the bias (activation term)
                a = np.dot(weights, features) + bias
                if tempSetNum * a <= 0:
                    # update the weights and bias.
                    weights = np.add(weights, np.multiply(tempSetNum, features))
                    bias += tempSetNum
                    mistakes += 1
        # return the weights and bias
        r = round((len(traindata) - mistakes) / len(traindata), 2)
        accuracy.append(r)
    return weights, bias, accuracy


# read file x and output an array containing a numpy array of the features and the class of the data
def readFile(x):
    # open the file
    file = open(x, "r")
    # get all the lines
    lines = file.readlines()
    # close the file
    file.close()
    # create a new array
    data = []
    # for every metadata
    for val in lines:
        # split it on ","
        a = val.split(",")
        # get the class
        a[4] = a[4][6:7]
        c = []
        # add all features to an empty array
        for val2 in range(0, len(a) - 1):
            c.append(float(a[val2]))
        # convert the array to a numpy array
        c = np.array(c)
        temp = [c, int(a[4])]
        data.append(temp)
    return data


def printTestAccuracy(TP, FP, FN, TN, index1, index2):
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print("Classification " + index1 + " vs " + index2 + "\n" +
          "\ttest Accuracy\t\t" + str(accuracy))
    return


def graph(data, episode_count, title):
    time_intervals = np.arange(1, episode_count + 1, 1)
    ax = plt.axes()
    ax.plot(time_intervals, data, 'r', linewidth=0.5)
    loc = ticker.MultipleLocator(base=1.0)
    # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.show()
    return


# runs all tests including binary training, binary testing
with np.errstate(over='ignore'):
    # ShuffleVal, if this is set to 1 will shuffle data .
    shuffleval = 1
    # iterations, how many times to repeatedly pass our data through test method.
    iterations = 20
    # get the training data and store it in an array.
    traindata = readFile("train.data")
    # get the training data and store it in an array.
    testdata = readFile("test.data")
    testAndTrain(iterations, traindata, testdata, shuffleval)
