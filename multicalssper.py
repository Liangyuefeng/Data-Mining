import numpy as np
from random import shuffle
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def testAndTrainMultiClass(coef, iteration, trainData, testData, shuffleVal):
    global one, two, three
    print("\n\n" + " multi-class classfication" + "\n\n")
    # Declare variables to store the results.
    p1A1 = p1A2 = p1A3 = 0
    p2A1 = p2A2 = p2A3 = 0
    p3A1 = p3A2 = p3A3 = 0

    # create a multi-class network to classify 1 or not 1.
    one = perceptTrainMulti(coef, iteration, trainData, 1, shuffleVal)
    # create a multi-class network to classify 1 or not 1.
    two = perceptTrainMulti(coef, iteration, trainData, 2, shuffleVal)
    # create a multi-class network to classify 1 or not 1.
    three = perceptTrainMulti(coef, iteration, trainData, 1, shuffleVal)

    # test each data
    for features, ActualClass in testData:
        classpred = np.argmax(
            [perceptTest(one, features), perceptTest(two, features), perceptTest(three, features)]) + 1
        # if the actual class is 1
        if ActualClass == 1:
            # and we predicted 1 then we predicted correctly
            if classpred == 1:
                p1A1 += 1
            # else we predicted class 1 when it was class 2
            elif classpred == 2:
                p2A1 += 1
            # else we predicted class 1 when it was class 3
            else:
                p3A1 += 1

        # if the actual class is 2
        if ActualClass == 2:
            # and we predicted 2 then we predicted correctly
            if classpred == 2:
                p2A2 += 1
            # else we predicted class 2 when it was class 1
            elif classpred == 1:
                p1A2 += 1
            # else we predicted class 2 when it was class 3
            else:
                p3A2 += 1

        # if the actual class is 3
        if ActualClass == 3:
            # and we predicted 3 then we predicted correctly
            if classpred == 3:
                p3A3 += 1
            # else we predicted class 3 when it was class 1
            elif classpred == 1:
                p1A3 += 1
            # else we predicted class 3 when it was class 2
            else:
                p2A3 += 1

    print("\nResults found for coef %0.2f" % coef)
    # prints out  the accuracy representing the results of testing.
    N = p2A1 + p3A1 + p1A2 + p1A3 + p1A1 + p2A2 + p2A3 + p3A2 + p3A3

    print("\tAccuracy of predicting class 1\t" + str((p1A1 + p2A2 + p2A3 + p3A3 + p3A2) / N))
    print("\tAccuracy of predicting class 2\t" + str((p2A2 + p1A1 + p1A3 + p3A1 + p3A3) / N))
    print("\tAccuracy of predicting class 3\t" + str((p3A3 + p1A1 + p1A2 + p2A1 + p2A2) / N))
    # print out the individual networks weights
    print("\tWeights of 1 vs all = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" % (
        one[0][0], one[0][1], one[0][2], one[0][3], one[1]))
    print("\tWeights of 2 vs all = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" % (
        two[0][0], two[0][1], two[0][2], two[0][3], two[1]))
    print("\tWeights of 3 vs all = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f\n" % (
        three[0][0], three[0][1], three[0][2], three[0][3], three[1]))

    # print and draw graph for accuracy on traindata
    accuracy1 = one[2]
    accuracy2 = two[2]
    accuracy3 = three[2]
    print("Accuracy during training class 1 vs all")
    graph(accuracy1, iteration, "1 vs all coef(" + str(coef) + ")")
    print(accuracy1)
    print("Accuracy during training class 2 vs all")
    graph(accuracy2, iteration, "2 vs all coef(" + str(coef) + ")")
    print(accuracy2)
    print("Accuracy during training class 3 vs all")
    graph(accuracy3, iteration, "3 vs all coef(" + str(coef) + ")")
    print(accuracy3)
    return


def perceptTrainMulti(regcoef, maxIter, Data, ClassValue, shuffleVal):
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
            # if its the class we are looking for set the temp class to be 1
            # else set it to be 0
            tempSetNum = 1 if classN == ClassValue else -1
            # perform matrix multiplication on features and weights, add the
            # bias (activation term)
            a = np.dot(weights, features) + bias
            # if the network activated and it shouldn't have
            if tempSetNum * a <= 0:
                # update the weights and bias.
                Tw = np.multiply(tempSetNum, features)
                weights = np.subtract(np.add(weights, Tw), np.multiply(2 * regcoef, weights))
                bias += tempSetNum
                mistakes += 1
        # return the weights and bias
        k = round((len(traindata) - mistakes) / len(traindata), 3)
        accuracy.append(k)
    return weights, bias, accuracy


# method to test the neural network takes the network and record.
def perceptTest(network, record):
    # return the activation value.
    return np.dot(network[0], record) + network[1]

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
        i = [c, int(a[4])]
        data.append(i)
    return data

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


# runs all tests including multi-class training, multi-class testing
# ShuffleVal, if this is set to 1 will shuffle data.
shuffleval = 1
# iterations, how many times to repeatedly pass our data through test method.
iterations = 20
# get the training data and store it in an array.
traindata = readFile("train.data")
# get the training data and store it in an array.
testdata = readFile("test.data")
# for each coefficient we would like to test
for Coef in ([0, 0.01, 0.1, 1, 10, 100]):  # , 0.01, 0.1, 1, 10, 100
    testAndTrainMultiClass(Coef, iterations, traindata, testdata, shuffleval)
