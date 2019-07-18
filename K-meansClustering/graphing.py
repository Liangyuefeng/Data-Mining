# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:15:50 2018

@author: Robert
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Method to draw the average f-score, precision, recall, candlesticks of f-score based on sd, the distancemeasure i.e euclidian
# and the filename to store the image
# perams
# fscore - an array storing the values of average f-scores for each k
# precision - an array storing the values of average precision
# recall - an array storing the values of average recall
# sdFscore - an array storing the values of SD for f-score
# distanceMeasure - a string stating the distance measure used
# filename - the filename to output at. 
def draw_graphIncSDFscore(fscore, precision, recall, sdFscore, distanceMeasure, filename, wouldYouLikeToSeeGraphs):
	# arrange the records to start from 1.
    tmp1 = np.arange(1, len(fscore) + 1)
    # plot f-score using a candlestick and precision and recall using normal plots
    plt.plot(tmp1, precision, 'g', linewidth=0.5)
    plt.plot(tmp1, recall, 'b', linewidth=0.5)  
    plt.errorbar(tmp1, 
        fscore, 
        sdFscore, 
        capsize=5, 
        elinewidth=1,
        markeredgewidth=2, ecolor='red',fmt='r',linewidth=1,capthick = 0.5)
    red_patch = mpatches.Patch(color='red', label='F-score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.xlabel("K")
    plt.ylabel("Percentage")
    plt.title("%s" % distanceMeasure)
    plt.grid(True)
    plt.ylim((0,1.1))
    if(wouldYouLikeToSeeGraphs == 1):
        print("Graph has been created! Please close the graph to continue\n")
        fig = plt.gcf()
        plt.show()
        fig.savefig(filename)
        plt.close(fig)
        plt.clf()
    else:
        plt.savefig(filename)
        plt.clf()
    return

# Method to draw the average f-score, precision, recall, candlesticks of precision based on sd, the distancemeasure i.e euclidian
# and the filename to store the image
# perams
# fscore - an array storing the values of average f-scores for each k
# precision - an array storing the values of average precision
# recall - an array storing the values of average recall
# sdPrecision - an array storing the values of SD for precision
# distanceMeasure - a string stating the distance measure used
# filename - the filename to output at. 
def draw_graphIncSDPrecision(fscore, precision, recall, sdPrecision, distanceMeasure, filename):
	# arrange the records to start from 1.
    tmp1 = np.arange(1, len(fscore) + 1)
    # plot Precision using a candlestick and f-score and recall using normal plots
    plt.plot(tmp1, fscore, 'r', linewidth=0.5)
    plt.plot(tmp1, recall, 'b', linewidth=0.5)  
    plt.errorbar(tmp1, 
        precision, 
        sdPrecision, 
        capsize=5, 
        elinewidth=1,
        markeredgewidth=2, ecolor='green',fmt='g',linewidth=1,capthick = 0.5)
    red_patch = mpatches.Patch(color='red', label='F-score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.xlabel("K")
    plt.ylabel("Percentage")
    plt.title("%s" % distanceMeasure)
    plt.grid(True)
    plt.ylim((0,1.1))
    plt.savefig(filename)
    plt.clf()
    return


# Method to draw the average f-score, precision, recall, candlesticks of recall based on sd, the distancemeasure i.e euclidian
# and the filename to store the image
# perams
# fscore - an array storing the values of average f-scores for each k
# precision - an array storing the values of average precision
# recall - an array storing the values of average recall
# sdRecall - an array storing the values of SD for Recall
# distanceMeasure - a string stating the distance measure used
# filename - the filename to output at. 
def draw_graphIncSDRecall(fscore, precision, recall, sdRecall, distanceMeasure, filename):
	# arrange the records to start from 1.
    tmp1 = np.arange(1, len(fscore) + 1)
    # plot Recall using a candlestick and precision and f-score using normal plots
    plt.plot(tmp1, precision, 'g', linewidth=0.5)
    plt.plot(tmp1, fscore, 'r', linewidth=0.5)  
    plt.errorbar(tmp1, 
        recall, 
        sdRecall, 
        capsize=5, 
        elinewidth=1,
        markeredgewidth=2, ecolor='blue',fmt='b',linewidth=1,capthick = 0.5)
    red_patch = mpatches.Patch(color='red', label='F-score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.xlabel("K")
    plt.ylabel("Percentage")
    plt.title("%s" % distanceMeasure)
    plt.grid(True)
    plt.ylim((0,1.1))
    plt.savefig(filename)
    plt.clf()
    return

# Method to draw the average f-score, precision, recall, candlesticks of f-score based on sd, the distancemeasure i.e euclidian
# and the filename to store the image
# perams
# fscore - an array storing the values of average f-scores for each k
# precision - an array storing the values of average precision
# recall - an array storing the values of average recall
# distanceMeasure - a string stating the distance measure used
# filename - the filename to output at. 
def draw_graphMeans(fscore, precision, recall, distanceMeasure, filename):
	# arrange the records to start from 1.
    tmp1 = np.arange(1, len(fscore) + 1)
    # plot Recall, precision, f-score using normal plotting
    plt.plot(tmp1, precision, 'g', linewidth=0.5)
    plt.plot(tmp1, fscore, 'r', linewidth=0.5)
    plt.plot(tmp1, recall, 'b', linewidth=0.5) 
    red_patch = mpatches.Patch(color='red', label='F-score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.xlabel("K")
    plt.ylabel("Percentage")
    plt.title("%s" % distanceMeasure)
    plt.grid(True)
    plt.ylim((0,1.1))
    plt.savefig(filename)
    plt.clf()
    return

# Method to draw the average f-score, precision, recall, candlesticks of f-score based on sd, the distancemeasure i.e euclidian
# and the filename to store the image
# perams
# sdfscore - an array storing the values of sd f-scores for each k
# sdprecision - an array storing the values of sd precision
# sdrecall - an array storing the values of sd recall
# distanceMeasure - a string stating the distance measure used
# filename - the filename to output at. 
def draw_graphSD(sdFscore, sdPrecision, sdRecall, distanceMeasure, filename):
	# arrange the records to start from 1.
    tmp1 = np.arange(1, len(sdPrecision) + 1)
    # plot SDRecall, SDprecision, SDf-score using normal plotting
    plt.plot(tmp1, sdPrecision, 'g', linewidth=0.5)
    plt.plot(tmp1, sdFscore, 'r', linewidth=0.5)
    plt.plot(tmp1, sdRecall, 'b', linewidth=0.5) 
    red_patch = mpatches.Patch(color='red', label='F-score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.xlabel("K")
    plt.ylabel("Percentage")
    plt.title("%s" % distanceMeasure)
    plt.grid(True)
    plt.ylim((0,0.18))
    plt.savefig(filename)
    plt.clf()
    return

# Method to draw the average f-score, precision, recall, candlesticks of f-score based on sd, the distancemeasure i.e euclidian
# and the filename to store the image
# perams
# fscore - an array storing the max values of f-scores for each k
# precision - an array storing the max values of precision
# recall - an array storing the max values of recall
# distanceMeasure - a string stating the distance measure used
# filename - the filename to output at. 
def draw_graphMax(fscore, precision, recall, distanceMeasure, filename):
	# arrange the records to start from 1.
    tmp1 = np.arange(1, len(fscore) + 1)
    # plot MaxRecall, Maxprecision, Maxf-score using normal plotting
    plt.plot(tmp1, precision, 'g', linewidth=0.5)
    plt.plot(tmp1, fscore, 'r', linewidth=0.5)
    plt.plot(tmp1, recall, 'b', linewidth=0.5) 
    red_patch = mpatches.Patch(color='red', label='F-score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.xlabel("K")
    plt.ylabel("Percentage")
    plt.title("%s" % distanceMeasure)
    plt.grid(True)
    plt.ylim((0,1))
    plt.savefig(filename)
    plt.clf()
    return