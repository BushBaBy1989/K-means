import numpy as np
import matplotlib.pyplot as plt
#better than import math
from math import sqrt
import random
import seaborn as sns
import pandas as pd

#needs to be more than max countries
pd.options.display.max_rows = 500

#read csv method
def read_csv():
    read = pd.read_csv("dataBoth.CSV", delimiter=',')
    ctryName = read[read.columns[0]].values
    ctryValues = read[[read.columns[1], read.columns[2]]].values
    return ctryValues, ctryName

#call read csv
x = read_csv()
x_list = np.ndarray.tolist(x[0][0:, :])
#get number of clusters from user
k = int(input("Enter the number of clusters: "))
centroids = random.sample(x_list, k)
#get number of iterations from user
i = int(input("Enter the number of iterations: "))

#calculate distance
def dist(centriods, datapoints):
    dist = []
    for centroid in centriods:
        for datapoint in datapoints:
            dist.append(sqrt((datapoint[0]-centroid[0])**2 + (datapoint[1]-centroid[1])**2))
    return dist

#calculate each data point
def sortCluster(x_in=x, centroids_in=centroids, inp=k):
    distRe = np.reshape(dist(centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
    dpCen = []
    for value in zip(*distRe):
        dpCen.append(np.argmin(value)+1)
    clusters = {}
    for j in range(0, inp):
        clusters[j+1] = []
    for d_point, cent in zip(x_in[0], dpCen):
        clusters[cent].append(d_point)
    for i, cluster in enumerate(clusters):
        reshaped = np.reshape(clusters[cluster], (len(clusters[cluster]), 2))
        centroids[i][0] = sum(reshaped[0:, 0])/len(reshaped[0:, 0])
        centroids[i][1] = sum(reshaped[0:, 1])/len(reshaped[0:, 1])
    print('Centroids:' + str(centroids))
    return dpCen, clusters

#iteration loop
for number in range(0, i):
    sort = sortCluster()
    cluster = pd.DataFrame({'Birth Rate': x[0][0:, 0], 'Life Expectancy': x[0][0:, 1], 'label': sort[0], 'Country': x[1]})
    group = cluster[['Country', 'Birth Rate', 'Life Expectancy', 'label']].groupby('label')
    print("Next iteration")
    print("Countries: \n" + str(group.count()))
    print("List: \n", list(group))
    print("Avg: \n", str(cluster.groupby(['label']).mean()))
    mean = sort[1]
    means = {}
    for clst in range(0, k):
        means[clst+1] = []
    for index, data in enumerate(mean):
        array = np.array(mean[data])
        array = np.reshape(array, (len(array), 2))
        birth = sum(array[0:, 0])/len(array[0:, 0])
        life = sum(array[0:, 1])/len(array[0:, 1])
        for dp in array:
            distance = sqrt((birth - dp[0])**2+(life - dp[1])**2)
            means[index+1].append(distance)
    tot = []
    for ind, summed in enumerate(means):
        tot.append(sum(means[ind+1]))
    print("Distance summed: " + str(sum(tot)))
    facet = sns.lmplot(data=cluster, x='Birth Rate', y='Life Expectancy', hue='label', fit_reg=False, legend=False)
    centr = np.reshape(centroids, (k, 2))
    plt.xlabel('Birthrate')
    plt.ylabel('Life Expectancy')
    plt.plot(centr[0:, 0], centr[0:, 1], linewidth=0, marker="X", c='#222222')
    plt.title('Iteration: ' + str(number+1) + "\nDistance summed: " + str(round(sum(tot), 0)))
    plt.show()
