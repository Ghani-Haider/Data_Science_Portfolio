"""
CS 351 - Artificial Intelligence 
Assignment 3
Student 1(Name and ID):
Student 2(Name and ID):
"""
import math
# from math import pi
import random
from matplotlib import colors
from matplotlib.colors import cnames
import numpy as np
import matplotlib.pyplot as plt

def initializePoints(count):
    points = []
    for i in range(int(count/3)):
        points.append([random.gauss(0,10),random.gauss(100,10)])
    for i in range(int(count/3)):
        points.append([random.gauss(-30,20),random.gauss(10,10)])
    for i in range(int(count/3)):
        points.append([random.gauss(30,20),random.gauss(10,10)])

    return points

def initializePoints_random(count):
    points = []
    for i in range(int(count/3)):
        points.append([random.gauss(-25,20),random.gauss(0,20)])
    for i in range(int(count/3)):
        points.append([random.gauss(0,40),random.gauss(80,10)])
    for i in range(int(count/3)):
        points.append([random.gauss(20,20),random.gauss(150,20)])

    return points


def euc_dist(point, centroid):
    return(math.sqrt((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2 ))

def check(past,current):
    count=0
    t_keys=0
    for keys in past:
        t_keys+=1
        for t in current:
            if euc_dist(keys,t)<0.75:
                count+=1
                break
    if count==t_keys:
        return False
    return True

def cluster(points,K,visuals = True):
    clusters=[]

    #Your kmeans code will go here to cluster given points in K clsuters. If visuals = True, the code will also plot graphs to show the current state of clustering
    # plot iteration 0
    if(visuals):
        x, y = zip(*points)
        plt.scatter(x, y, color='black', marker="+")
        plt.title("Iteration 0")
        plt.show()

    centroids = dict()
    past_centroids = dict()
    
    iterations = 0
    if (iterations == 0):
        for i in range(K):
            temp = points[random.randrange(0, len(points))]
            centroids[(temp[0], temp[1])] = []
    
    # past_centroids = centroids
    for keys in centroids:
        past_centroids[(random.randrange(10000, 20000), random.randrange(10000, 20000))] = []
    
    # generating centroids (cluster centers)
    # cluster
    # check = 0
    while(check(past_centroids,centroids)==True): #check() != True):
        if (iterations != 0):
            past_centroids = centroids
            centroids = dict()
            for key in past_centroids:
                # get all x, y
                x = []
                y = []
                point_lst = past_centroids[key]
                for i in range(len(point_lst)):
                    x.append(point_lst[i][0])
                    y.append(point_lst[i][1])

                # print(list(zip(past_centroids[key]))[0][0])
                # mean_x = sum(list(zip(past_centroids[key]))[0][0]) / len(past_centroids[key])
                # mean_y = sum(list(zip(past_centroids[key]))[1][0]) / len(past_centroids[key])
                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)
                centroids[(mean_x, mean_y)] = []

        for point in range(len(points)):
            # print(point)
            min_dist = math.inf
            choosen_center = list(centroids.keys())[0] # randomly choosing first centroid
            for each_centroid in centroids:
                # print(each_centroid)
                curr_dist = euc_dist(points[point], each_centroid)
                if(curr_dist < min_dist):
                    choosen_center = each_centroid
                    min_dist = curr_dist
            centroids[choosen_center].append(points[point])

        iterations += 1

        # plotting
        if(visuals):
            clr = ['red','green','blue', 'pink']
            i = 0
            for centroid in centroids:
                # print(centroid)
                lst = centroids[centroid]
                # print("lst = ", lst)
                if(len(lst) != 0):
                    x, y = zip(*lst)
                    plt.scatter(x,y , color=clr[i], marker="+")
                    i += 1
                plt.scatter(centroid[0], centroid[1], color='black', marker="D")

            plt.title("Iteration "+str(iterations))
            plt.show()

    
    return centroids



def clusterQuality(clusters):
    score = -1 
    score_lst = []
    #Your code to compute the quality of cluster will go here.
    for cluster in clusters:
        # number_points = 0
        total_cluster_dist = 0
        for key in cluster.keys():
            # print(key)
            each_key_dist = 0
            points = cluster[key]
            # number_points += len(points)
            # print(points)
            for point in points:
                each_key_dist += (euc_dist(point, key))**2
            total_cluster_dist += each_key_dist
        score_lst.append(total_cluster_dist)

    score = min(score_lst)
    print("Score list is :", score_lst)
    return score
    

def keepClustering(points,K,N,visuals):
    clusters = []
    
    #Write you code to run clustering N times and return the formation having the best quality. 
    for n in range(N):
        clusters.append(cluster(points, K, visuals))
    
    return clusters
    



K = 3
N = 1

# points = initializePoints_random(1000)

points = initializePoints(1000)

clusters = keepClustering(points,K,N,True)
# print(clusters)

print ("The score of best Kmeans clustering is:", clusterQuality(clusters))

