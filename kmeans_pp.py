import kmeans_module as KM
import numpy as np
import math


class Point:
    def __init__(self, coords, dim, cluster):
        self.coords = coords
        self.dim = dim
        self.cluster = cluster

def main():
    np.random.seed(0)
    file = open("t.txt",'r')

    data = [Point([0]*4,4,-1)]*100 # Initialize data array to default values

    for i in range(100):
        line = file.readline()
        args = line.split(",")
        data[i]= Point(list(map(float,args)), 4, -1) #map function applies float() to each element of args, then turn it to a list using list()

    cents = INIT_CENTS(data, 4, 3)
    
    for i in cents:
       print(findPinARR(i, data), end = ',')
    print()

    mat = KM.kmeans(3,100,4,1000,0.01,data,cents);

    for i in mat:
        for j in i:
            print("%.4f" % j, end=',')
        print()

#Returns the index of a Point in a given Point array, returns -1 if not found
def findPinARR(x, arr):
    for i in range(len(arr)):
        flag = True
        for j in range(len(arr[i].coords)):
            if((arr[i].coords)[j] != x.coords[j]):
                flag = False
            
        if(flag == True):
            return i
    return -1

#Initializes centroids from datapoints, returns centroids
def INIT_CENTS(dp, d, k):
    cents = []
    prob = []

    cents.append(np.random.choice(dp)) #First centroid is chosen uniformly from datapoints
    
    for i in range(1,k):
        prob = computeNewProb(cents, dp, d) #Update distribution
        cents.append(np.random.choice(dp,p=prob)) #Draw a new centroid according to computed distribution

    return cents

#Computes weighted probabily distribution among points - according to distance from other centroids
def computeNewProb(cents, dp, dim):
    DistArr = [0]*len(dp)
    Distsum = 0
    prob = [0]*len(dp)

    for i in range(len(dp)):
        _, DistArr[i] = FindClosestCentroid(dp[i], cents, dim) #Compute distance from closest centroid
        Distsum += DistArr[i]
    
    for i in range(len(dp)):
        prob[i] = float(DistArr[i])/float(Distsum)

    return prob

#Changed this function from the last Homework - now returns a tuple including assigned cluster as well as the min distance from it
def FindClosestCentroid(x, centroids, dim):
    assigned = 0
    minDist = dist(x, centroids[0], dim)
    for i in range(len(centroids)):
        curDist = dist(x, centroids[i], dim)
        if(curDist < minDist):
            minDist = curDist
            assigned = i
    
    return (assigned, minDist)

#Compute distance between two points
#"Borrowed" from my last Homework.
def dist(x,y, dim):
    dist = 0
    for i in range(dim):
        dist += pow(x.coords[i] - y.coords[i], 2)

    dist = math.sqrt(dist)
    return dist

if __name__ == '__main__':
    main()