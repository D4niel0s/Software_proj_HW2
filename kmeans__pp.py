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
       print(findPinARR(i, data))
    

def findPinARR(x, arr):
    for i in range(len(arr)):
        flag = True
        for j in range(len(arr[i].coords)):
            if((arr[i].coords)[j] != x.coords[j]):
                flag = False
            
        if(flag == True):
            return i
    return -1
        


def INIT_CENTS(dp, d, k):
    dataclone = dp.copy()
    cents = []
    prob = []

    cents.append(np.random.choice(dataclone))
    dataclone.remove(cents[0])
    
    for i in range(1,k):
        prob = computeNewProb(cents, dataclone, d)
        cents.append(np.random.choice(dataclone,p=prob))
        dataclone.remove(cents[i])

    return cents


def computeNewProb(cents, dp, dim):
    DistArr = [0]*len(dp)
    Distsum = 0
    prob = [0]*len(dp)

    for i in range(len(dp)):
        _, DistArr[i] = FindClosestCentroid(dp[i], cents, dim);
        Distsum += DistArr[i]
    
    for i in range(len(dp)):
        prob[i] = float(DistArr[i])/float(Distsum)

    return prob

def FindClosestCentroid(x, centroids, dim):
    assigned = 0
    minDist = dist(x, centroids[0], dim)
    for i in range(len(centroids)):
        curDist = dist(x, centroids[i], dim)
        if(curDist < minDist):
            minDist = curDist
            assigned = i
    
    return (assigned, minDist)


def dist(x,y, dim):
    dist = 0
    for i in range(dim):
        dist += pow(x.coords[i] - y.coords[i], 2)

    dist = math.sqrt(dist)
    return dist

if __name__ == '__main__':
    main()