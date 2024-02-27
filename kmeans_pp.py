import mykmeanssp as KM
import numpy as np
import pandas as pd
import math
import sys


class Point:
    def __init__(self, coords, dim, cluster):
        self.coords = coords
        self.dim = dim
        self.cluster = cluster

def main():
    np.random.seed(0)
    DEF_MAX_ITER = 300

    argc = len(sys.argv)
    if(argc == 5): #Iter not given
        iter = DEF_MAX_ITER       
        f1 = pd.read_csv(sys.argv[3])
        f2 = pd.read_csv(sys.argv[4])
    elif(argc == 6): #Iter given
        f1 = pd.read_csv(sys.argv[4])
        f2 = pd.read_csv(sys.argv[5])    
    else:
        print("An Error Has Occurred")
        exit(1)

    f1.set_index(0, inplace=True)
    f2.set_index(0, inplace=True)

    print("F1:")
    print(f1)
    print("F2:")
    print(f2)
    N = len(f1) #Assume f1,f2 have the same length

    if((not isInt(sys.argv[1])) or (not 1<int(sys.argv[1])) or (not int(sys.argv[1]) < N)):
        print("Invalid number of clusters!")
        exit(1)
    K = int(sys.argv[1])

    if(argc == 5):
        eps = float(sys.argv[2])
    elif(argc == 6): #iter is given
        if((not isInt(sys.argv[2])) or (not 1<int(sys.argv[2])) or (not int(sys.argv[2])<1000)):
            print("Invalid maximum iteration!")
            exit(1)
        iter = int(sys.argv[2])
        eps = float(sys.argv[3])

    file = pd.merge(f1,f2, key=f1.columns[0] ,how="inner")
    _,d = pd.shape(file) #N is already set to correct value, set d

    file = file.sort_values(by=file.columns[0])

    data = [Point([0]*d,d,-1)]*N # Initialize data array to default values

    for i in range(N):
        line = file.readline()
        args = line.split(",")
        data[i]= Point(list(map(float,args)), d, -1) #map function applies float() to each element of args, then turn it to a list using list()

    cents = INIT_CENTS(data, d, K)
    
    for i in cents:
       print(findPinARR(i, data), end = ',')
    print()

    mat = KM.fit(K,N,d,iter,eps,data,cents);
    
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

def isInt(inp):
    flag = True
    try:
        int(inp)
    except ValueError:
        flag = False
    return flag

if __name__ == '__main__':
    main()