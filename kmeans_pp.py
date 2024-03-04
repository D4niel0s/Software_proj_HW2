import mykmeanssp as KM
import numpy as np
import pandas as pd
import math
import sys


class Point:
    def __init__(self, coords, dim, cluster, index):
        self.coords = coords
        self.dim = dim
        self.cluster = cluster
        self.index = index

def main():
    np.random.seed(0)
    DEF_MAX_ITER = 300

    argc = len(sys.argv)
    if(argc == 5): #Iter not given
        eps = float(sys.argv[2])
        iter = DEF_MAX_ITER       
        f1 = pd.read_csv(sys.argv[3], header=None)
        f2 = pd.read_csv(sys.argv[4], header=None)
    elif(argc == 6): #Iter given
        if((not isInt(sys.argv[2])) or (not 1<int(sys.argv[2])) or (not int(sys.argv[2])<1000)):
            print("Invalid maximum iteration!")
            exit(1)
        iter = int(sys.argv[2])
        eps = float(sys.argv[3])
        f1 = pd.read_csv(sys.argv[4], header=None)
        f2 = pd.read_csv(sys.argv[5], header=None)    
    else:
        print("An Error Has Occurred")
        exit(1)

    file = pd.merge(f1,f2, on=f1.columns[0] ,how="inner",sort=True) #Also sorts
    #When using this technique, i still have the 'keys' column which should be ignored
    N,d = file.shape #Set N and d
    d -= 1 #Ignore keys

    if((not isInt(sys.argv[1])) or (not 1<int(sys.argv[1])) or (not int(sys.argv[1]) < N)):
        print("Invalid number of clusters!")
        exit(1)
    K = int(sys.argv[1])

    #Convert dataframe to matrix of values
    file = list(file.to_numpy())
    for i in file:
        i = list(i)

    #Initialize data to our data in file
    data = [Point([0]*d,d,-1,-1)]*N #Initialize data array to default values
    for i in range(len(file)):
        data[i]= Point(list(map(float,(file[i])[1:])), d, (file[i])[0]) #map function applies float() to each element of i, then turn it to a list using list()

    cents = INIT_CENTS(data, d, K) #Initialize centroids using Kmeans++ algorithm
    
    #Print calculated centroids (Their indices)
    for i in range(len(cents)):
        print(cents[i].index,end='')
        if(i==len(cents)-1):
            break
        print(",",end='')
    print()
    
    #Perform K-Means clustering using my module
    mat = KM.fit(K,N,d,iter,eps,data,cents);
    
    #Print output from clustering (Output from fit() is given in a matrix where each row is a centroid)
    for i in mat:
        for j in range(len(i)):
            print("%.4f" % i[j], end='')
            if(j==len(i)-1):
                break
            print(",",end='')
        print()
        
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