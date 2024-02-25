#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

typedef struct point{
    double *coords; /*Coordinates of the vector will be inserted here. (memory allocation in main)*/
    int dim; /*Dimension.*/

    int cluster; /*Cluster the point belongs to.*/
}Point;

void KMeans(int, int, int, int,Point *, Point *, short *, double);
int FindClosestCentroid(Point, Point *, int);
double dist(Point, Point);
void ADD(Point, Point);
void MULT(Point, double);

#define DEF_MAX_ITER 200

/*K clusters, N points, d dimension, iter iterations, data - the points, centroids - the centroids .
//Assumes data is valid and allocated.
//Assumes centroids size matches clusters number, overrides centroids' data.
//Note: If we want the clustering, we can check for each point which cluster is the closest (outside the function)*/
void KMeans(int K, int N, int d, int iter, Point *data, Point *centroids, short *ERR_FLAG, double Epsilon){
    int i; int j; int iterations = 0;
    short END_FLAG = 0;
    
    Point *KMEANS = (Point *)malloc(sizeof(Point) * K); /*The mean (each cluster is represented by index)*/
    Point *PREV_Centroids = (Point *)malloc(sizeof(Point) * K); /*The previous mean (each cluster is represented by index)*/
    double *Delta_vector = (double *)malloc(sizeof(double) * K); /*the differences between previous and current centroids.*/
    int *PtCounter = (int *)malloc(sizeof(int) * K); /*Number of points (each cluster is represented by index)*/

    if(KMEANS == NULL || PREV_Centroids == NULL || Delta_vector == NULL || PtCounter == NULL){
        fprintf(stderr, "An Error Has Occurred\n");
        *ERR_FLAG = 1;
        goto FREE;
    }

    /*Init KMEANS and PREV_Centroids*/
    for(i=0; i<K; ++i){
        KMEANS[i].coords = (double *)calloc(sizeof(double) ,d); /*Init to zero.*/
        PREV_Centroids[i].coords = (double *)malloc(sizeof(double) * d); /*Same here - no need to zero.*/

        if(KMEANS[i].coords == NULL || PREV_Centroids[i].coords == NULL){
            fprintf(stderr, "An Error Has Occurred\n");
            *ERR_FLAG = 1;
            goto FREE;
        }

        KMEANS[i].dim = d;
        PREV_Centroids[i].dim = d;
        
        KMEANS[i].cluster = -1;
        PREV_Centroids[i].cluster = -1;
    }

    /*Init PREV_centroids to be centroids*/
    for(i=0; i<K; ++i){
        for(j=0; j<d; ++j){ /*initialize coords*/
            (PREV_Centroids[i].coords)[j] = (centroids[i].coords)[j];
        }
        PREV_Centroids[i].dim = d;
        PREV_Centroids[i].cluster = -1; /*The centroids aren't a part of a cluster.*/
    }
    
    do{
        /*Init means and No. of points to zero.*/
        for(i=0; i<K; ++i){
            PtCounter[i] = 0;

            for(j=0; j<d; ++j){
                (KMEANS[i].coords)[j] = 0; /*Although we start with zeroed KMEANS, we need this line for next iteration(s).*/
            }
        }

        /*Assignment stage*/
        for(i=0; i<N; ++i){ 
            data[i].cluster = FindClosestCentroid(data[i], centroids, K);/*Categorize points to clusters*/
            ADD(KMEANS[data[i].cluster], data[i]); /*Add points to according sums*/
            PtCounter[data[i].cluster]++; /*Add to counter for according cluster*/
        }
    
        for(i=0; i<K; ++i){
            MULT(KMEANS[i], (1.0/(double)PtCounter[i])); /*Normalize the summations, creating the means*/

            for(j=0; j<d; ++j){ /*Update centroids.*/
                (centroids[i].coords)[j] = (KMEANS[i].coords)[j];
            }
        }

        /*Update delta_vector and PREV_Centroids*/
        for(i=0; i<K; ++i){
            Delta_vector[i] = dist(centroids[i], PREV_Centroids[i]);
            for(j=0; j<d; ++j){
                (PREV_Centroids[i].coords)[j] = (centroids[i].coords)[j];
            }
        }
        /*Check Delta < Epsilon (shouldn't be checked on first iteration)*/
        if(iterations > 0){
            END_FLAG = 1;
            for(i=0; i<K; ++i){
                if(Delta_vector[i] >= Epsilon){
                    END_FLAG = 0;
                }
            }
        }

        iterations++;
    }while(iterations < iter && END_FLAG == 0);

    FREE: /*Label here for jumping to free if allocation(s) fail.*/
    /*Free memory allocated for local variables*/
    for(i=0; i<K; ++i){
        free(KMEANS[i].coords);
        free(PREV_Centroids[i].coords);
    }
    free(KMEANS);
    free(PREV_Centroids);
    free(Delta_vector);
    free(PtCounter);
    /*Centroids is not freed since it's the "Output" of the KMeans function (Should be freed outside)*/
}


static PyObject* kmeans_c(PyObject *self, PyObject *args){
    /*NEED TO ADD PYTHON HANDLING OF FUNCTION KMeans()*/
}



static PyMethodDef kmeans_methods[] = {
    {"kmeans",(PyFunction) KMeans, METH_VARARGS, PyDoc_STR("Implementation of kmeans algorithm in C!")},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef KmeansModule = {
    PyModuleDef_HEAD_INIT,"kmeans_module",NULL, -1, kmeans_methods
};


/*Finds closest centroid (denoted by cluster number) to a given point*/
int FindClosestCentroid(Point x, Point *centroids, int K){
    int i;
    int assigned = 0;
    double mindist = dist(x,centroids[0]);
    double curdist;
    for(i=0; i<K; ++i){
        curdist = dist(x,centroids[i]);
        if(curdist < mindist){
            mindist = curdist;
            assigned = i;
        }
    }
    return assigned;
}


/*Computes distance between two given points.
//Assumes dimensions match.*/
double dist(Point a,Point b){
    double dist = 0;
    int i;
    
    for(i=0; i<a.dim; ++i){
        dist += pow(a.coords[i] - b.coords[i] , 2);
    }
    dist = sqrt(dist);

    return dist;
}

/*adds b to a (changes a)*/
void ADD(Point a, Point b){
    int i;
    for(i=0; i<a.dim; ++i){
        a.coords[i] += b.coords[i];
    }
}

/*multiplies a by scalar x (changes a)*/
void MULT(Point a, double x){
    int i;
    for(i=0; i<a.dim; ++i){
        a.coords[i] *= x;
    }
}