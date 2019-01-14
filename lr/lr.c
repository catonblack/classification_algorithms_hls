/**
* AUTHOR: MAW
* DATE: 11/12/2017
* COMMENTS: This file contains a simple implementation in C of LR algorithm. 
* The important functions to look in order to understand the code are : 
* the LR and main functions. 
**/
#include "lr.h"

/** 
* This function computes the sigmoid function. 
**/
double sigmoid(double z){
	return (double)1./((double)(1. + exp(-z)));
}

/**
* This function takes a the test matrix of size TEST_SIZE x FEATURE_SIZE size and returns sigmoid(sum(WEIGHTS.X[k,:]))
**/
void classify(double x[TEST_SIZE][NB_FEATURES], double weights[NB_FEATURES], double output[TEST_SIZE]){
	double temp = .0;
	unsigned int i, k;
	double w, xtemp, prod;

	for (k = 0; k < TEST_SIZE; k++){
		temp = .0;
		for (i = 0; i < NB_FEATURES; i++){
			xtemp = x[k][i];
			w = weights[i];
			prod = w*xtemp;
			temp = temp + prod;
		}
		output[k] = sigmoid(temp);
	}
}

// Function needed for training
/**
* This function takes a feature vector of FEATURE_SIZE size and returns sigmoid(sum(WEIGHTS.X))
**/
double classify_o(double x[NB_FEATURES], double weights[NB_FEATURES]){
	double temp = .0;
	unsigned int i;
	for (i = 0; i < NB_FEATURES; i++)
			temp += weights[i]*x[i];
	return sigmoid(temp);
}


/** 
* This function creates the gradient component for each Theta value.
* x is a matrix containing feature vectors. Its size is FEATURE_SIZE (rows) x NB_FEATURES (columns).
* y is a vector contaning labels associated to a row of matrix x. Its size is FEATURE_SIZE.
**/
double Cost_Function_Derivative(double x[FEATURE_SIZE][NB_FEATURES], unsigned int y[FEATURE_SIZE], double theta[NB_FEATURES], unsigned int j, unsigned int m, double alpha){
	double ret; 
	double error, sum_errors = .0;
	unsigned int i,jj;
	double xij, hi; 
	double x_r[FEATURE_SIZE]; 	
	double constant;
	static unsigned count;

	for (i = 0; i < m; i++){
		// printf("in CFD");
		// Store the row in x_r
		for (jj = 0; jj < NB_FEATURES; jj++)
			x_r[jj] = x[i][jj], printf("x_r[%d] = %lf ",jj,x_r[jj]);;//, printf("x[%u][%u] = %f\n",i,jj,x[i][jj] ); 
		printf("\n count = %d \n", count++);
		xij = x[i][j];
		hi = classify_o(x_r, theta);
		printf("hi %lf\n", hi);
		error = (hi - y[i])*xij;
		sum_errors += error;
	}
	m = SIZEOF_Y;//sizeof(y)/sizeof(unsigned int);
	constant = (double)alpha/(double)m;
	ret = constant * sum_errors;
	return ret;
}

/**
* This function computes the partial differential for each theta. 
**/
void Gradient_Descent(double x[][NB_FEATURES], unsigned int y[FEATURE_SIZE], double theta[NB_FEATURES], unsigned int m, double alpha, double* new_weights[NB_FEATURES]){
	//double *new_theta = malloc (sizeof (double) * NB_FEATURES);
	static double new_theta[NB_FEATURES];
	double CFDerivative, new_theta_value;
	unsigned int j;
	for (j = 0; j < NB_FEATURES; j++){
		// printf("in GD");
		CFDerivative = Cost_Function_Derivative(x,y,theta,j,m,alpha);
		printf("CFDerivative = %lf\n", CFDerivative);
		new_theta_value = theta[j] - CFDerivative;
		new_theta[j] = new_theta_value;
	}
	*new_weights = new_theta;
}

/**
* This function implements the LR algorithm. 
* It is used to train a set and obtain a weight/theta vector 
* which is then used for computing the predicted values given a set of features.
**/
void Logistic_regression(double x[][NB_FEATURES], unsigned int y[], double alpha, double theta[NB_FEATURES], unsigned int num_iters, double **weights){
	unsigned int m = SIZEOF_Y; //sizeof(y)/sizeof(y[0]);
	unsigned int i,j; 
	double *new_theta[NB_FEATURES];
	//double *theta_v = malloc (sizeof (double) * NB_FEATURES);
	static double theta_v[NB_FEATURES];
	// Initializati
	for (j = 0; j < NB_FEATURES; j++)
		theta_v[j] = theta[j];
	// Computation
	for (i = 0; i < num_iters; i++){
		// printf("in LR");
		Gradient_Descent(x,y,theta_v,m,alpha,new_theta);
		for (j = 0; j < NB_FEATURES; j++)
			theta_v[j] = (*new_theta)[j];
		print_matrix(theta_v);
	}
	*weights = theta_v;
}

/**
* A utility function to print the matrix. 
**/
void print_matrix(double tab[NB_FEATURES]){
	unsigned i, j; 
	// for (i = 0; i < FEATURE_SIZE; i++)
		for (j = 0; j < NB_FEATURES; j++)
			printf("tab[%d] = %f\n", j, tab[j]);
}
