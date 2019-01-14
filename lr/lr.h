#ifndef __LR_H_
#define __LR_H_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WEIGHTS_LENGTH 	2
#define SIZEOF_Y		67
#define NB_FEATURES 	2
#define FEATURE_SIZE	67
#define TEST_SIZE		33


// #define WEIGHTS_LENGTH 	7
// #define SIZEOF_Y		456601
// #define NB_FEATURES 	7
// #define FEATURE_SIZE	456601
// #define TEST_SIZE		50734


double sigmoid(double z);
double classify_o(double *x, double *weights);
void classify(double x[TEST_SIZE][NB_FEATURES], double weights[NB_FEATURES], double output[TEST_SIZE]);
double Cost_Function_Derivative(double x[FEATURE_SIZE][NB_FEATURES], unsigned int y[FEATURE_SIZE], double theta[NB_FEATURES], unsigned int j, unsigned int m, double alpha);
void Gradient_Descent(double x[][NB_FEATURES], unsigned int y[], double theta[NB_FEATURES], unsigned int m, double alpha, double* new_weights[NB_FEATURES]);
void Logistic_regression(double x[][NB_FEATURES], unsigned int y[], double alpha, double theta[NB_FEATURES], unsigned int num_iters, double **weights);
void print_matrix(double tab[NB_FEATURES]);
#endif //__LR_H_