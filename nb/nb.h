#ifndef __NB_H_
#define __NB_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NB_CLASSES		2 
#define NB_FEATURES  	8
#define TRAINING_SIZE	514
//#define SIZEOF_Y		1380
#define TEST_SIZE		254

//#define M_PI acos(-1.0)

/** FUNCTIONS **/
void print_tab_d(double *tab, unsigned count);
void print_tab(unsigned *tab, unsigned count);
void separateByClass(unsigned y[TRAINING_SIZE], unsigned int *class, unsigned *ind_class_0, unsigned *ind_class_1);
double mean(double v[], unsigned size);
double variance(double v[], unsigned size);
void summarize_dataset(double x[][NB_FEATURES], double *m, double *var);
void summarizeByClass(double x[][NB_FEATURES], unsigned y[TRAINING_SIZE], double *m, double *var, double *priors);
double calculate_probability(double x, double mean, double var);
void calculateClassProbabilities(double *means, double *variances, double *priors, double *v, double *probabilities);
void predict(double means[NB_CLASSES*NB_FEATURES], double variances[NB_CLASSES*NB_FEATURES], double priors[NB_CLASSES], double v[NB_FEATURES], double *best_probability, unsigned *index);
void getPredictions(double means[NB_CLASSES*NB_FEATURES], double variances[NB_CLASSES*NB_FEATURES], double priors[NB_CLASSES], double x_test[TEST_SIZE][NB_FEATURES], unsigned predictions[TEST_SIZE]);

#endif /* __NB_H_*/
