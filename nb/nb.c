/**
 * AUTHOR: MAW
* DATE: 11/12/2017 - 13/12/2017
* TESTED: 08/01/2018 - 09/01/2018
* COMMENTS: This file contains a simple implementation in C of LR algorithm. 
* The important functions to look in order to understand the code are : 
* the LR and main functions. 
**/
#include "nb.h"

struct class_s{
	unsigned int count;
	unsigned int label;
	unsigned int ind[TRAINING_SIZE]; // in practise, it will be less but ..
} class;


//struct class_s classes[];

void print_tab_d(double *tab, unsigned count){
	unsigned i;
	for (i = 0; i < count; i++){
		printf("tab[%u] = %lf\n", i, tab[i]);
	}
}

void print_tab(unsigned *tab, unsigned count){
	unsigned i; 
	for (i = 0; i < count; i++){
		printf("tab[%u] = %u\n", i, tab[i]);
	}
}

// X is a matrix of size TRAINING_SIZE x NB_FEATURES
void separateByClass(unsigned y[TRAINING_SIZE], unsigned int *class, unsigned *ind_class_0, unsigned *ind_class_1){
	unsigned i;
	unsigned count_0 = 0, count_1 = 0;	
	for (i = 0; i < TRAINING_SIZE; i++){
		// Binary classifier so only two classes
		if (y[i] == 0) {
			// store indexes for label 0
			ind_class_0[count_0] = i;
			// count the ²number of occurences of label 0
			count_0++;
		}
		else { 
			// store indexes for label 1
			ind_class_1[count_1] = i;
			// count the number of occurences of label 1
			count_1++;
		}
	}	
	class[0] = count_0;
	class[1] = count_1;	
	// printf("count 0 %u\n", count_0);
	// printf("count 1 %u\n", count_1);
	// printf("indices 0\n");
	// for (i = 0; i < count_0; i++)
	// 	printf("%d ",ind_class_0[i]);
	// printf("\nindices 1\n");
	// for (i = 0; i < count_0; i++)
	// 	printf("%d ",ind_class_1[i]);
	// printf("\nEnd separateByClass\n");
}

double mean(double v[], unsigned size){
	unsigned int i; 
	double temp_mean = 0.0;
	// printf("\nMean\n");
	for (i = 0; i < size; i++){
		// printf("v[%d]: %lf ", i, v[i]);
		temp_mean += v[i];
	}
	temp_mean /= size;
	// printf("End mean: ret %lf\n", temp_mean);	
	return temp_mean;
}

double variance(double v[], unsigned size){
	unsigned int i; 
	double temp_mean;
	temp_mean = mean(v,size);
	double temp_var = 0.0;
	for (i = 0; i < size; i++){
		temp_var += pow((v[i] - temp_mean),2);
	}
	temp_var /= (size-1);
	return temp_var;
}

void summarize_dataset(double x[][NB_FEATURES], double *m, double *var){
	double x_col[TRAINING_SIZE];
	double mu[NB_FEATURES];
	unsigned i, j;
	for (j = 0; j < NB_FEATURES; j++){
		for (i = 0; i < TRAINING_SIZE; i++){
			x_col[i] = x[i][j];
		}
		m[j] = mean(x_col,TRAINING_SIZE);
		var[j] = variance(x_col,TRAINING_SIZE);
		mu[j] = sqrt(var[j]);
	}
	// printf("MEAN (summarize_dataset)\n" );
	// print_tab_d(m, NB_FEATURES);
	// printf("VARIANCE\n" );
	// print_tab_d(var, NB_FEATURES);
	// printf("MU\n" );
	// print_tab_d(mu, NB_FEATURES);
}

void summarizeByClass(double x[][NB_FEATURES], unsigned y[TRAINING_SIZE], double *m, double *var, double *priors){
	unsigned i = 0, j, k = 0, l = 0;
	unsigned int count = 0; 
	unsigned temp = 0;
	unsigned int classes[NB_CLASSES];
	unsigned int indices_0[TRAINING_SIZE], indices_1[TRAINING_SIZE];	
	unsigned int class, *indices;
	static double x_col[NB_FEATURES];

	separateByClass(y, classes, indices_0, indices_1);
	printf("class_0 = %u\n", classes[0]);
	printf("class_1 = %u\n", classes[1]);
	//double *col_temp = (double *)x_col;	
	//printf("x[k][l] = %u\n",x[5][2]);
	for (j = 0; j < NB_CLASSES; j++){
		class = classes[j];
		// printf("class: %u\n",class);
		// determine right indexes 
		indices = (j == 0) ? (unsigned *) indices_0 : 
							 (unsigned *) indices_1 ;
		for (l = 0; l < NB_FEATURES; l++){
			temp = l;
			for (i = 0; i < class; i++){
				k = indices[i];
				// printf("k: %u\t",k);
				x_col[i] = x[k][temp];
				// printf("x[%d][%d] = %lf\n", k, temp, x_col[i]);
			}
			// The first NB_FEATURES values correspond to mean of features for label 0
			// The next NB_FEATURES values correspond to mean of features for label 1
			// m and var size is 2 times NB_FEATURES		
			m[count] = mean(x_col,class);
			// printf("m[%d] = %lf\n", count, m[count]);			
			var[count] = variance(x_col,class);
			// printf("var[%d] = %lf\n", count, var[count]);
			count++;
		}
		priors[j] = class/(double)TRAINING_SIZE;
		// printf("class = %u, prior = %lf\n", class, priors[j]);
	}

	printf("summarizeByClass \nMeans \t\t Variances\n");
	for (j = 0; j < NB_CLASSES*NB_FEATURES; j++){
		printf("%lf \t\t %lf\n", m[j], var[j]);
	}
	// print_tab_d(means,NB_CLASSES*NB_FEATURES);
	
	// printf("\nVariance (summarizeByClass) \n");	
	// print_tab_d(variances,NB_CLASSES*NB_FEATURES);

	// printf("\nmeans %p\n", means);
	// printf("variances %p\n", variances);
}

double calculate_probability(double x, double mean, double var){
	double coeff = 0., eps = 0., exponent = 0.;
	eps = 1e-4; // To avoid division by 0 
	coeff = 1.0/sqrt(2*M_PI*var + eps);
	exponent = exp(-(pow((x-mean),2)/(2*var + eps)));
	//printf("x = %lf\n", x);
	return (coeff * exponent);
}

void calculateClassProbabilities(double *means, double *variances, double *priors, double *v, double *probabilities){
	unsigned int i, j;
	double mean = 0.0;
	double variance = 0.0, x;
	unsigned count = 0;
	double temp;	
//	double *probabilities = (double *)calloc(NB_CLASSES, sizeof(double));
	
	for (j = 0 ; j < NB_CLASSES; j++){
		probabilities[j] = 1;//priors[j];
		//printf("count = %u, \t temp = %lf, \t probability = %lf\n", count, temp, probabilities[j]);
		for (i = 0; i < NB_FEATURES; i++){
			mean = means[count];
			variance = variances[count];	
			x = v[i];
			temp = calculate_probability(x,mean,variance);
			probabilities[j] = probabilities[j]*temp;			
			// printf("x = %lf, mean = %lf, \t variance = %lf\n", x, mean, variance);
			// printf("count = %u, \t temp = %lf, \t probability = %g\n", count, temp, probabilities[j]);
			count++;			
		}
		// if (probabilities[j]>=1)
			//printf("count = %u, \t probability = %lf\n", count, probabilities[j]);
	}
	//printf("probabilities\n\n");
	//print_tab_d(probabilities,NB_CLASSES);
}

void predict(double means[NB_CLASSES*NB_FEATURES], double variances[NB_CLASSES*NB_FEATURES], double priors[NB_CLASSES], double v[NB_FEATURES], double *best_probability, unsigned *index){
	unsigned int i, j;
	static unsigned bestIndex;
	static double bestProb;
	static double probabilities[NB_CLASSES];
	double prob;	

	calculateClassProbabilities(means,variances,priors,v,probabilities);
	bestProb = 0.0;
	bestIndex = 0;
	for (i = 0; i < NB_CLASSES; i++){
		prob = probabilities[i];
		if (prob > bestProb){
			bestProb = prob;
			bestIndex = i;
		}
		// if (prob >= 1.0)
		// 	printf("\nWAOUH %lf\n", prob);
		// printf("best_probability = %g\n", bestProb);
		// printf("best_index = %u\n", bestIndex);
		// printf("prob = %lf\n", prob);
	}
	*best_probability = bestProb;
	*index = bestIndex;
}

void getPredictions(double means[NB_CLASSES*NB_FEATURES], double variances[NB_CLASSES*NB_FEATURES], double priors[NB_CLASSES], double x_test[TEST_SIZE][NB_FEATURES], unsigned predictions[TEST_SIZE]){
	double best_probability = 0., prob;
	unsigned bestIndex=0;
	double x_row[NB_FEATURES];
	double probabilities[NB_CLASSES];
	//double *probabilities = (double *)temp_probabilities;
	unsigned i,j,k;
	for (i = 0; i < TEST_SIZE; i++){
		best_probability = 0.;
		for (j = 0; j < NB_FEATURES; j++){
			x_row[j] = x_test[i][j];
		}
		predict(means, variances, priors, x_row, &best_probability, &bestIndex);
		predictions[i] = bestIndex;
	}
}
