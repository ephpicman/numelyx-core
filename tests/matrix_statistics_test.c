/**
 * @file matrix_statistics_test.c
 * @brief Native tests for Matrix descriptive statistics, variance, and standard deviation.
 *
 * @details These native tests define regression coverage for the named Matrix
 * contract. Assertions are executable specifications: intentional behavior
 * changes should update these checks together with the corresponding API
 * documentation. This file is test support, not runtime code.
 */

#include "../numerus_matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
static void test_population_sample(void) {
 const double values[]={1,2,3,4}; numerus_matrix *m=NULL; double value=-1;
 assert(numerus_matrix_create_dense(2,2,values,&m)==NUMERUS_MATRIX_SUCCESS);
 assert(numerus_matrix_variance(m,false,&value)==NUMERUS_MATRIX_SUCCESS);
 assert(fabs(value-1.25)<1e-12);
 assert(numerus_matrix_standard_deviation(m,false,&value)==NUMERUS_MATRIX_SUCCESS);
 assert(fabs(value-sqrt(1.25))<1e-12);
 assert(numerus_matrix_variance(m,true,&value)==NUMERUS_MATRIX_SUCCESS);
 assert(fabs(value-5.0/3.0)<1e-12);
 assert(numerus_matrix_standard_deviation(m,true,&value)==NUMERUS_MATRIX_SUCCESS);
 assert(fabs(value-sqrt(5.0/3.0))<1e-12); numerus_matrix_destroy(m);
}
static void test_failures(void) {
 const double values[]={1,NAN}, one[]={4}; numerus_matrix *m=NULL,*single=NULL;
 double value=123.0;
 assert(numerus_matrix_create_dense(1,2,values,&m)==NUMERUS_MATRIX_SUCCESS);
 assert(numerus_matrix_variance(m,false,&value)==NUMERUS_MATRIX_NON_FINITE);
 assert(value==123.0);
 assert(numerus_matrix_create_dense(1,1,one,&single)==NUMERUS_MATRIX_SUCCESS);
 assert(numerus_matrix_variance(single,true,&value)==NUMERUS_MATRIX_INVALID_ARGUMENT);
 assert(value==123.0);
 assert(numerus_matrix_standard_deviation(NULL,false,&value)==NUMERUS_MATRIX_INVALID_ARGUMENT);
 assert(value==123.0);
 assert(numerus_matrix_variance(m,false,NULL)==NUMERUS_MATRIX_INVALID_ARGUMENT);
 numerus_matrix_destroy(single); numerus_matrix_destroy(m);
}
int main(void){test_population_sample();test_failures();puts("Matrix variance/stddev tests passed.");return 0;}
