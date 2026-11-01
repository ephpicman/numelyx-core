/**
 * @file matrix_normalize_test.c
 * @brief Native tests for global, row-wise, and column-wise Matrix normalization.
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
static void check(const numerus_matrix *m,size_t r,size_t c,double expected) {
 double actual=0; assert(numerus_matrix_get(m,r,c,&actual)==NUMERUS_MATRIX_SUCCESS);
 assert(fabs(actual-expected)<1e-10);
}
static void test_axes_and_norms(void) {
 const double values[]={3,4,1,2,0,1};
 numerus_matrix *m=NULL,*n=NULL;
 assert(numerus_matrix_create_dense(2,3,values,&m)==NUMERUS_MATRIX_SUCCESS);
 assert(numerus_matrix_create_normalized(m,NUMERUS_MATRIX_NORMALIZE_ROWS,NUMERUS_MATRIX_NORMALIZE_L2,&n)==NUMERUS_MATRIX_SUCCESS);
 check(n,0,0,3.0/sqrt(26.0)); check(n,0,1,4.0/sqrt(26.0)); check(n,1,0,2.0/sqrt(5.0));
 numerus_matrix_destroy(n); n=NULL;
 assert(numerus_matrix_create_normalized(m,NUMERUS_MATRIX_NORMALIZE_COLUMNS,NUMERUS_MATRIX_NORMALIZE_L1,&n)==NUMERUS_MATRIX_SUCCESS);
 check(n,0,0,0.6); check(n,0,1,1.0); check(n,1,0,0.4); check(n,1,1,0.0); check(n,0,2,0.5); check(n,1,2,0.5);
 numerus_matrix_destroy(n); n=NULL;
 assert(numerus_matrix_create_normalized(m,NUMERUS_MATRIX_NORMALIZE_WHOLE,NUMERUS_MATRIX_NORMALIZE_INFINITY,&n)==NUMERUS_MATRIX_SUCCESS);
 check(n,0,0,0.75); check(n,0,1,1.0); check(n,1,0,0.5);
 numerus_matrix_destroy(n); numerus_matrix_destroy(m);
}
static void test_zero_and_nonfinite(void) {
 const double zero[]={0,0}; const double bad[]={1,NAN};
 numerus_matrix *m=NULL,*n=(numerus_matrix *)1;
 assert(numerus_matrix_create_dense(1,2,zero,&m)==NUMERUS_MATRIX_SUCCESS);
 assert(numerus_matrix_create_normalized(m,NUMERUS_MATRIX_NORMALIZE_ROWS,NUMERUS_MATRIX_NORMALIZE_L2,&n)==NUMERUS_MATRIX_DIVISION_BY_ZERO);
 assert(n==NULL); numerus_matrix_destroy(m);
 assert(numerus_matrix_create_dense(1,2,bad,&m)==NUMERUS_MATRIX_SUCCESS);
 assert(numerus_matrix_create_normalized(m,NUMERUS_MATRIX_NORMALIZE_WHOLE,NUMERUS_MATRIX_NORMALIZE_L2,&n)==NUMERUS_MATRIX_NON_FINITE);
 assert(n==NULL); numerus_matrix_destroy(m);
}
int main(void){test_axes_and_norms();test_zero_and_nonfinite();puts("Matrix normalization tests passed.");return 0;}
