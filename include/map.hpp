#ifndef MAP_HPP
#define MAP_HPP

#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

/*matrix addition*/
void range_usm_matrix_addition(sycl::queue &Q, int size, int dim, int iter,bool print);

void ndrange_usm_matrix_addition(sycl::queue &Q, int size, int dim, int block_size, int iter, bool print);

/*transpose*/
void range_usm_matrix_transpose(sycl::queue &Q, int size, int dim, int iter, bool print);

void ndrange_usm_matrix_transpose(sycl::queue &Q, int size, int dim, int block_size, int iter, bool print);


#endif