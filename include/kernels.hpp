#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>
#include <string>


#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"
#include "../include/parallel-bench.hpp"

//using namespace cl;

/// init arrays

void init_arrays(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global);

void init_arrays(sycl::queue &Q, sycl::buffer<TYPE, 1>  m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global);

/// copy

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global);

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global, sycl::range<1> local);

void kernel_copy(sycl::queue &Q, sycl::buffer<TYPE, 1> m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global);

void kernel_copy(TYPE* m, TYPE* a, int size);

/// parallel

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global);

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local);

void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global);

void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global, sycl::range<1> local);


void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global);

void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local);

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global);

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global, sycl::range<2> local);


void kernel_parallel_omp(int size, TYPE* sum);

void kernel_parallel_omp_nested(int size, TYPE* sum);

/// atomics

void kernel_atomics(sycl::queue &Q, sycl::range<1> global, TYPE* m_shared, TYPE* sum);

void kernel_atomics(sycl::queue &Q, sycl::range<1> global, sycl::buffer<TYPE, 1> m_buff, sycl::buffer<TYPE, 1> sum_buff);

void kernel_atomics(int size, TYPE &sum, TYPE* m);

/// reduction


void kernel_reduction(sycl::queue &Q, TYPE* sum, TYPE* m_shared, sycl::range<1> global);

void kernel_reduction(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::buffer<TYPE, 1> m_buff, sycl::range<1> global);

void kernel_reduction(int size, TYPE &sum, TYPE* m);

/// barriers

void kernel_group_barrier(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local);

void kernel_group_barrier(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local);

void kernel_group_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<1> global, sycl::range<1> local);

void kernel_group_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<2> global, sycl::range<2> local);

void kernel_subgroup_barrier(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local);

void kernel_subgroup_barrier(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local);

void kernel_subgroup_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<1> global, sycl::range<1> local);

void kernel_subgroup_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<2> global, sycl::range<2> local);

void kernel_barrier_omp(int size, TYPE* sum);


#endif