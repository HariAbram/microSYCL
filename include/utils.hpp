#ifndef UTILS_H
#define UTILS_H

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
#include <iomanip>

//using namespace cl;

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#ifndef TYPE
#define TYPE double
#endif

void print_results(double *timings, int iter, int size, std::string benchmark, int dim, int bench);

void delay_time(int size);

/*sparse matrix utilites*/
void init_sparse_arrays(TYPE *m, int size, int sparsity);
unsigned long xorshf96(void);

#endif