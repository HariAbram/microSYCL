#include <iostream>
#include <math.h>
#include <vector>
#include <sycl/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>



//using namespace cl;

/*gemv*/

void range_usm_map(sycl::queue &Q, int size, int dim);



