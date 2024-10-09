# sycl-microbench

This is a micro-benchmark for testing the overhead of SYCL features, the following features are tested in this benchmark 

* memory allocation
* parallelization 
* atomics 
* barriers
* vectorization

# Building 

Make is used to build this benchmark. 

```
make VENDOR=acpp|intel-llvm|<empty=DPCPP> BACKEND=omp|<empty> ARCH=x86|a64fx
```
Depending on the Implementation choosen, paths to all the the binaries and libraries should be added to the standard environment variables, such as $PATH and $LD_LIBRARY_PATH

For optimal performance set `OMP_PROC_BIND` environment variable is set to true. 

### Example

```
./binary [-s size |-b blocksize <optional> |-I No. iterations | --print-system\n"
			--mat-mul : to run matrix multiplication 
			--mat-vec : to run matrix vector multiplication 
			--triad   : to run a triad operation 
			--outer-product   : to run a outer product operation 
			-i : for different routines in vectorization benchmark
			      1 - range with USM
			      2 - range with Buffer and Accessors
			      3 - nd_range with USM
			      4 - nd_range with Buffer and Accessor
			--mem-alloc : to alloc memory using SYCL and standard malloc 
			--reduction : to test reduction using atomics and sycl reduction construct
			--range : to test sycl range construct
			--ndrange : to test sycl nd_range construct
			--barrier : to test sycl barrier construct
			
   
```

