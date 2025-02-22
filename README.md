# microSYCL

This is a micro-benchmark for testing the overhead of SYCL features, the following features are tested in this benchmark 

* memory allocation
* parallelization 
* atomics 
* barriers
* vectorization

# Building 

Make is used to build this benchmark. 

```
make VENDOR=acpp|intel-llvm|<empty=DPCPP> BACKEND=omp|<empty=ocl> ARCH=x86|a64fx|graviton3
```
Depending on the Implementation choosen, paths to all the the binaries and libraries should be added to the standard environment variables, such as $PATH and $LD_LIBRARY_PATH

For optimal performance set `OMP_PROC_BIND` environment variable is set to true. 

### Example

```
./binary [-s size |-b blocksize <optional> |-I No. iterations | --print-system
			--gemm 	: to run matrix matrix multiplication 
			--gemm-opt : to optimized matrix matrix multiplication 
			--gemv 	: to run matrix vector multiplication 
			--triad	: to run a triad benchmark 
			--outer-product	: to run an outer product benchmark
			--cross-product	: to run an cross product benchmark
			--spmv : execute spmv kernel
 			--map  : test for different memory access patterns 
					--transpose : with transpose 
					--mat-add   : with matrix addition
			-i: for different routines in vectorization benchmark
		    	  1 - range with USM
		    	  2 - range with Buffer and Accessors
		    	  3 - nd_range with USM
		    	  4 - nd_range with Buffer and Accessor
			--mem-alloc	: to alloc memory using SYCL and standard malloc 
			--reduction	: to test reduction using atomics and sycl reduction construct
			--range	: to test sycl range construct
			--ndrange : to test sycl nd_range construct
			--barrier : to test sycl barrier construct
			
   
```

