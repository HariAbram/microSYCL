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

#include "../include/utils.hpp"

#ifndef TYPE
#define TYPE double
#endif

/////////////////////////////////////////////// init arrays

void init_arrays(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1> it){

            const int k = it.get_id(0);
            m[k] = 0.0;
            a[k] = 1.0;
            
        });
    });
    Q.wait();

}

void init_arrays(sycl::queue &Q, sycl::buffer<TYPE, 1>  m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        sycl::accessor m_acc(m_buff, cgh, sycl::write_only, sycl::no_init);
        sycl::accessor a_acc(a_buff, cgh, sycl::write_only, sycl::no_init);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_acc[k] = 0.0 ;
            a_acc[k] = 1.0;
        
        });

    });
    Q.wait();

}

/////////////////////////////////////////////// copy

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1> it){
            
            const int k = it.get_id(0);
            m[k] = a[k];

        });
    });
    Q.wait();
    

}

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1> it){

            const int k = it.get_global_id(0);

            m[k] = a[k];

        });
    });
    Q.wait();

}

void kernel_copy(sycl::queue &Q, sycl::buffer<TYPE, 1> m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        auto m_acc = m_buff.get_access<sycl::access::mode::write>(cgh);
        auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_acc[k] = a_acc[k];
        
        });

    });
    Q.wait();

}

void kernel_copy(TYPE* m, TYPE* a, int size)
{
    #pragma omp parallel for 
    for (size_t j = 0; j < size*size; j++)
    {
    m[j] = a[j];
    }

}

/////////////////////////////////////////////// parallel 
/// 1 dim

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[k] < 0)
                {
                    break;
                }
                sum[k] += 1;
            }
        });
    });

    Q.wait();

}

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            const int k = it.get_global_id(0);
            for (size_t l = 0; l < 1024; l++)
            {  
                if (sum[k] < 0)
                {
                    break;
                }  
                sum[k] += 1;
            }
        });
    });
    Q.wait();
}


void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){
            
            const int k = it.get_id(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[k] < 0)
                {
                    break;
                } 
                sum_acc[k] += 1;
            }
        });
    });
    Q.wait();
}

void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){
            
            const int k = it.get_global_id(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[k] < 0)
                {
                    break;
                } 
                sum_acc[k] += 1;    
            }
        });
    });
    Q.wait();

}

/// 2 dim

void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::range<2>(global), [=](sycl::item<2>it){

            const int k = it.get_id(0);
            const int k1 = it.get_id(1);

            const int N = it.get_range(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[k*N+k1] < 0)
                {
                    break;
                } 
                sum[k*N+k1] += 1;  
            }
        });
    });

    Q.wait();

}

void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2>it){

            const int k = it.get_global_id(0);
            const int k1 = it.get_global_id(1);

            const int N = it.get_global_range(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[k*N+k1] < 0)
                {
                    break;
                }  
                sum[k*N+k1] += 1;   
            }
        });
    });
    Q.wait();
}

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::range<2>(global), [=](sycl::item<2>it){

            const int k = it.get_id(0);
            const int k1 = it.get_id(1);

            const int N = it.get_range(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[k*N+k1] < 0)
                {
                    break;
                }  
                sum_acc[k*N+k1] += 1; 
            }
        });
    });
    Q.wait();
}

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2>it){

            const int k = it.get_global_id(0);
            const int k1 = it.get_global_id(1);

            const int N = it.get_global_range(0);
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[k*N+k1] < 0)
                {
                    break;
                } 
                sum_acc[k*N+k1] += 1;
            }        
        });
    });
    Q.wait();
}

/// OMP

void kernel_parallel_omp(int size, TYPE* sum)
{
    #pragma omp parallel 
    {
    #pragma omp for 
    for (size_t j = 0; j < size*size; j++)        
    {
        for (size_t l = 0; l < 1024; l++)
        {
            if (sum[j] < 0)
            {
                break;
            } 
            sum[j] += 1;
        }
    };
    }
}

void kernel_parallel_omp_nested(int size, TYPE* sum)
{
    #pragma omp parallel  
    {
    #pragma omp for collapse(2)
    for (size_t j = 0; j < size; j++)        
    {
        for (size_t k = 0; k < size; k++)
        {
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[j*size+k] < 0)
                {
                    break;
                } 
                sum[j*size+k] += 1;
            }
        }
    }
    
    }
}

/////////////////////////////////////////////// atomics 

void kernel_atomics(sycl::queue &Q, sycl::range<1> global, TYPE* m_shared, TYPE* sum)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int j = it.get_id(0);

            auto v = sycl::atomic_ref<TYPE, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
            sum[0]);

            
            v.fetch_add(m_shared[j]);
            
        });
    });

    Q.wait();
}

void kernel_atomics(sycl::queue &Q, sycl::range<1> global, sycl::buffer<TYPE, 1> m_buff, sycl::buffer<TYPE, 1> sum_buff)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto m_acc = m_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int j = it.get_id(0);

            auto v = sycl::atomic_ref<TYPE, sycl::memory_order::seq_cst,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
            sum_acc[0]);

            
            v.fetch_add(m_acc[j]);
            
        });
    });

    Q.wait();
}

void kernel_atomics(int size, TYPE &sum, TYPE* m)
{
    #pragma omp parallel for 
    for (size_t j = 0; j < size*size; j++)        
    {
      #pragma omp atomic
      sum+= m[j];

    };
}

/////////////////////////////////////////////// Reduction 

void kernel_reduction(sycl::queue &Q, TYPE* sum, TYPE* m_shared, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

#if defined(ACPP) 
        auto sum_red = sycl::reduction(sum, sycl::plus<TYPE>());
#else
        auto sum_red = sycl::reduction(sum, sycl::plus<TYPE>());    
#endif

        cgh.parallel_for<>(sycl::range<1>(global), sum_red ,[=](sycl::item<1>it, auto &sum){

           const int j = it.get_id(0);

            sum += m_shared[j];
            
        });
    });

    Q.wait();

}

void kernel_reduction(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::buffer<TYPE, 1> m_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){
            
        auto m_acc = m_buff.get_access<sycl::access::mode::read>(cgh);    

#if defined(ACPP) 
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);
        auto sum_red = sycl::reduction(sum_acc, sycl::plus<TYPE>());
#else
        auto sum_red = sycl::reduction(sum_buff, cgh,sycl::plus<TYPE>());
        
#endif

        cgh.parallel_for<>(sycl::range<1>(global), sum_red ,[=](sycl::item<1>it, auto &sum){

            const int j = it.get_id(0);

            sum += m_acc[j];
            
        });
    });

    Q.wait();
}

void kernel_reduction(int size, TYPE &sum, TYPE* m)
{
    #pragma omp parallel for reduction(+:sum) 
    for (size_t j = 0; j < size*size; j++)        
    {
      sum+= m[j];

    };
}

/////////////////////////////////////////////// Barriers 


void kernel_group_barrier(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){
            sycl::group_barrier(it.get_group());

            const int k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[k] < 0)
                {
                    break;
                } 
                sum[k]+= 1;
            }
        });
    });
    Q.wait();   
}

void kernel_group_barrier(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2>it){
            sycl::group_barrier(it.get_group());

            const int i = it.get_global_id(0);
            const int j = it.get_global_id(1);
            const int N = it.get_global_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[i*N+j] < 0)
                {
                    break;
                } 
                sum[i*N+j]+= 1;
            }
        });
    });
    Q.wait();   
}

void kernel_group_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){
            sycl::group_barrier(it.get_group());

            const int k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[k] < 0)
                {
                    break;
                } 
                sum_acc[k]+= 1;
            }
        });
    });
    Q.wait();
}

void kernel_group_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2>it){
            sycl::group_barrier(it.get_group());

            const int i = it.get_global_id(0);
            const int j = it.get_global_id(1);
            const int N = it.get_global_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[i*N+j] < 0)
                {
                    break;
                } 
                sum_acc[i*N+j]+= 1;
            }
        });
    });
    Q.wait();
}

void kernel_subgroup_barrier(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1> it){
            sycl::group_barrier(it.get_sub_group());

            const int k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[k] < 0)
                {
                    break;
                } 
                sum[k]+= 1;
            }
        });
    });
    Q.wait();   
}

void kernel_subgroup_barrier(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2> it){
            sycl::group_barrier(it.get_sub_group());

            const int i = it.get_global_id(0);
            const int j = it.get_global_id(1);
            const int N = it.get_global_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[i*N+j] < 0)
                {
                    break;
                } 
                sum[i*N+j]+= 1;
            }
        });
    });
    Q.wait();   
}

void kernel_subgroup_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1> it){
            sycl::group_barrier(it.get_sub_group());

            const int k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[k] < 0)
                {
                    break;
                } 
                sum_acc[k]+= 1;
            }  
        });
    });
    Q.wait();
}

void kernel_subgroup_barrier(sycl::queue &Q, sycl::buffer<TYPE,1> sum_buff, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2> it){
            sycl::group_barrier(it.get_sub_group());

            const int i = it.get_global_id(0);
            const int j = it.get_global_id(1);
            const int N = it.get_global_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                if (sum_acc[i*N+j] < 0)
                {
                    break;
                } 
                sum_acc[i*N+j]+= 1;
            }  
        });
    });
    Q.wait();
}

void kernel_barrier_omp(int size, TYPE* sum)
{
    #pragma omp parallel 
    {
        #pragma omp for
        for (size_t j = 0; j < size*size; j++)        
        {   
            for (size_t l = 0; l < 1024; l++)
            {
                if (sum[j] < 0)
                {
                    break;
                } 
                sum[j] += 1;
            }       
        };
        #pragma omp barrier
    }
}