nvcc -O3 --fmad false -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=compute_52 $1 && ./a.out
