#include <cuda_runtime.h>
#include <stdio.h>
using namespace std;
#define real float
#define H 1024
#define W 1024
#define S 3
#define R 3
#define BX 32
#define BY 32

__global__ void baseline(real* input, real* output, real* K, int outh, int outw) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    output[i*outw+j] = 0;
    for (int ii = i; ii < i+S; ++ii) {
        for (int jj = j; jj < j+R; ++jj) {
            output[i*outw+j] += input[ii*W+jj] * K[(R-1-(jj-j))*S+(S-1-(ii-i))];
        }
    }
}

int main() {
    int insize = H*W*sizeof(float);
    int outsize = (H-R+1)*(W-S+1)*sizeof(float);
    int ksize = S*R*sizeof(float);
    real* host_input = (real*)malloc(insize);
    real* host_output = (real*)malloc(outsize);
    real* host_k = (real*)malloc(ksize);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            host_input[i*W+j] = 1;
        }
    }
    for (int i = 0; i < H-R+1; ++i) {
        for (int j = 0; j < W-S+1; ++j) {
            host_output[i*(W-S+1)+j] = 0;
        }
    }
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < S; ++j) {
            host_k[i*S+j] = i*S+j+1;
        }
    }
    real *dev_input, *dev_output, *dev_k;
    cudaMalloc(&dev_input, insize);
    cudaMalloc(&dev_output, outsize);
    cudaMalloc(&dev_k, ksize);
    cudaMemcpy(dev_input, host_input, insize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, host_k, ksize, cudaMemcpyHostToDevice);
    int outh = H - R + 1;
    int outw = W - S + 1;
    dim3 threadPerBlock(BX, BY);
    dim3 blockPerGrid((outw+BX-1)/BX, (outh+BY-1)/BY);
    baseline<<<blockPerGrid, threadPerBlock>>>(dev_input, dev_output, dev_k, outh, outw);
    cudaMemcpy(host_output, dev_output, outsize, cudaMemcpyDeviceToHost);
    printf("%f\n", host_output[0]); //45


    return 0;
}
