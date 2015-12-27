#include <cuda_runtime.h>
#include <stdio.h>
#include <cudnn.h>
#include <cublas_v2.h>
using namespace std;
#define real float
#define H 16
#define W 16
#define S 3
#define R 3
#define BX 16
#define BY 16
#define checkCUDNNError(status) \
    do{ \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("CUDA FAILURE: %s\n", cudnnGetErrorString(status)); \
    } \
    }while(0)
#define checkCudaError(status) \
    if (status != cudaSuccess) { \
        printf("CUDA FAILURE: %s\n", cudaGetErrorString(status)); \
    } 

float alpha=1.f, beta=0.f;
cudnnHandle_t cudnnHandle;
cudnnTensorDescriptor_t bottom_desc_, top_desc_; 
cudnnFilterDescriptor_t filter_desc_;
cudnnConvolutionDescriptor_t conv_desc_;
cudnnConvolutionFwdAlgo_t algo_;
size_t workspaceSizeInBytes;
void* workspace;

void setup() {
    checkCUDNNError(cudnnCreateTensorDescriptor(&bottom_desc_));
    checkCUDNNError(cudnnCreateTensorDescriptor(&top_desc_));
    checkCUDNNError(cudnnCreateFilterDescriptor(&filter_desc_));
    checkCUDNNError(cudnnCreateConvolutionDescriptor(&conv_desc_));
    checkCUDNNError(cudnnCreate(&cudnnHandle));
    checkCUDNNError(cudnnSetTensor4dDescriptor(bottom_desc_,
                    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 1, H, W));
    checkCUDNNError(cudnnSetTensor4dDescriptor(top_desc_,
                    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 1, H-R+1, W-S+1));
    checkCUDNNError(cudnnSetFilter4dDescriptor(filter_desc_,
                    CUDNN_DATA_FLOAT, 1, 1, R, S));
    checkCUDNNError(cudnnSetConvolution2dDescriptor(conv_desc_,
                    0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    checkCUDNNError(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                    bottom_desc_, filter_desc_, conv_desc_, top_desc_,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
    checkCUDNNError(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                    bottom_desc_, filter_desc_, conv_desc_, top_desc_,
                    algo_, &workspaceSizeInBytes));
    checkCudaError(cudaMalloc((void**)&workspace, workspaceSizeInBytes));
}

__global__ void baseline(real* input, real* output, real* K, int outh, int outw) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    float tmp = 0;
    for (int a = 0; a < R; ++a) {
        for (int b = 0; b < S; ++b) {
            tmp += input[(j+a)*W + (i+b)] * K[(R-1-a)*S + (S-1-b)];
        }
    }
    output[j*outw+i] = tmp;
}

//fixed on 3*3
__global__ void opt_register(real* input, real* output, real* K, int outh, int outw) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    /*int thi = threadIdx.x;*/
    int thj = threadIdx.y;
    float a0,a1,a2,a3,a4,a5,a6,a7,a8;
    if (thj == 0) {
        a0 = K[0];
        a1 = K[1];
        a2 = K[2];
        a3 = K[3];
        a4 = K[4];
        a5 = K[5];
        a6 = K[6];
        a7 = K[7];
        a8 = K[8];
    }
    __syncthreads();
    output[j*outw+i] = input[j*W+i]*a8 + input[j*W+i+1]*a7 + input[j*W+i+2]*a6 + 
                       input[(j+1)*W+i]*a5 + input[(j+1)*W+i+1]*a4 + input[(j+1)*W+i+2]*a3 +
                       input[(j+2)*W+i]*a2 + input[(j+2)*W+i+1]*a1 + input[(j+2)*W+i+2]*a0;
    /*output[i*outw+j] = input[j*W+i]*K[8] + input[j*W+i+1]*K[7] + input[j*W+i+2]*K[6] + */
                       /*input[(j+1)*W+i]*K[5] + input[(j+1)*W+i+1]*K[4] + input[(j+1)*W+i+2]*K[3] +*/
                       /*input[(j+2)*W+i]*K[2] + input[(j+2)*W+i+1]*K[1] + input[(j+2)*W+i+2]*K[0];*/
}
__global__ void opt_shm(real* input, real* output, real* K, int outh, int outw) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    output[i*outw+j] = 0;
    __shared__ real shm_b[BY][BX];
    int thi = threadIdx.x;
    int thj = threadIdx.y;
    if (thi == 0 && thj == 0) {
        for (int a = 0; a < R; ++a) {
            for (int b = 0; b < S; ++b) {
                shm_b[a][b] = input[i*S+j];
            }
        }
    }
    for (int a = i; a < i+S; ++a) {
        for (int b = j; b < j+R; ++b) {
            output[i*outw+j] += shm_b[a][b] * K[(R-1-(b-j))*S+(S-1-(a-i))];
        }
    }
}
void cpu_comp(real* input, real* output, real* K, int h, int w, int kr, int ks, int outh, int outw) {
    for (int i = 0; i < outh; ++i) {
        for (int j = 0; j < outw; ++j) {
            real a = 0;
            for (int r = 0; r < kr; ++r) {
                for (int s = 0; s < ks; ++s) {
                    a += input[(i+r)*w + (j+s)] * K[(kr-1-r)*ks+(ks-1-s)];
                }
            }
            output[i*outw + j] = a;
        }
    }
}
bool check(real* A, real* B, int size) {
    for (int i = 0; i < size; ++i)
        if (int(A[i]) != int(B[i])) {
            printf("ERROR at %d: %d %d\n", i, int(A[i]), int(B[i]));
            return false;
        }
    return true;
}
int main() {
    setup();
    int insize = H*W*sizeof(float);
    int outsize = (H-R+1)*(W-S+1)*sizeof(float);
    int ksize = S*R*sizeof(float);
    int outh = H - R + 1;
    int outw = W - S + 1;
    real* host_input = (real*)malloc(insize);
    real* host_baseline_output = (real*)malloc(outsize);
    real* host_output = (real*)malloc(outsize);
    real* cpu_output = (real*)malloc(outsize);
    real* host_k = (real*)malloc(ksize);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            host_input[i*W+j] = j;
        }
    }
    for (int i = 0; i < H-R+1; ++i) {
        for (int j = 0; j < W-S+1; ++j) {
            host_output[i*(W-S+1)+j] = 0;
            host_baseline_output[i*(W-S+1)+j] = 0;
        }
    }
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < S; ++j) {
            host_k[i*S+j] = i*S+j+1;
        }
    }
    cpu_comp(host_input, cpu_output, host_k, H, W, R, S, outh, outw);
    printf("cpu result:\n");
    for (int i = 0; i < outh; ++i) {
        for (int j = 0; j < outw; ++j) {
            printf("%f ", cpu_output[i*outw+j]);
        }
        printf("\n");
    }
    printf("start...\n");
    real *dev_input, *dev_output, *dev_k;
    cudaMalloc(&dev_input, insize);
    cudaMalloc(&dev_output, outsize);
    cudaMalloc(&dev_k, ksize);
    cudaMemcpy(dev_input, host_input, insize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k, host_k, ksize, cudaMemcpyHostToDevice);
    dim3 threadPerBlock(BX, BY);
    dim3 blockPerGrid((outw+BX-1)/BX, (outh+BY-1)/BY);

    //init
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    //baseline
    cudaEventRecord(start, 0);
    //for (int i = 0; i < 1000; ++i)
    baseline<<<blockPerGrid, threadPerBlock>>>(dev_input, dev_output, dev_k, outh, outw);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("baseline: time = %fms \n", time);
    cudaMemcpy(host_baseline_output, dev_output, outsize, cudaMemcpyDeviceToHost);
    printf("baseline result:\n");
    for (int i = 0; i < outh; ++i) {
        for (int j = 0; j < outw; ++j) {
            printf("%f ", host_baseline_output[i*outw+j]);
        }
        printf("\n");
    }
    //opt_register
    cudaEventRecord(start, 0);
    //for (int i = 0; i < 1000; ++i)
    opt_register<<<blockPerGrid, threadPerBlock>>>(dev_input, dev_output, dev_k, outh, outw);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("opt_register: time = %fms \n", time);
    cudaMemcpy(host_output, dev_output, outsize, cudaMemcpyDeviceToHost);
    check(host_baseline_output, host_output, outsize);
    /*for (int i = 0 ; i < 10 ; ++i) */
        /*printf("%f ", host_baseline_output[i]);*/
    /*printf("\n");*/
    /*for (int i = 0 ; i < 10 ; ++i) */
        /*printf("%f ", host_output[i]);*/
    /*printf("\n");*/
    //recover dev_output[0] to 0
    host_output[0] = 0;
    cudaMemcpy(dev_output, host_output, outsize, cudaMemcpyHostToDevice);
    //cudnn
    cudaEventRecord(start, 0);
    //for (int i = 0; i < 1000; ++i)
    checkCUDNNError(cudnnConvolutionForward(cudnnHandle, &alpha, bottom_desc_,
                (void*)dev_input, filter_desc_, (void*)dev_k,
                conv_desc_, algo_, workspace, workspaceSizeInBytes, &beta,
                top_desc_, (void*)dev_output));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("cudnn: time = %fms \n", time);
    cudaMemcpy(host_output, dev_output, outsize, cudaMemcpyDeviceToHost);
    check(host_baseline_output, host_output, outsize);

    return 0;
}
