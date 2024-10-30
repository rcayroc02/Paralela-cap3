#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int N = 1024; 

// B
__global__ void matrixAddKernelElement(float *C, const float *A, const float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

// C
__global__ void matrixAddKernelRow(float *C, const float *A, const float *B, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// D
__global__ void matrixAddKernelColumn(float *C, const float *A, const float *B, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int row = 0; row < N; ++row) {
            int idx = row * N + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// A
void matrixAdditionHost(float *h_C, const float *h_A, const float *h_B, int N, int option) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (option == 1) {
        //1thread per element
    
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrixAddKernelElement<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);
    } else if (option == 2) {
        //1thread per row
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
        matrixAddKernelRow<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);
    } else if (option == 3) {
        //one thread per collum
        dim3 threadsPerBlock(256);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
        matrixAddKernelColumn<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tiempo de ejecucion (Opcion " << option << "): " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int option = 1; option <= 3; ++option) {
        matrixAdditionHost(h_C.data(), h_A.data(), h_B.data(), N, option);
        
    }

    return 0;
}
