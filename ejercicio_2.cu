#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

const int N = 1024;


__global__ void matrixVectorMultiplyKernel(float *A, const float *B, const float *C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += B[row * N + j] * C[j];
        }
        A[row] = sum;
    }
}


void matrixVectorMultiplyHost(float *h_A, const float *h_B, const float *h_C, int N) {

    // asignar memoria para los devices
    float *d_B, *d_C, *d_A;
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMalloc(&d_A, N * sizeof(float));

    // copiar la entrada a los devices
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configuración de los parámetros de ejecución
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;


    matrixVectorMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // copiar salida al device
    cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A);
}


void initializeData(float *B, float *C, int N) {
    for (int i = 0; i < N * N; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
    for (int i = 0; i < N; ++i) {
        C[i] = static_cast<float>(rand()) / RAND_MAX; 
    }
}

int main() {
    srand(static_cast<unsigned>(time(0))); 

    std::vector<float> h_B(N * N); 
    std::vector<float> h_C(N);      
    std::vector<float> h_A(N);      

    
    initializeData(h_B.data(), h_C.data(), N);

    
    matrixVectorMultiplyHost(h_A.data(), h_B.data(), h_C.data(), N);


    for(int i=0; i < N; i++){
        std::cout << h_A[i] << " ";
    }

    return 0;
}
