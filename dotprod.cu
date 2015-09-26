#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define N  1<<1
#define TILE_WIDTH 1
#define ThreadsPerBlock 1024
#define RANDRANGE  5

__global__ void dot(float* a, float* b, float* c, unsigned int width) {
    __shared__ int temp[ThreadsPerBlock];
    temp[threadIdx.x] = a[threadIdx.x]*b[threadIdx.x];

    if(0 == threadIdx.x) {
        float sum = 0;
        for(int i = 0; i < width; i++) {
            sum += temp[i];
        }
        *c = sum;
    }
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % RANDRANGE +1);
}

int main(int argc, char** argv) {

    // Seed rand
    srand(0);

    // Allocate host memory fors vector A and B
    unsigned int size_Vect = N;
    unsigned int mem_size_Vect = sizeof(float) * size_Vect;
    float* h_A = (float*) malloc(mem_size_Vect);
    float* h_B = (float*) malloc(mem_size_Vect);

    // Initialize host memory for vectors A and B
    randomInit(h_A, size_Vect);
    randomInit(h_B, size_Vect);

    // Allocate device memory for vectors A and B
    float* d_A; float* d_B;
    cudaMalloc((void**) &d_A, mem_size_Vect);
    cudaMalloc((void**) &d_B, mem_size_Vect);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_Vect, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_Vect, cudaMemcpyHostToDevice);

    // Allocate host memory for the result C = A dot B
    unsigned int mem_size_C = sizeof(float);
    float* h_C = (float*) malloc(mem_size_C);

    // Allocate device memory for the result
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // Set up and perform the calculation
    dim3 blocks(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(N/ TILE_WIDTH, N/ TILE_WIDTH);

    dot<<< grid, blocks >>>(d_A, d_B, d_C, size_Vect);
    
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    // Basic test
    printf("A = [ ");
    for(int i=0; i < size_Vect; i++) {
        printf("%0.1f ", h_A[i]);
    }
    printf("]\nB = [ ");
    for(int i=0; i < size_Vect; i++) {
        printf("%0.1f ", h_B[i]);
    }
    printf("]\nC = %0.2f\n", *h_C);

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}