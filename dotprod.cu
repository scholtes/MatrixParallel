#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// P = max power of 2 to test up to
// i.e., test for N = 2^0, 2^1, 2^2... 2^P
#define P 2
#define TILE_WIDTH 1
#define ThreadsPerBlock 1<<5
#define RANDRANGE  5
#define VERBOSE 1

__global__ void dot(float* a, float* b, float* c, unsigned int width) {
    __shared__ float temp[ThreadsPerBlock];
    temp[threadIdx.x] = a[threadIdx.x]*b[threadIdx.x];
    int offset = blockDim.x/2;

    __syncthreads();
    while(offset != 0) {
        if(threadIdx.x < offset) {
            temp[threadIdx.x] += temp[threadIdx.x + offset];
        }
        __syncthreads();
        offset >>= 1;
    }
    if(threadIdx.x == 0) {
        c[blockIdx.x] = temp[0];
    }
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % RANDRANGE +1);
}

int main(int argc, char** argv) {

    unsigned int size_Vect;
    unsigned int mem_size_Vect;
    unsigned int mem_size_C;
    float* h_A;
    float* h_B;
    float* h_C;
    float* d_A;
    float* d_B;
    float* d_C;

    // Test for different powers
    for(int p = 1; p <= P; p++) {

        // Allocate host memory fors vector A and B
        size_Vect = 1<<p;
        mem_size_Vect = sizeof(float) * size_Vect;
        h_A = (float*) malloc(mem_size_Vect);
        h_B = (float*) malloc(mem_size_Vect);

        // Initialize host memory for vectors A and B
        // We seed twice so that the beginning sequences in the
        // loop are the same
        srand(0);
        randomInit(h_A, size_Vect);
        srand(1);
        randomInit(h_B, size_Vect);

        // Allocate device memory for vectors A and B
        cudaMalloc((void**) &d_A, mem_size_Vect);
        cudaMalloc((void**) &d_B, mem_size_Vect);

        // Copy host memory to device
        cudaMemcpy(d_A, h_A, mem_size_Vect, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, mem_size_Vect, cudaMemcpyHostToDevice);

        // Allocate host memory for the result C = A dot B
        mem_size_C = sizeof(float);
        h_C = (float*) malloc(mem_size_C);

        // Allocate device memory for the result
        cudaMalloc((void**) &d_C, mem_size_C);

        // Set up and perform the calculation
        dim3 blocks_Vect(ThreadsPerBlock);
        dim3 grid_Vect(size_Vect/ TILE_WIDTH, size_Vect/ TILE_WIDTH);

        dot<<< grid_Vect, blocks_Vect >>>(d_A, d_B, d_C, size_Vect);
        
        // Copy result from device to host
        cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

        // Basic test
        #if VERBOSE
            printf("A = [ ");
            for(int i=0; i < size_Vect; i++) {
                printf("%0.1f ", h_A[i]);
            }
            printf("]\nB = [ ");
            for(int i=0; i < size_Vect; i++) {
                printf("%0.1f ", h_B[i]);
            }
            printf("]\n");
        #endif
        printf("C = %0.2f\n", *h_C);

        // Clean up memory
        free(h_A);
        free(h_B);
        free(h_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}