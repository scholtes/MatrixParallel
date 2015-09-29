#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// P = max power of 2 to test up to
// i.e., test for N = 2^0, 2^1, 2^2... 2^P
#define P 6
#define TILE_WIDTH 1
#define ThreadsPerBlock (1<<3)
#define BlocksPerGrid ((1<<16)-1)
#define RANDRANGE  5
#define VERBOSE 0

__global__ void dot(float* a, float* b, float* c, unsigned int width) {
    __shared__ float temp[ThreadsPerBlock];
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int index = ThreadsPerBlock*bx + tx;

    if(index < width) {
        temp[tx] = a[index]*b[index];
    }
    //__syncthreads();
    if(tx == 0) {
        float sum = 0;
        for(int i = 0; i < width; i++) {
            sum += temp[i];
        }
        c[bx] = sum;
    }
}

// Num subresults is the number of sub- dot products computed in the
// GPU.  The host will add them all up.
void dotprod(float* a, float* b, float* c, unsigned int width,
        unsigned int num_subresults) {
    // Set up the calculation
    dim3 blocks_Vect(ThreadsPerBlock);
    dim3 grid_Vect(num_subresults);

    dot<<< grid_Vect, blocks_Vect >>>(a, b, c, width);

    // Perform additional sums on the host
    //for(int i = 1; i < num_subresults; i++) {
    //    c[0] += c[i];
    //}
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % RANDRANGE +1);
}

int main(int argc, char** argv) {

    unsigned int size_Vect; // Number of elements in vectors
    unsigned int size_C; // Number of elements in result vector
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
        size_C = 1 + size_Vect/ThreadsPerBlock;
        mem_size_C = sizeof(float) * size_C;
        h_C = (float*) malloc(mem_size_C);
        *h_C = 0;

        // Allocate device memory for the result
        cudaMalloc((void**) &d_C, mem_size_C);

        // Perform the calculation
        dotprod(d_A, d_B, d_C, size_Vect, size_C);
        
        
        // Copy result from device to host
        // We only care about the first element, so we only copy that
        cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

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