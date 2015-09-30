#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// P = max power of 2 to test up to
// i.e., test for N = 2^0, 2^1, 2^2... 2^P
#define P 3
#define ThreadsPerBlock (1<<10)
#define TILE_WIDTH 1
#define RANDRANGE  5
#define VERBOSE 0

__global__ void dot(float* a, float* b, float* c, unsigned int width) {
    __shared__ float temp[ThreadsPerBlock];
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int index = ThreadsPerBlock*bx + tx;
    int sumrange = width < ThreadsPerBlock ? width : ThreadsPerBlock;

    if(index < width) {
        temp[tx] = a[index]*b[index];
    }

    __syncthreads();
    // Iterative halving sum
    for(int offset = sumrange >> 1; offset > 0; offset >>= 1) {
        if(tx < offset) {
            temp[tx] += temp[tx+offset];
        }
        __syncthreads();
    }

    if(tx == 0) {
        c[bx] = temp[0];
    }

}

__global__ void matrixMultKernel(float* Md, float* Nd, float* Pd, int Width, int tile_width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * tile_width + ty;
    int Col = bx * tile_width + tx;

    float Pvalue = 0;
    // Loop over the Md and Nd tiles required to compute the Pd element

    for (int m = 0; m < Width/tile_width; ++m) {
        // Collaborative loading of Md and Nd tiles into shared memory
        Mds[ty][tx] = Md[Row*Width + (m*tile_width + tx)];
        Nds[ty][tx] = Nd[Col + (m*tile_width + ty)*Width];
        __syncthreads();

        for (int k = 0; k < tile_width; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    Pd[Row*Width+Col] = Pvalue;
}

// Num subresults is the number of sub- dot products computed in the
// GPU.  The host will add them all up.
float dotprod(float* a, float* b, unsigned int width) {
    unsigned int size_C; // Number of elements in result vector
    unsigned int mem_size_C;
    float ret;
    float* h_C;
    float* d_A;
    float* d_B;
    float* d_C;

    // Allocate device memory for vectors A and B
    unsigned int mem_size_Vect = sizeof(float) * width;
    cudaMalloc((void**) &d_A, mem_size_Vect);
    cudaMalloc((void**) &d_B, mem_size_Vect);

    // Copy host memory to device
    cudaMemcpy(d_A, a, mem_size_Vect, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, mem_size_Vect, cudaMemcpyHostToDevice);

    // Allocate host memory for the result C = A dot B
    size_C = 1 + ((width - 1) / ThreadsPerBlock);
    mem_size_C = sizeof(float) * size_C;
    h_C = (float*) malloc(mem_size_C);
    *h_C = 0;

    // Allocate device memory for the result
    cudaMalloc((void**) &d_C, mem_size_C);
    
    // Set up the calculation
    dim3 blocks_Vect(ThreadsPerBlock);
    dim3 grid_Vect(size_C);

    dot<<< grid_Vect, blocks_Vect >>>(d_A, d_B, d_C, width);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);/*

    // Finish adding together the partial sums on the host (linearly).
    // See the kernel dot product function to see the iterative halving
    // (i.e., O(log n)) sum.

    for(int i = 1; i < size_C; i++) {
        h_C[0] += h_C[i];
    }
    */
    ret = h_C[0];

    // Clean up memory
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ret;
}

// Multiplies A with B and puts the result in C
void matrixMult(float* A, float* B, float* C, int width, int tile_width) {
    // Memory allocation grunt work
    unsigned int mem_size_Matrix = sizeof(float) * width * width;
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**) &d_A, mem_size_Matrix);
    cudaMalloc((void**) &d_B, mem_size_Matrix);
    cudaMalloc((void**) &d_C, mem_size_Matrix);
    cudaMemcpy(d_A, A, mem_size_Matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, mem_size_Matrix, cudaMemcpyHostToDevice);

    // Set up and perform the actual computation
    dim3 blocks(tile_width, tile_width);
    dim3 grid(width / tile_width, width / tile_width);

    matrixMultKernel<<< grid, blocks >>> (d_A, d_B, d_C, width, tile_width);

    // Copy result from device to host
    cudaMemcpy(C, d_C, mem_size_Matrix, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main(int argc, char** argv) {

    float h_Row[4] = {
            1, 3, 7, 2
    };
    float h_Col[4] = {
            4, 1, 2, 5
    };

    float h_A[16] = {
            1, 3, 7, 2,
            1, 3, 7, 2,
            1, 3, 7, 2,
            1, 3, 7, 2
    }; // Matricies A and B
    float h_B[16] = {
            4, 4, 4, 4,
            1, 1, 1, 1,
            2, 2, 2, 2,
            5, 5, 5, 5
    };
    float h_C[16] = { 0 }; // Matrix multiplication AB result

    // Seed the random number generator
    srand(0);

    printf("Dot product: %0.1f\n", dotprod(h_Row, h_Col, 4));
    matrixMult(h_A, h_B, h_C, 4, 1);
    printf("Dot product: %0.1f\n", h_C[0]);
    printf("Dot product: %0.1f\n", dotprod(h_Row, h_Col, 4));
    printf("Dot product: %0.1f\n", dotprod(h_Row, h_Col, 4));

    for(int i = 0; i < 4; i++) {
        printf("%0.1f ", h_Row[i]);
    }

}