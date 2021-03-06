#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define N  1<<1
#define TILE_WIDTH 1
#define RANDRANGE  5


__global__ void matrixMultKernel(float* Md, float* Nd, float* Pd, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < Width/TILE_WIDTH; ++m) {
        // Collaborative loading of Md and Nd tiles into shared memory
        Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*Width];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    Pd[Row*Width+Col] = Pvalue;
}



// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % RANDRANGE +1);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////


int main(int argc, char** argv) {

    // set seed for rand()
    srand(2015);

    // allocate host memory for matrices A and B
    unsigned int size_A = N * N;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = N * N;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    //  allocate device memory
    float* d_A;   float* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_B);

    //  copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, 
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, 
        cudaMemcpyHostToDevice);

    // allocate host memory for the result C
    unsigned int size_C = N*N ;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);
    memset(h_C,0,mem_size_C);


    //  allocate device memory for the result
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    //  perform the calculation
    // setup execution parameters
    dim3 blocks(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(N/ TILE_WIDTH, N/ TILE_WIDTH);

    // execute the kernel
    matrixMultKernel<<< grid, blocks >>>(d_A, d_B, d_C, N);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, 
        cudaMemcpyDeviceToHost);

    //ToDo: Your Test code here......
    printf("N= %d and TILE_WIDTH =%d\n", N,TILE_WIDTH);
    for (int i=0; i < N*N; i++) {
        printf("%20.15f : %20.15f  : %20.15f \n", h_A[i], h_B[i], h_C[i] );
    }

    //  clean up memory
    free(h_A);   free(h_B); free(h_C);
    cudaFree(d_A);    cudaFree(d_B);    cudaFree(d_C);

}
