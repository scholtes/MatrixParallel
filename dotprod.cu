#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// P = max power of 2 to test up to
// i.e., test for N = 2^0, 2^1, 2^2... 2^P
#define P 8
#define ThreadsPerBlock (1<<10)
#define MAX_TILE_WIDTH 16
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
    // Notice that we are allocating MORE shared memory than we
    // will actually use.  MAX_TILE_WIDTH^2 (each) is allocated
    // but only tile_width^2 (each) is used for computation.
    __shared__ float Mds[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
    __shared__ float Nds[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
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
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    // Finish adding together the partial sums on the host (linearly).
    // See the kernel dot product function to see the iterative halving
    // (i.e., O(log n)) sum.

    for(int i = 1; i < size_C; i++) {
        h_C[0] += h_C[i];
    }

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

    matrixMultKernel<<< grid, blocks, 2*tile_width*tile_width >>> (
            d_A, d_B, d_C, width, tile_width
    );

    // Copy result from device to host
    cudaMemcpy(C, d_C, mem_size_Matrix, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)(RAND_MAX+1.0);
}

// Copies a random row from M to V
void extractRow(float* V, float* M, int rowlen, int row) {
    for(int i = 0; i < rowlen; i++) {
        V[i] = M[row*rowlen + i];
    }
}

// Copies a random column from M to V
void extractCol(float* V, float* M, int collen, int col) {
    for(int i = 0; i < collen; i++) {
        V[i] = M[i*collen + col];
    }
}

int main(int argc, char** argv) {

    unsigned int size_Vect; // Number of elements in vectors
    unsigned int mem_size_Vect;
    unsigned int size_Matrix; // Number of elements in matricies
    unsigned int mem_size_Matrix;

    float dotprod_expected; // Computed by dotprod -- not by Matrix Mult
    float dotprod_ABij; // Value of AB at row i column j (random sample)
    int random_i; // Random i to choose a row
    int random_j; // Random j to choose a column

    float* h_Row; // Vectors for a Row and Column
    float* h_Col;
    float* h_A; // Matricies A and B
    float* h_B;
    float* h_C; // Matrix multiplication AB result

    // Seed the random number generator
    // Let's use this year for fun
    srand(2015);

    // Test for different powers
    for(int p = 1; p <= P; p++) {
        printf("p=%d (N=2^%d)\n", p, p);

        // Allocate host memory fors vector Row and Col
        size_Vect = 1<<p;
        mem_size_Vect = sizeof(float) * size_Vect;
        h_Row = (float*) malloc(mem_size_Vect);
        h_Col = (float*) malloc(mem_size_Vect);

        // Allocate host memory for matricies A and B
        size_Matrix = size_Vect * size_Vect;
        mem_size_Matrix = sizeof(float) * size_Matrix;
        h_A = (float*) malloc(mem_size_Matrix);
        h_B = (float*) malloc(mem_size_Matrix);
        h_C = (float*) malloc(mem_size_Matrix);
        memset(h_C, 0, mem_size_Matrix);
        
        // Initialize host memory for matricies Row and Col
        randomInit(h_A, size_Matrix);
        randomInit(h_B, size_Matrix);

        // Initialize host memory for vectors Row and Col
        // These are random samples
        random_i = (int)(rand() % size_Vect);
        random_j = (int)(rand() % size_Vect);
        extractRow(h_Row, h_A, size_Vect, random_i);
        extractCol(h_Col, h_B, size_Vect, random_j);

        // Perform the dot product
        dotprod_expected = dotprod(h_Row, h_Col, size_Vect);

        printf("    (row i, col j) = (%d, %d)\n", random_i, random_j);
        printf("    Expected dot product   = %0.5f...\n",
                dotprod_expected);

        #if VERBOSE
            printf("    Row i = <  ");
            for (int i=0; i < size_Vect; i++) {
                        printf("%0.5f  ", h_Row[i]);
            }
            printf(">\n");
            printf("    Col j = <  ");
            for (int i=0; i < size_Vect; i++) {
                printf("%0.5f  ", h_Col[i]);
            }
            printf(">\n");
        #endif
        for(int tile_width = 1; tile_width <= MAX_TILE_WIDTH; tile_width <<= 1) {
            // Don't test tiles that are larger than the respective matricies
            if(size_Vect < tile_width) { break; }

            // Perform the matrix multiplication
            memset(h_C, 0, mem_size_Matrix);
            matrixMult(h_A, h_B, h_C, size_Vect, tile_width);

            // Extract the desired dot product
            dotprod_ABij = h_C[size_Vect*random_i + random_j];

            // Print results
            printf("    tile_width = %d\n", tile_width);
            printf("        MM dot product = %0.5f...\n",
                    dotprod_ABij);
            #if VERBOSE
                printf("\n");
                for (int i=0; i < size_Matrix; i++) {
                    if(i % size_Vect > 0) { printf(" "); }
                    printf("        %0.5f : %0.5f  :  %0.5f %s \n", h_A[i], h_B[i], h_C[i],
                            i == size_Vect*random_i + random_j ? "*" : "");
                }
            #endif
        }

        // Clean up memory
        free(h_Row);
        free(h_Col);
        free(h_A);
        free(h_B);
        free(h_C);
    }
}