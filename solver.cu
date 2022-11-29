
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

#define ERR(source) fprintf(stderr, source); goto Error

#define BOARDSIZE 9
#define BLANK 0

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void reduce0(int* g_idata, int* g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void readSudokuFromFile(std::string filepath, int* board)
{
    std::ifstream fileStream(filepath);
    std::string input;
    int index = 0;
    while(getline(fileStream, input))
    {
        for(auto c : input)
        {
            board[index++] = c - '0';
        }
    }
}

void printSudoku(int* board)
{
    const std::string lineBreak = "+-------+-------+-------+\n";
    const std::string columnBreak = "| ";

    for(auto i = 0; i < BOARDSIZE; ++i)
    {
        if(i % 3 == 0)
        {
            std::cout << lineBreak;
        }
        for(auto j = 0; j < BOARDSIZE; ++j)
        {
            if(j % 3 == 0)
            {
                std::cout << columnBreak;
            }

            auto value = board[i * BOARDSIZE + j];
            if(value == BLANK)
            {
                std::cout << ". ";
            }
            else
            {
                std::cout << value << ' ';
            }
        }
        std::cout << columnBreak << std::endl;
    }
    std::cout << lineBreak;
}

bool findEmpty(int* board, int& i, int& j)
{
    for(int k = 0; k < BOARDSIZE; ++k)
    {
        for(int l = 0; l < BOARDSIZE; ++l)
        {
            if(board[k * BOARDSIZE + l] == 0)
            {
                i = k;
                j = l;
                return true;
            }
        }
    }
    return false;
}

bool checkIfCorrectRow(int* board, const int& i, const int& value)
{
    for(int j = 0; j < BOARDSIZE; ++j)
    {
        if(board[i * BOARDSIZE + j] == value)
        {
            return false;
        }
    }
    return true;
}

bool checkIfCorrectColumn(int* board, const int& j, const int& value)
{
    for(int i = 0; i < BOARDSIZE; ++i)
    {
        if(board[i * BOARDSIZE + j] == value)
        {
            return false;
        }
    }
    return true;
}

bool checkIfCorrectBox(int* board, const int& i, const int& j, const int& value)
{
    int rowCenter = (i / 3) * 3 + 1;
    int columnCenter = (j / 3) * 3 + 1;

    for(int k = -1; k < 2; ++k)
    {
        for(int l = -1; l < 2; ++l)
        {
            if(board[(rowCenter + k) * BOARDSIZE + (columnCenter + l)] == value)
            {
                return false;
            }
        }
    }
    return true;
}

bool checkIfCorrect(int* board, int i, int j, int value)
{
    return checkIfCorrectRow(board, i, value) && checkIfCorrectColumn(board, j, value) && checkIfCorrectBox(board, i, j, value);
}

bool solveBacktracking(int* board)
{
    int i = 0;
    int j = 0;

    if(!findEmpty(board, i, j))
    {
        return true;
    }

    for(int x = 1; x < 10; ++x)
    {
        if(checkIfCorrect(board, i, j, x))
        {
            board[i * BOARDSIZE + j] = x;
            if(solveBacktracking(board))
            {
                return true;
            }
            board[i * BOARDSIZE + j] = BLANK;
        }
    }
    return false;
}

int main()
{
    // const int arraySize = 6;
    // const int a[arraySize] = { 1, 2, 3, 4, 5, 6 };
    // //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    // int c[arraySize] = { 0 };

    // Add vectors in parallel.
    //cudaError_t cudaStatus = ReduceWithCuda(c, a, arraySize);
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "addWithCuda failed!");
    //     return 1;
    // }

    // printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //     c[0], c[1], c[2], c[3], c[4]);

    // // cudaDeviceReset must be called before exiting in order for profiling and
    // // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // cudaStatus = cudaDeviceReset();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaDeviceReset failed!");
    //     return 1;
    // }
    int board[BOARDSIZE*BOARDSIZE];

    readSudokuFromFile("hard.in", board);
    printSudoku(board);
    std::cout << "Solving sudoku..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto result = solveBacktracking(board);
    auto stop = std::chrono::high_resolution_clock::now();
    if(result)
    {
        std::cout << "Sudoku solved!" << std::endl;
        printSudoku(board);
    }
    else
    {
        std::cout << "Could not solve sudoku :(" << std::endl;
    }

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Total time for solving sudoku: " << duration.count() << " microseconds" << std::endl;
    return 0;
}

cudaError_t ReduceWithCuda(int* out, const int* a, unsigned int size)
{
    int* dev_input = 0;
    int* dev_output = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);
    if (cudaSetDevice(0) != cudaSuccess) ERR("cudaSetDevice");

    // Allocate GPU buffers for three vectors (two input, one output)    .
    if (cudaMalloc((void**)&dev_output, size * sizeof(int)) != cudaSuccess) ERR("cudaMalloc");    
    if (cudaMalloc((void**)&dev_input, size * sizeof(int)) != cudaSuccess) ERR("cudaMalloc");

    // Copy input vectors from host memory to GPU buffers.
    if (cudaMemcpy(dev_input, a, size * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) ERR("cudaMemcpy");

    reduce0<<<1, size>>>(dev_input, dev_output);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
    if (cudaDeviceSynchronize() != cudaSuccess) ERR("cudaDeviceSynchronize");


Error:
    cudaFree(dev_input);
    cudaFree(dev_output);

    return cudaErrorAlreadyAcquired;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
