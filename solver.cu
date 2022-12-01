
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define BOARDSIZE 9
#define BOARDLENGTH 81
#define BLANK 0

// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// __global__ void reduce0(int* g_idata, int* g_odata) {
//     extern __shared__ int sdata[];
//     // each thread loads one element from global to shared mem
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     sdata[tid] = g_idata[i];
//     __syncthreads();
//     // do reduction in shared mem
//     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//         if (tid % (2 * s) == 0) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }
//     // write result for this block to global mem
//     if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

// __global__ void addKernel(int *c, const int *a, const int *b)
// {
//     int i = threadIdx.x;
//     c[i] = a[i] + b[i];
// }

void readSudokuFromFile(std::string filepath, char* board)
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

void printSudoku(char* board)
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
                std::cout << (int)value << ' ';
            }
        }
        std::cout << columnBreak << std::endl;
    }
    std::cout << lineBreak;
}

__device__ bool findEmpty(const char* board, int& i, int& j)
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

__device__ bool checkIfCorrectRow(const char* board, const int& i, const char& value)
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

__device__ bool checkIfCorrectColumn(const char* board, const int& j, const char& value)
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

__device__ bool checkIfCorrectBox(const char* board, const int& i, const int& j, const char& value)
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


__device__ bool checkIfCorrect(char* board, int i, int j, char value)
{
    return checkIfCorrectRow(board, i, value) && checkIfCorrectColumn(board, j, value) && checkIfCorrectBox(board, i, j, value);
}

__device__ void copyBoardToOutput(const char* board, char* output)
{
    for(int i = 0; i < BOARDSIZE; ++i)
    {
        for(int j = 0; j < BOARDSIZE; ++j)
        {
            output[i * BOARDSIZE + j] = board[i * BOARDSIZE + j];
        }
    }
}

enum GENERATE_STATUS {
    OK = 0,
    SOLVED = 1,
    FAILURE = 2,
    OUT_OF_MEMORY = 3
};

__global__ void generate(char* input, char* output, int inputSize, int* outputIndex, int maxOutputSize, GENERATE_STATUS* status)
{
    auto id = blockDim.x * blockIdx.x + threadIdx.x;

    while(id < inputSize && *status == OK)
    {
        # if __CUDA_ARCH__>=200
            //printf("%d \n", id);
        #endif 
        int i = 0, j = 0;

        auto tab = input + id * BOARDLENGTH; // set the correct input board according to threadIdx
            # if __CUDA_ARCH__>=200
                //printf("Value at 0, 1: %d \n", tab[1]);
            #endif 
        if(!findEmpty(tab, i, j))
        {
            *status = SOLVED;
            return;
        }
        // generate a separate board for all numbers available in the empty spot
        for(int num = 1; num < 10; ++num)
        {
            if(*outputIndex >= maxOutputSize - 1)
            {
                *status = OUT_OF_MEMORY;
                return;
            }
            # if __CUDA_ARCH__>=200
                //printf("Testing id %d %d, value %d \n", i, j, num);
            #endif 
            if(checkIfCorrect(tab, i, j, num))
            {
                            # if __CUDA_ARCH__>=200
                //printf("OK, writing id %d %d, value %d \n", i, j, num);
            #endif 
                tab[i * BOARDSIZE + j] = num;
                copyBoardToOutput(tab, output + atomicAdd(outputIndex, 1) * BOARDLENGTH);
                tab[i * BOARDSIZE + j] = BLANK;
            }
        }
        id += gridDim.x * blockDim.x;
    }
}

bool solveGpu(char* board)
{
    char *dev_input = 0, *dev_output = 0;
    int* dev_outputIndex = 0;
    GENERATE_STATUS* dev_status;
    GENERATE_STATUS status;
    int maxOutputSize = 100;
    int inputSize = 1;
    int generation = 0;
    int grids = 1024;
    int blocks = 512;

    ERR(cudaMalloc(&dev_input, sizeof(char) * BOARDLENGTH * maxOutputSize));
    ERR(cudaMalloc(&dev_output, sizeof(char) * BOARDLENGTH * maxOutputSize));
    ERR(cudaMalloc(&dev_outputIndex, sizeof(int)));
    ERR(cudaMalloc(&dev_status, sizeof(int)));

    ERR(cudaMemcpy(dev_input, board, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_output, 0, sizeof(char) * BOARDLENGTH * maxOutputSize));
    std::cout << "Solving sudoku..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    while(generation < 60)
    {
        std::cout << "Running with input size " << inputSize << std::endl;

        ERR(cudaMemset(dev_outputIndex, 0, sizeof(int)));
        if(generation % 2 == 0)
        {
            // char test[BOARDLENGTH*11];
            // ERR(cudaMemcpy(test, dev_input, sizeof(char) * BOARDLENGTH * 11, cudaMemcpyKind::cudaMemcpyDeviceToHost));
            // for(int i = 0; i < 10; ++i)
            // {
            //     printSudoku(test + BOARDLENGTH * i);
            // }
            generate<<<grids, blocks>>>(dev_input, dev_output, inputSize, dev_outputIndex, maxOutputSize, dev_status);
            cudaDeviceSynchronize();
        }
        else
        {
            // char test[BOARDLENGTH*11];
            // ERR(cudaMemcpy(test, dev_output, sizeof(char) * BOARDLENGTH * 11, cudaMemcpyKind::cudaMemcpyDeviceToHost));
            // for(int i = 0; i < 10; ++i)
            // {
            //     printSudoku(test + BOARDLENGTH * i);
            // }
            generate<<<grids, blocks>>>(dev_output, dev_input, inputSize, dev_outputIndex, maxOutputSize, dev_status);
            cudaDeviceSynchronize();
        }
        ERR(cudaMemcpy(&inputSize, dev_outputIndex, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(&status, dev_status, sizeof(GENERATE_STATUS), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        if(status != OK || inputSize == 0)
            break;
        ++generation;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Total time for generating boards: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Finished generating boards with " << inputSize << " boards on generation " << generation << std::endl;

    auto result = generation % 2 == 0 ? dev_input : dev_output;
    if (status == SOLVED)
    {
        std::cout << "Sudoku solved by BFS!" << std::endl;
        char output[BOARDLENGTH];
        ERR(cudaMemcpy(output, result, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        printSudoku(output);
    }
    else if (inputSize == 0)
    {
        std::cout << "No valid solutions were found for sudoku" << std::endl;
    }
    else if (status == OUT_OF_MEMORY)
    {
        std::cout << "No bueno amigo, out of memory excepcione nicht gut jajajajaj";
    }


    ERR(cudaFree(dev_input));
    ERR(cudaFree(dev_output));
    ERR(cudaFree(dev_outputIndex));
    ERR(cudaFree(dev_status));

    return true;
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
    // size_t free_memory;
	// cudaMemGetInfo(&free_memory, nullptr);
    // std::cout << "memory " << free_memory;
    // return 0;
    
    // data in our board will always be from range <1, 9>, so we use chars as they use only 1B of memory
    char board[BOARDLENGTH];


    readSudokuFromFile("test1.in", board);
    printSudoku(board);
    //std::cout << "Solving sudoku..." << std::endl;
    //auto start = std::chrono::high_resolution_clock::now();
    auto result = solveGpu(board);
    // auto stop = std::chrono::high_resolution_clock::now();
    // if(result)
    // {
    //     std::cout << "Sudoku solved!" << std::endl;
    //     printSudoku(board);
    // }
    // else
    // {
    //     std::cout << "Could not solve sudoku :(" << std::endl;
    // }

    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Total time for solving sudoku: " << duration.count() << " microseconds" << std::endl;
    return 0;
}

// cudaError_t ReduceWithCuda(int* out, const int* a, unsigned int size)
// {
//     int* dev_input = 0;
//     int* dev_output = 0;

//     // Choose which GPU to run on, change this on a multi-GPU system.
//     cudaSetDevice(0);
//     if (cudaSetDevice(0) != cudaSuccess) ERR("cudaSetDevice");

//     // Allocate GPU buffers for three vectors (two input, one output)    .
//     if (cudaMalloc((void**)&dev_output, size * sizeof(int)) != cudaSuccess) ERR("cudaMalloc");    
//     if (cudaMalloc((void**)&dev_input, size * sizeof(int)) != cudaSuccess) ERR("cudaMalloc");

//     // Copy input vectors from host memory to GPU buffers.
//     if (cudaMemcpy(dev_input, a, size * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) ERR("cudaMemcpy");

//     reduce0<<<1, size>>>(dev_input, dev_output);

//     if (cudaGetLastError() != cudaSuccess) {
//         fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
//         goto Error;
//     }

//     // cudaDeviceSynchronize waits for the kernel to finish, and returns
// // any errors encountered during the launch.
//     if (cudaDeviceSynchronize() != cudaSuccess) ERR("cudaDeviceSynchronize");


// Error:
//     cudaFree(dev_input);
//     cudaFree(dev_output);

//     return cudaErrorAlreadyAcquired;
// }

// // Helper function for using CUDA to add vectors in parallel.
// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
// {
//     int *dev_a = 0;
//     int *dev_b = 0;
//     int *dev_c = 0;
//     cudaError_t cudaStatus;

//     // Choose which GPU to run on, change this on a multi-GPU system.
//     cudaStatus = cudaSetDevice(0);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//         goto Error;
//     }

//     // Allocate GPU buffers for three vectors (two input, one output)    .
//     cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMalloc failed!");
//         goto Error;
//     }

//     cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMalloc failed!");
//         goto Error;
//     }

//     cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMalloc failed!");
//         goto Error;
//     }

//     // Copy input vectors from host memory to GPU buffers.
//     cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

//     cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

//     // Launch a kernel on the GPU with one thread for each element.
//     addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

//     // Check for any errors launching the kernel
//     cudaStatus = cudaGetLastError();
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//         goto Error;
//     }
    
//     // cudaDeviceSynchronize waits for the kernel to finish, and returns
//     // any errors encountered during the launch.
//     cudaStatus = cudaDeviceSynchronize();
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//         goto Error;
//     }

//     // Copy output vector from GPU buffer to host memory.
//     cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

// Error:
//     cudaFree(dev_c);
//     cudaFree(dev_a);
//     cudaFree(dev_b);
    
//     return cudaStatus;
// }
