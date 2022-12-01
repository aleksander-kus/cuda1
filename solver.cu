
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <bitset>

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define BOARDSIZE 9
#define BOARDLENGTH 81
#define BLANK 0
#define MEMORY_USED 0.1

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

__device__ bool tryToInsertRow(const char* board, const int& i, const char& value)
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

__device__ bool tryToInsertColumn(const char* board, const int& j, const char& value)
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

__device__ bool tryToInsertBox(const char* board, const int& i, const int& j, const char& value)
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


__device__ bool tryToInsert(char* board, int i, int j, char value)
{
    return tryToInsertRow(board, i, value) && tryToInsertColumn(board, j, value) && tryToInsertBox(board, i, j, value);
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
    OUT_OF_MEMORY = 2
};

__global__ void generate(char* input, char* output, int inputSize, int* outputIndex, int maxOutputSize, GENERATE_STATUS* status)
{
    auto id = blockDim.x * blockIdx.x + threadIdx.x;

    while(id < inputSize && *status == OK)
    {
        int i = 0, j = 0;

        auto tab = input + id * BOARDLENGTH; // set the correct input board according to threadIdx
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
            if(tryToInsert(tab, i, j, num))
            {
                tab[i * BOARDSIZE + j] = num;
                copyBoardToOutput(tab, output + atomicAdd(outputIndex, 1) * BOARDLENGTH);
                tab[i * BOARDSIZE + j] = BLANK;
            }
        }
        id += gridDim.x * blockDim.x;
    }
}

int getMaxBoardNumber()
{
    size_t free_memory;
	cudaMemGetInfo(&free_memory, nullptr);
    return free_memory * MEMORY_USED / (sizeof(char) * BOARDLENGTH * 2);
}

bool solveGpu(char* board)
{
    int maxBoardNumber = getMaxBoardNumber();
    char *dev_input = 0, *dev_output = 0;
    int* dev_outputIndex = 0;
    GENERATE_STATUS* dev_status;
    GENERATE_STATUS status;
    int inputSize = 1;
    int oldInputSize = 1;
    int generation = 0;
    int grids = 2048;
    int blocks = 1024;

    ERR(cudaMalloc(&dev_input, sizeof(char) * BOARDLENGTH * maxBoardNumber));
    ERR(cudaMalloc(&dev_output, sizeof(char) * BOARDLENGTH * maxBoardNumber));
    ERR(cudaMalloc(&dev_outputIndex, sizeof(int)));
    ERR(cudaMalloc(&dev_status, sizeof(int)));

    ERR(cudaMemcpy(dev_input, board, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_output, 0, sizeof(char) * BOARDLENGTH * maxBoardNumber));
    std::cout << "Solving sudoku..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    while(generation < 15)
    {
        std::cout << "Running with input size " << inputSize << std::endl;

        ERR(cudaMemset(dev_outputIndex, 0, sizeof(int)));
        if(generation % 2 == 0)
        {
            generate<<<grids, blocks>>>(dev_input, dev_output, inputSize, dev_outputIndex, maxBoardNumber, dev_status);
            cudaDeviceSynchronize();
        }
        else
        {
            generate<<<grids, blocks>>>(dev_output, dev_input, inputSize, dev_outputIndex, maxBoardNumber, dev_status);
            cudaDeviceSynchronize();
        }
        oldInputSize = inputSize;
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

    if (status == SOLVED)
    {
        std::cout << "Sudoku solved by BFS!" << std::endl;
        auto result = generation % 2 == 0 ? dev_input : dev_output; // take the output as result
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
        std::cout << "Available memory exceeded, falling back to last generation of boards" << std::endl;
        std::cout << "Generation " << generation - 1 << " with " << oldInputSize << " boards" << std::endl;
        auto result = generation % 2 == 1 ? dev_input : dev_output; // take the last input as result
    }


    ERR(cudaFree(dev_input));
    ERR(cudaFree(dev_output));
    ERR(cudaFree(dev_outputIndex));
    ERR(cudaFree(dev_status));

    return true;
}

bool checkIfValid(const char* board)
{
    std::bitset<10> bitset;
    // check rows
    for(int i = 0; i < BOARDSIZE; ++i)
    {
        for(int j = 0; j < BOARDSIZE; ++j)
        {
            auto value = board[i * BOARDSIZE + j];
            if (value == BLANK)
            {
                continue;
            }
            if (bitset.test(value))
            {
                return false;
            }
            bitset.set(value, true);
        }
        bitset.reset();
    }

    // check columns
    for (int j = 0; j < BOARDSIZE; ++j)
    {
        for (int i = 0; i < BOARDSIZE; ++i)
        {
            auto value = board[i * BOARDSIZE + j];
            if (value == BLANK)
            {
                continue;
            }
            if (bitset.test(value))
            {
                return false;
            }
            bitset.set(value, true);
        }
        bitset.reset();
    }


    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            int rowCenter = (i / 3) * 3 + 1;
            int columnCenter = (j / 3) * 3 + 1;
            for(int k = -1; k < 2; ++k)
            {
                for(int l = -1; l < 2; ++l)
                {
                    auto value = board[(rowCenter + k) * BOARDSIZE + (columnCenter + l)];
                    if (value == BLANK)
                    {
                        continue;
                    }
                    if (bitset.test(value))
                    {
                        return false;
                    }
                    bitset.set(value, true);
                }
            }
            bitset.reset();
        }
    }



    return true;
}

int main(int argc, char** argv)
{    
    if(argc != 2)
    {
        std::cout << "USAGE: solvecpu filepath" << std::endl;
        return 1;
    }

    // data in our board will always be from range <1, 9>, so we use chars as they use only 1B of memory
    char board[BOARDLENGTH];

    readSudokuFromFile(argv[1], board);
    printSudoku(board);

    if(!checkIfValid(board))
    {
        std::cout << "Given sudoku is invalid" << std::endl;
        exit(EXIT_FAILURE);
    }
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
