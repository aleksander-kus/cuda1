#include "cuda_runtime.h"

#include <iostream>
#include <chrono>

#include "solvegpu.cuh"

#define MEMORY_USED 0.2
#define GENERATION_LIMIT 81
#define BLOCKS 2048
#define THREADS 1024

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

enum STATUS {
    OK = 0,
    SOLVED = 1,
    OUT_OF_MEMORY = 2
};

__host__ __device__ bool findEmpty(const char* board, int& i, int& j)
{
    for (int k = 0; k < BOARDSIZE; ++k)
    {
        for (int l = 0; l < BOARDSIZE; ++l)
        {
            if (board[k * BOARDSIZE + l] == 0)
            {
                i = k;
                j = l;
                return true;
            }
        }
    }
    return false;
}

__host__ __device__ bool tryToInsertRow(const char* board, const int& i, const char& value)
{
    for (int j = 0; j < BOARDSIZE; ++j)
    {
        if (board[i * BOARDSIZE + j] == value)
        {
            return false;
        }
    }
    return true;
}

__host__ __device__ bool tryToInsertColumn(const char* board, const int& j, const char& value)
{
    for (int i = 0; i < BOARDSIZE; ++i)
    {
        if (board[i * BOARDSIZE + j] == value)
        {
            return false;
        }
    }
    return true;
}

__host__ __device__ bool tryToInsertBox(const char* board, const int& i, const int& j, const char& value)
{
    int rowCenter = (i / 3) * 3 + 1;
    int columnCenter = (j / 3) * 3 + 1;

    for (int k = -1; k < 2; ++k)
    {
        for (int l = -1; l < 2; ++l)
        {
            if (board[(rowCenter + k) * BOARDSIZE + (columnCenter + l)] == value)
            {
                return false;
            }
        }
    }
    return true;
}

__host__ __device__ bool tryToInsert(const char* board, const int& i, const int& j, const char& value)
{
    return value > 0 && value < 10 && tryToInsertRow(board, i, value) && tryToInsertColumn(board, j, value) && tryToInsertBox(board, i, j, value);
}

__device__ void copyBoardToOutput(const char* board, char* output)
{
    for (int i = 0; i < BOARDSIZE; ++i)
    {
        for (int j = 0; j < BOARDSIZE; ++j)
        {
            output[i * BOARDSIZE + j] = board[i * BOARDSIZE + j];
        }
    }
}

__global__ void generate(char* input, char* output, int inputSize, int* outputIndex, int maxOutputSize, STATUS* status)
{
    auto id = blockDim.x * blockIdx.x + threadIdx.x;

    while (id < inputSize && *status == OK)
    {
        int i = 0, j = 0;

        auto board = input + id * BOARDLENGTH; // set the correct input board according to threadIdx
        if (!findEmpty(board, i, j))
        {
            *status = SOLVED;
            return;
        }
        // generate a separate board for all numbers available in the empty spot
        for (int num = 1; num < 10; ++num)
        {
            if (*outputIndex >= maxOutputSize - 1)
            {
                *status = OUT_OF_MEMORY;
                return;
            }
            if (tryToInsert(board, i, j, num))
            {
                board[i * BOARDSIZE + j] = num;
                copyBoardToOutput(board, output + atomicAdd(outputIndex, 1) * BOARDLENGTH);
                board[i * BOARDSIZE + j] = BLANK;
            }
        }
        id += gridDim.x * blockDim.x;
    }
}

__device__ void getEmptyIndices(const char* board, char* indices, char* size)
{
    for (char i = 0; i < BOARDLENGTH; ++i)
    {
        if (board[i] == BLANK)
        {
            indices[*size] = i;
            ++(*size);
        }
    }
}

__device__ bool backtrackBoard(char* board, char* emptyIndices)
{
    char emptyIndicesSize = 0;
    getEmptyIndices(board, emptyIndices, &emptyIndicesSize);
    char index = 0, i = 0, j = 0;
    while(index >= 0 && index < emptyIndicesSize)
    {
        auto emptyIndex = emptyIndices[index];
        i = emptyIndex / BOARDSIZE;
        j = emptyIndex % BOARDSIZE;
        // #if __CUDA_ARCH__>=200
        //     printf("Scanning index %d, i = %d, j = %d, value %d \n", emptyIndex, i, j, board[emptyIndex] + 1);
        // #endif
        if (!tryToInsert(board, i, j, board[emptyIndex] + 1))
        {
            if (board[emptyIndex] >= 8)
            {
                board[emptyIndex] = -1;
                --index;
            }
        }
        else
        {
            ++index;
        }
        ++board[emptyIndex];
    }

    return index == emptyIndicesSize;
}

__global__ void backtrack(char* input, char* output, int inputSize, STATUS* status)
{
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    char emptyIndices[BOARDLENGTH];

    while(id < inputSize && *status != SOLVED)
    {
        auto board = input + id * BOARDLENGTH;
        
        if(backtrackBoard(board, emptyIndices))
        {
            *status = SOLVED;
            copyBoardToOutput(board, output);
            return;
        }

        id += gridDim.x * blockDim.x;
    }
}

bool solveBoard(char* board, char* dev_input, char* dev_output, int* dev_outputIndex, STATUS* dev_status, const int& maxBoardCount)
{
    int inputSize = 1;
    int oldInputSize = 1;
    int generation = 0;
    STATUS status;

    ERR(cudaMemcpy(dev_input, board, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_output, 0, sizeof(char) * BOARDLENGTH * maxBoardCount));

    auto start = std::chrono::high_resolution_clock::now();
    while(generation < GENERATION_LIMIT)
    {
        ERR(cudaMemset(dev_outputIndex, 0, sizeof(int)));
        if (generation % 2 == 0)
        {
            generate<<<BLOCKS, THREADS>>>(dev_input, dev_output, inputSize, dev_outputIndex, maxBoardCount, dev_status);
        }
        else
        {
            generate<<<BLOCKS, THREADS>>>(dev_output, dev_input, inputSize, dev_outputIndex, maxBoardCount, dev_status);
        }
        oldInputSize = inputSize;
        ERR(cudaMemcpy(&inputSize, dev_outputIndex, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(&status, dev_status, sizeof(STATUS), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ++generation;
        if (status != OK || inputSize == 0)
            break;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Generating boards took: " << duration.count() << " microseconds" << std::endl;

    if (status == SOLVED)
    {
        auto generationResult = generation % 2 == 1 ? dev_input : dev_output; // take the output as result
        ERR(cudaMemcpy(board, generationResult, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        return true;
    }
    else if (inputSize == 0)
    {
        std::cout << "No valid solutions were found for sudoku" << std::endl;
        return false;
    }
    else
    {
        auto generationResult = generation % 2 == 1 ? dev_input : dev_output; // take the last input as result
        auto output = generation % 2 == 0 ? dev_input : dev_output; // reuse the other array as output

        start = std::chrono::high_resolution_clock::now();

        backtrack<<<BLOCKS, THREADS>>>(generationResult, output, oldInputSize, dev_status);
        ERR(cudaMemcpy(&status, dev_status, sizeof(STATUS), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        if (status != SOLVED)
        {
            std::cout << "No valid solutions were found for sudoku" << std::endl;
            return false;
        }
        ERR(cudaMemcpy(board, output, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyDeviceToHost));

        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Backtracking took: " << duration.count() << " microseconds" << std::endl;
        return true;
    }
}

int getMaxBoardCount()
{
    size_t free_memory;
	cudaMemGetInfo(&free_memory, nullptr);
    return free_memory * MEMORY_USED / (sizeof(char) * BOARDLENGTH * 2);
}

char* solveGpu(const char* board)
{
    auto start = std::chrono::high_resolution_clock::now();
    int maxBoardCount = getMaxBoardCount();
    char *dev_input = 0, *dev_output = 0;
    int* dev_outputIndex = 0;
    STATUS* dev_status;    

    ERR(cudaMalloc(&dev_input, sizeof(char) * BOARDLENGTH * maxBoardCount));
    ERR(cudaMalloc(&dev_output, sizeof(char) * BOARDLENGTH * maxBoardCount));
    ERR(cudaMalloc(&dev_outputIndex, sizeof(int)));
    ERR(cudaMalloc(&dev_status, sizeof(int)));

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "GPU initialization took: " << duration.count() << " microseconds" << std::endl;

    char* copy = new char[BOARDLENGTH];
    memcpy(copy, board, BOARDLENGTH);

    auto result = solveBoard(copy, dev_input, dev_output, dev_outputIndex, dev_status, maxBoardCount);

    ERR(cudaFree(dev_input));
    ERR(cudaFree(dev_output));
    ERR(cudaFree(dev_outputIndex));
    ERR(cudaFree(dev_status));

    return result ? copy : nullptr;
}