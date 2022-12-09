#include "cuda_runtime.h"

#include <iostream>
#include <chrono>

#include "solvegpu.cuh"

#define MEMORY_USED 0.2
#define GENERATION_LIMIT 81
#define GRID_SIZE 2048
#define BLOCK_SIZE 1024

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

__device__ void generateBoards(char* board, char* output, int* outputIndex, int maxOutputSize, STATUS* status)
{
    int i = 0, j = 0;

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
}

__global__ void generate(char* input, char* output, int inputLength, int* outputIndex, int maxOutputSize, STATUS* status)
{
    auto id = blockDim.x * blockIdx.x + threadIdx.x;

    while (id < inputLength && *status == OK)
    {
        auto board = input + id * BOARDLENGTH; // set the correct input board according to threadIdx

        generateBoards(board, output, outputIndex, maxOutputSize, status);

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
    // get the array of empty indices. we will try to fill them with correct values
    getEmptyIndices(board, emptyIndices, &emptyIndicesSize);
    char index = 0, i = 0, j = 0;
    while (index >= 0 && index < emptyIndicesSize)
    {
        // get the next empty space
        auto emptyIndex = emptyIndices[index];
        i = emptyIndex / BOARDSIZE;
        j = emptyIndex % BOARDSIZE;

        auto valid = tryToInsert(board, i, j, board[emptyIndex] + 1);
        ++board[emptyIndex];

        if (valid) // if the board after incrementation is valid, advance the index
        {
            ++index;
            continue;
        }
        if (board[emptyIndex] > 9) // if we tried all possible values in this space, revert the index
        {
            board[emptyIndex] = 0;
            --index;
        }
    }

    return index == emptyIndicesSize;
}

__global__ void backtrack(char* input, char* output, int inputLength, STATUS* status)
{
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    char emptyIndices[BOARDLENGTH];

    while(id < inputLength && *status != SOLVED)
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

inline char* getInputArray(int generation, char* array1, char* array2)
{
    return generation % 2 == 0 ? array1 : array2;
}

inline char* getOutputArray(int generation, char* array1, char* array2)
{
    return generation % 2 == 0 ? array2 : array1;
}

bool solveBoard(char* board, char* dev_array1, char* dev_array2, int* dev_outputLength, STATUS* dev_status, const int& maxBoardCount)
{
    int inputLength = 1;
    int oldInputLength = 1;
    int generation = 0;
    STATUS status;

    ERR(cudaMemcpy(dev_array1, board, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyHostToDevice));
    ERR(cudaMemset(dev_array2, 0, sizeof(char) * BOARDLENGTH * maxBoardCount));

    auto start = std::chrono::high_resolution_clock::now();
    while(generation < GENERATION_LIMIT)
    {
        ERR(cudaMemset(dev_outputLength, 0, sizeof(int)));
        auto inputArray = getInputArray(generation, dev_array1, dev_array2);
        auto outputArray = getOutputArray(generation, dev_array1, dev_array2);

        generate<<<GRID_SIZE, BLOCK_SIZE>>>(inputArray, outputArray, inputLength, dev_outputLength, maxBoardCount, dev_status);

        oldInputLength = inputLength;
        ERR(cudaMemcpy(&inputLength, dev_outputLength, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(&status, dev_status, sizeof(STATUS), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        ++generation;
        if (status != OK || inputLength == 0)
            break;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Generating boards took: " << duration.count() << " microseconds" << std::endl;

    if (status == SOLVED) // sudoku was solved only with generating boards
    {
        auto generationResult = getOutputArray(generation, dev_array1, dev_array2); // take the output as result
        ERR(cudaMemcpy(board, generationResult, sizeof(char) * BOARDLENGTH, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        return true;
    }
    else if (inputLength == 0) // generating boards ended preemptively, no solutions were found
    {
        std::cout << "No valid solutions were found for sudoku" << std::endl;
        return false;
    }
    else // try backtracking
    {
        auto generationResult = getInputArray(generation - 1, dev_array1, dev_array2); // take the last input as result
        auto output = getOutputArray(generation - 1, dev_array1, dev_array2); // reuse the other array as output

        start = std::chrono::high_resolution_clock::now();

        backtrack<<<GRID_SIZE, BLOCK_SIZE>>>(generationResult, output, oldInputLength, dev_status);
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
    char *dev_array1 = 0, *dev_array2 = 0;
    int* dev_outputLength = 0;
    STATUS* dev_status;    

    ERR(cudaMalloc(&dev_array1, sizeof(char) * BOARDLENGTH * maxBoardCount));
    ERR(cudaMalloc(&dev_array2, sizeof(char) * BOARDLENGTH * maxBoardCount));
    ERR(cudaMalloc(&dev_outputLength, sizeof(int)));
    ERR(cudaMalloc(&dev_status, sizeof(int)));

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "GPU initialization took: " << duration.count() << " microseconds" << std::endl;

    char* copy = new char[BOARDLENGTH];
    memcpy(copy, board, BOARDLENGTH);

    auto result = solveBoard(copy, dev_array1, dev_array2, dev_outputLength, dev_status, maxBoardCount);

    ERR(cudaFree(dev_array1));
    ERR(cudaFree(dev_array2));
    ERR(cudaFree(dev_outputLength));
    ERR(cudaFree(dev_status));

    return result ? copy : nullptr;
}