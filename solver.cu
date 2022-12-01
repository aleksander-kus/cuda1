#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bitset>
#include "solvecpu.cuh"
#include "solvegpu.cuh"

#define ERR(status) { \
    if (status != cudaSuccess) { \
        printf("Error: %s, file: %s, line: %d\n", cudaGetErrorString(status), __FILE__,__LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


#define MEMORY_USED 0.2

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



bool checkIfSudokuValid(const char* board)
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

    // check boxes
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
        std::cout << "USAGE: solver.out filepath" << std::endl;
        return 1;
    }

    // data in our board will always be from range <1, 9>, so we use chars as they use only 1B of memory
    char board[BOARDLENGTH];

    readSudokuFromFile(argv[1], board);
    printSudoku(board);

    if(!checkIfSudokuValid(board))
    {
        std::cout << "Given sudoku is invalid" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Solving sudoku cpu..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto result = solveCpu(board);
    auto stop = std::chrono::high_resolution_clock::now();
    if(result != nullptr)
    {
        std::cout << "Cpu solution: " << std::endl;
        printSudoku(result);
        free(result);
    }
    else
    {
        std::cout << "Cpu did not find a solution" << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Total time for cpu: " << duration.count() << " microseconds" << std::endl;

    std::cout << "Solving sudoku gpu..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    result = solveGpu(board);
    stop = std::chrono::high_resolution_clock::now();
    if(result != nullptr)
    {
        std::cout << "Gpu solution: " << std::endl;
        printSudoku(result);
        free(result);
    }
    else
    {
        std::cout << "Gpu did not find a solution" << std::endl;
    }
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Total time for gpu: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
