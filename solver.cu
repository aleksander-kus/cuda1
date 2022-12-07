#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <bitset>
#include <getopt.h>
#include "solvecpu.cuh"
#include "solvegpu.cuh"

void readSudokuFromFile(std::string filepath, char* board)
{
    std::ifstream fileStream(filepath);
    std::string input;
    int index = 0;
    while (getline(fileStream, input))
    {
        for (auto c : input)
        {
            board[index++] = c - '0';
        }
    }
}

void printSudoku(char* board)
{
    const std::string lineBreak = "+-------+-------+-------+\n";
    const std::string columnBreak = "| ";

    for (auto i = 0; i < BOARDSIZE; ++i)
    {
        if (i % 3 == 0)
        {
            std::cout << lineBreak;
        }
        for (auto j = 0; j < BOARDSIZE; ++j)
        {
            if (j % 3 == 0)
            {
                std::cout << columnBreak;
            }

            auto value = board[i * BOARDSIZE + j];
            if (value == BLANK)
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
    for (int i = 0; i < BOARDSIZE; ++i)
    {
        for (int j = 0; j < BOARDSIZE; ++j)
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
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int rowCenter = (i / 3) * 3 + 1;
            int columnCenter = (j / 3) * 3 + 1;
            for (int k = -1; k < 2; ++k)
            {
                for (int l = -1; l < 2; ++l)
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

void compareResults(char* cpu, char* gpu)
{
    bool ok = true;
    for (int i = 0; i < BOARDSIZE; ++i)
    {
        for (int j = 0; j < BOARDSIZE; ++j)
        {
            if (cpu[i * BOARDSIZE + j] != gpu[i * BOARDSIZE + j])
            {
                ok = false;
            }
        }
    }
    if (ok)
    {
        std::cout << "Cpu and gpu results match" << std::endl;
    }
    else
    {
        std::cout << "Cpu and gpu results don't match" << std::endl;
    }
}

void usage()
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  solver.out [options] filepath" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -c, --cpu-only      only run cpu algorithm" << std::endl;
    std::cout << "  -g, --gpu-only      only run gpu algorithm" << std::endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    int c;
    bool isCpuOnly = false;
    bool isGpuOnly = false;
    static struct option long_options[] = {
        {"cpu-only", no_argument, NULL, 'c'},
        {"gpu-only", no_argument, NULL, 'g'},
        { NULL, 0, NULL, 0 }
    };

    while (1)
    {
        c = getopt_long(argc, argv, "cg", long_options, NULL);
        if(c == -1)
            break;

        switch(c)
        {
            case 'c':
                isCpuOnly = true;
                break;
            case 'g':
                isGpuOnly = true;
                break;
            default:
                usage();
                break;
        }
    }

    if (optind != argc - 1) {
        usage();
    }

    std::string filepath = argv[optind++];

    if(isCpuOnly && isGpuOnly)
    {
        std::cout << "The -c and -g flags are mutually exclusive" << std::endl;
        exit(EXIT_FAILURE);
    }

    // data in our board will always be from range <1, 9>, so we use chars as they use only 1B of memory
    char board[BOARDLENGTH];

    readSudokuFromFile(filepath, board);
    printSudoku(board);

    if(!checkIfSudokuValid(board))
    {
        std::cout << "Given sudoku is invalid" << std::endl;
        exit(EXIT_FAILURE);
    }

    char* resultCpu = nullptr, *resultGpu = nullptr;

    if(!isGpuOnly)
    {
        std::cout << "Solving sudoku cpu..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        resultCpu = solveCpu(board);
        auto stop = std::chrono::high_resolution_clock::now();
        if(resultCpu != nullptr)
        {
            std::cout << "Cpu solution: " << std::endl;
            printSudoku(resultCpu);
        }
        else
        {
            std::cout << "Cpu did not find a solution" << std::endl;
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time for cpu: " << duration.count() << " microseconds" << std::endl;
    }

    if(!isCpuOnly)
    {
        std::cout << "Solving sudoku gpu..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        resultGpu = solveGpu(board);
        auto stop = std::chrono::high_resolution_clock::now();
        if(resultGpu != nullptr)
        {
            std::cout << "Gpu solution: " << std::endl;
            printSudoku(resultGpu);
        }
        else
        {
            std::cout << "Gpu did not find a solution" << std::endl;
        }
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total time for gpu: " << duration.count() << " microseconds" << std::endl;
    }

    if(!isCpuOnly && !isGpuOnly)
    {
        compareResults(resultCpu, resultGpu);
    }

    if( resultCpu != nullptr)
    {
        delete[] resultCpu;
    }
    if (resultGpu != nullptr)
    {
        delete[] resultGpu;
    }

    return 0;
}
