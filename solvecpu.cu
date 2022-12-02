
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

#include "solvecpu.cuh"
#include "solvegpu.cuh"

bool solveBacktracking(char* board)
{
    int i = 0;
    int j = 0;

    if(!findEmpty(board, i, j))
    {
        return true;
    }

    for(int num = 1; num < 10; ++num)
    {
        if(tryToInsert(board, i, j, num))
        {
            board[i * BOARDSIZE + j] = num;
            if(solveBacktracking(board))
            {
                return true;
            }
            board[i * BOARDSIZE + j] = BLANK;
        }
    }
    return false;
}

void getEmpty(const char* board, char* indices, char* size)
{
    for(char i = 0; i < BOARDLENGTH; ++i)
    {
        if(board[i] == BLANK)
        {
            indices[*size] = i;
            ++(*size);
        }
    }
}

// bool solveBacktracking(char* input)
// {
//     char emptyIndices[BOARDLENGTH];
//     char emptyIndicesSize = 0;
//     int i = 0, j = 0;


//     auto board = input;
//     emptyIndicesSize = 0;
//     getEmpty(board, emptyIndices, &emptyIndicesSize);
//     std::cout << (int)emptyIndicesSize << std::endl;
//     fflush(stdout);
//     for(int i = 0; i < emptyIndicesSize; ++i)
//     {
//         std::cout << (int)emptyIndices[i] << ' ';
//     }
//     fflush(stdout);
//     int index = 0;
//     while(index >= 0 && index < emptyIndicesSize)
//     {
//         auto emptyIndex = emptyIndices[index];
//         i = emptyIndex / BOARDSIZE;
//         j = emptyIndex % BOARDSIZE;
//         printf("Scanning index %d, i = %d, j = %d, value %d \n", emptyIndex, i, j, board[emptyIndex] + 1);
//         if(!tryToInsert(board, i, j, board[emptyIndex] + 1))
//         {
//             if(board[emptyIndex] >= 8)
//             {
//                 board[emptyIndex] = -1;
//                 --index;
//             }
//         }
//         else
//         {
//             ++index;
//         }
//         ++board[emptyIndex];
//         if(board[emptyIndex] > 9)
//         {
//             return true;
//         }
//     }

//     return index == emptyIndicesSize;
// }

char* solveCpu(const char* board)
{
    char* board2 = (char*)malloc(sizeof(char) * BOARDLENGTH);
    memcpy(board2, board, sizeof(char) * BOARDLENGTH);
    auto result = solveBacktracking(board2);
    if(result)
    {
        return board2;
    }
    else
    {
        free(board2);
        return nullptr;
    }
}
