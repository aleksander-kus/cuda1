
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
