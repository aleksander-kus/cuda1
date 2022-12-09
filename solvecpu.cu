
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

#include "solvecpu.cuh"
#include "solvegpu.cuh"

bool solveBacktracking(char* board)
{
    int i = 0, j = 0;

    if (!findEmpty(board, i, j))
    {
        return true;
    }

    for(int num = 1; num < 10; ++num)
    {
        if (tryToInsert(board, i, j, num))
        {
            board[i * BOARDSIZE + j] = num;
            if (solveBacktracking(board))
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
    char* copy = new char[BOARDLENGTH];
    memcpy(copy, board, sizeof(char) * BOARDLENGTH);
    auto result = solveBacktracking(copy);
    if (result)
    {
        return copy;
    }
    else
    {
        free(copy);
        return nullptr;
    }
}
