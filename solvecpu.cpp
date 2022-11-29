
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <chrono>

#define BOARDSIZE 9
#define BLANK 0

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

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "USAGE: solvecpu filepath" << std::endl;
        return 1;
    }
    int board[BOARDSIZE*BOARDSIZE];

    readSudokuFromFile(argv[1], board);
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
