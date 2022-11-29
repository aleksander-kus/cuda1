all: solver solvecpu

solver:
	nvcc -std=c++11 -o solver.out solver.cu

solvecpu:
	g++ -o solvecpu.out solvecpu.cpp