all:
	nvcc -arch=sm_21 -Xcompiler -fopenmp -o test_gpu testbench_for_gpu.cu -lrt -lm
clean:
	rm -f test_gpu

