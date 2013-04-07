all: kratos

kratos:
	pwd
	mpicc ./src/matrix_multiply.c -o ./bin/matrix_multiply
	mpirun -n 8 bin/matrix_multiply 8192 2

quick:
	mpicc src/matrix_multiply.c -o bin/matrix_multiply
	mpirun -n 8 bin/matrix_multiply 1024 4

verify:
	mpicc src/matrix_multiply.c -o bin/matrix_multiply
	mpirun -n 8 bin/matrix_multiply 1024 2 --print_matrix

bgq:
	mpicc src/matrix_multiply.c -o bin/matrix_multiply -DCLOCK_RATE=1666700000.0

clean:
	rm -rf ./bin/*