#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#include "rdtsc.h"
#include "mt19937.h"

#define CLOCK_RATE 2666700000.0; // change to 700000000.0 for Blue Gene/L

/* Structure for passing arguments to the thread function */
typedef struct args_struct {
  double** A;
  double** B;
  double** C;
  int size;         // Total size of matrix
  int frac;         // Total fraction the node is responsible for
  int thread_frac;  // Fraction this thread is responsible for
  int col_offset;
  int row_offset;
} thread_args;

/* Each individual thread runs this matrix multiply function */
void* matrix_multiply_thread (void* arg)
{
  int i = 0, j = 0, k = 0;
  thread_args* t_arg = (thread_args*)arg;
  
  /* i begins at row offset (depending on which thread this is) */
  for (i = t_arg->row_offset; i < t_arg->row_offset + t_arg->thread_frac; i++)
  {
    for (j = 0; j < t_arg->frac; j++)
    {
      for (k = 0; k < t_arg->size; k++)
      {
        /* j is offset depending on which B matrix is being worked on */
        t_arg->C[i][j + t_arg->col_offset] += t_arg->A[i][k] * t_arg->B[k][j];
      }
    }
  }
}

/* Delegates matrix multiplication pieces to threads */
void matrix_multiply_part (double** A, double** B, double** C, int size,
                           int frac, int col_offset, int num_threads,
                           unsigned long long* timer, int task_id)
{
  unsigned long long begin = rdtsc();

  int i = 0;

  pthread_t* threads = malloc(sizeof(pthread_t) * num_threads);
  thread_args** args = malloc(sizeof(thread_args*) * num_threads);

  int row_offset = 0;
  col_offset *= frac;

  int thread_offset = 0;
  int thread_frac = frac / num_threads;

  /* For each thread: create an arg struct, spawn thread, increase offset */
  for (i = 0; i < num_threads; i++)
  {
    args[i] = malloc(sizeof(thread_args));
    args[i]->A = A;
    args[i]->B = B;
    args[i]->C = C;
    args[i]->size = size;
    args[i]->frac = frac;
    args[i]->thread_frac = thread_frac;
    args[i]->col_offset = col_offset;
    args[i]->row_offset = row_offset;
    pthread_create(&threads[i], NULL, matrix_multiply_thread, args[i]);
    row_offset += thread_frac;
  }

  /* Wait for threads to terminate */
  for (i = 0; i < num_threads; i++)
  {
    pthread_join(threads[i], NULL);
    free(args[i]);
  }

  free(threads);
  free(args);

  unsigned long long end = rdtsc();

  *timer += (end - begin);
}

/* Allocates a (portion of a) matrix that is contiguous in memory */
double* alloc_matrix_contig(int rows, int cols) {
  double* data = (double *)calloc(rows * cols, sizeof(double));
  return data;
}

/* Creates a 2D array given matrix data by creating pointers to rows */
double** create_matrix_2d(double* matrix_data, int rows, int cols) {
  double** array = (double **)calloc(rows, sizeof(double*));

  int i;
  for (i = 0; i < rows; i++)
    array[i] = &(matrix_data[cols * i]);

  return array;
}

int main (int argc, char* argv[])
{
  int i, j;
  int task_id, num_tasks;
  int flag_send, flag_recv;
  
  MPI_Status status;
  MPI_Request buffer_send, buffer_recv;
  
  unsigned long long pre_send_recv = 0;
  unsigned long long send_complete = 0;
  unsigned long long recv_complete = 0;
  unsigned long long time_mult = 0, time_send = 0, time_recv = 0;
  unsigned long long min_mult = 0, max_mult = 0, avg_mult = 0;
  unsigned long long min_send = 0, max_send = 0, avg_send = 0;
  unsigned long long min_recv = 0, max_recv = 0, avg_recv = 0;
  unsigned long long prog_begin = 0, prog_end = 0, prog_total = 0;
  double min_mult_secs, max_mult_secs, avg_mult_secs;
  double min_send_secs, max_send_secs, avg_send_secs;
  double min_recv_secs, max_recv_secs, avg_recv_secs;

  double** A = NULL;
  double** B = NULL;
  double** C = NULL;
  double** B_buffer_addr = NULL;
  double** B_switch_addr = NULL;

  double* A_data = NULL;
  double* B_data = NULL;
  double* C_data = NULL;
  double* B_buffer = NULL;
  double* B_switch = NULL;

  prog_begin = rdtsc();

  if (argc < 3)
  {
    fprintf(stderr, "Usage: %s <matrix_size> <num_threads> [--print_matrix]\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  unsigned int matrix_size = atoi(argv[1]);
  unsigned int num_threads = atoi(argv[2]);
  unsigned long rng_init_seeds[6] = { 0x0, 0x123, 0x234, 0x345, 0x456, 0x789 };
  unsigned long rng_init_length = 6;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  rng_init_seeds[0] = task_id;
  init_by_array(rng_init_seeds, rng_init_length);

  int matrix_fraction = matrix_size / num_tasks;

  /* Initialize the three matrices */
  A_data = alloc_matrix_contig(matrix_fraction, matrix_size);
  B_data = alloc_matrix_contig(matrix_size, matrix_fraction);
  C_data = alloc_matrix_contig(matrix_fraction, matrix_size);

  A = create_matrix_2d(A_data, matrix_fraction, matrix_size);
  B = create_matrix_2d(B_data, matrix_size, matrix_fraction);
  C = create_matrix_2d(C_data, matrix_fraction, matrix_size);
  
  B_buffer = alloc_matrix_contig(matrix_size, matrix_fraction);
  B_buffer_addr = create_matrix_2d(B_buffer, matrix_size, matrix_fraction);

  for (i = 0; i < matrix_fraction; i++)
  {
    for (j = 0; j < matrix_size; j++)
    {
      A[i][j] = genrand_res53(); // (double)i*j;
      B[j][i] = genrand_res53(); // A[i][j] * A[i][j];
    }
  }

  /* For each MPI task, pass around the B matrix */
  for (i = 0; i < num_tasks; i++)
  {
    pre_send_recv = rdtsc();
    recv_complete = send_complete = 0;
    int dest = (task_id + 1) % num_tasks;
    MPI_Isend(B_data, matrix_size * matrix_fraction, MPI_DOUBLE, dest, 200,
              MPI_COMM_WORLD, &buffer_send);
    MPI_Irecv(B_buffer, matrix_size * matrix_fraction, MPI_DOUBLE,
              MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &buffer_recv);

    matrix_multiply_part(A, B, C, matrix_size, matrix_fraction, i, num_threads,
                         &time_mult, task_id);
    
    /* Wait for send and receive to finish */
    while (1)
    {
      MPI_Test(&buffer_recv, &flag_recv, &status);
      MPI_Test(&buffer_send, &flag_send, &status);
      if (flag_recv != 0 && recv_complete == 0)
        recv_complete = rdtsc();
      if (flag_send != 0 && send_complete == 0)
        send_complete = rdtsc();
      if (flag_recv != 0 && flag_send != 0)
        break;
    }

    time_send += (send_complete - pre_send_recv);
    time_recv += (recv_complete - pre_send_recv);

    /* Switch the addresses of B and B_buffer */
    B_switch = B_data;
    B_switch_addr = B;
    B_data = B_buffer;
    B = B_buffer_addr;
    B_buffer = B_switch;
    B_buffer_addr = B_switch_addr;
  }

  /* Have MPI rank 0 printf its part of the matrix */
  if (task_id == 0 && argc > 3 && strcmp(argv[3], "--print_matrix") == 0)
  {
    for (i = 0; i < matrix_fraction; i++)
    {
      for (j = 0; j < matrix_size; j++)
      {
        printf("%f ", C[i][j]);
      }
      printf("\n");
    }
  }

  /* MIN, MAX, AVG multiplication time */
  MPI_Allreduce(&time_mult, &min_mult, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                MPI_COMM_WORLD);
  min_mult_secs = (double)min_mult / CLOCK_RATE;

  MPI_Allreduce(&time_mult, &max_mult, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                MPI_COMM_WORLD);
  max_mult_secs = (double)max_mult / CLOCK_RATE;

  MPI_Allreduce(&time_mult, &avg_mult, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  avg_mult /= num_tasks;
  avg_mult_secs = (double)avg_mult / CLOCK_RATE;

  /* MIN, MAX, AVG send time */
  MPI_Allreduce(&time_send, &min_send, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                MPI_COMM_WORLD);
  min_send_secs = (double)min_send / CLOCK_RATE;

  MPI_Allreduce(&time_send, &max_send, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                MPI_COMM_WORLD);
  max_send_secs = (double)max_send / CLOCK_RATE;
  
  MPI_Allreduce(&time_send, &avg_send, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  avg_send /= num_tasks;
  avg_send_secs = (double)avg_send / CLOCK_RATE;

  /* MIN, MAX, AVG receive time */
  MPI_Allreduce(&time_recv, &min_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                MPI_COMM_WORLD);
  min_recv_secs = (double)min_recv / CLOCK_RATE;

  MPI_Allreduce(&time_recv, &max_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                MPI_COMM_WORLD);
  max_recv_secs = (double)max_recv / CLOCK_RATE;

  MPI_Allreduce(&time_recv, &avg_recv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  avg_recv /= num_tasks;
  avg_recv_secs = (double)avg_recv / CLOCK_RATE;

  prog_end = rdtsc();
  prog_total = prog_end - prog_begin;
  double prog_secs = (double)prog_total / CLOCK_RATE;

  /* Print out the times (once) */
  if (task_id == 0)
  {
    printf("Total execution time: %f\n", prog_secs);
    printf("\t\tmin\t\tmax\t\tavg\n");
    printf("Multiplication:\t%f\t%f\t%f\n", min_mult_secs, max_mult_secs,
           avg_mult_secs);
    printf("Sending:\t%f\t%f\t%f\n", min_send_secs, max_send_secs,
           avg_send_secs);
    printf("Receciving:\t%f\t%f\t%f\n", min_recv_secs, max_recv_secs,
           avg_recv_secs);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}