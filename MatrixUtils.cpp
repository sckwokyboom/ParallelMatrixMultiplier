#include "MatrixUtils.h"
#include <mpi.h>
#include <cstdlib>

double randDouble() {
  return (double) rand() / RAND_MAX * 50.0 - 2.0;
}

void dataInit(double *pMatrix, int rowCount, int colCount) {
  for (int i = 0; i < rowCount; i++) {
    for (int j = 0; j < colCount; j++) {
      pMatrix[i * colCount + j] = randDouble();
    }
  }
}

void setToZero(double *pMatrix, int rowCount, int colCount) {
  for (int i = 0; i < rowCount; i++) {
    for (int j = 0; j < colCount; j++) {
      pMatrix[i * colCount + j] = 0;
    }
  }
}

// Function for formatted vector output
void printVector(const double *pVector, int size, int procNum) {
  printf("proc #%d ", procNum);
//    MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; i++)
    printf("%7.4f ", pVector[i]);

//    MPI_Barrier(MPI_COMM_WORLD);
  printf("\n");
}

// Function for formatted vector output
void printVector(const int *pVector, int size, int procNum) {
  printf("proc # %d ", procNum);
//    MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; i++)
    printf("%d ", pVector[i]);

//    MPI_Barrier(MPI_COMM_WORLD);
  printf("\n");
}

// Function for formatted matrix output
void printMatrix(const double *pMatrix, int rowCount, int colCount) {
  int i, j; // Loop variables
  for (i = 0; i < rowCount; i++) {
    for (j = 0; j < colCount; j++)
      printf("%7.4f ", pMatrix[i * colCount + j]);
    printf("\n");
  }
}

// Function for matrix multiplication
void matrixMul(const double *pAMatrix, const double *pBMatrix, double *pCMatrix, int heightA, int widthAheightB,
               int widthB) {
  for (int i = 0; i < heightA; ++i) {
    double *c = pCMatrix + i * widthB;
    for (int j = 0; j < widthB; ++j)
      c[j] = 0;
    for (int k = 0; k < widthAheightB; ++k) {
      const double *b = pBMatrix + k * widthB;
      double a = pAMatrix[i * heightA + k];
      for (int j = 0; j < widthB; ++j)
        c[j] += a * b[j];
    }
  }
}
