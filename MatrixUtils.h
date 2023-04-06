#ifndef PARALLELMATRIXMUL_MATRIXUTILS_H
#define PARALLELMATRIXMUL_MATRIXUTILS_H

void dataInit(double *pMatrix, int rowCount, int colCount);

void printMatrix(const double *pMatrix, int RowCount, int ColCount);

void printVector(const double *pVector, int Size, int ProcNum);

void setToZero(double *pMatrix, int rowCount, int colCount);

void matrixMul(const double *pAMatrix, const double *pBMatrix, double *pCMatrix,
               int heightA, int widthAheightB, int widthB);

void printVector(const int *pVector, int Size, int ProcNum);

#endif
