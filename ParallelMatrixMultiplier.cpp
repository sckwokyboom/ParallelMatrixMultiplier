#include <iostream>
#include <mpi.h>
#include "MatrixOperations.h"

class MultiplierException : public std::runtime_error {
public:
  explicit MultiplierException(const std::string &message) : std::runtime_error(message) {}
};

static constexpr int NUM_OF_DIMS_OF_CARTESIAN_GRID = 2;
// Grid communicator
MPI_Comm GridComm;
// Column communicator
MPI_Comm ColComm;
// Row communicator
MPI_Comm RowComm;
int GridCoords[2];
int ProcNum = 0;
int ProcRank = 0;

// The matrix A is split into p1 rows.
static constexpr int p1 = 6;
// The matrix B is split into p2 columns.
static constexpr int p2 = 3;
// Height of the matrix A
static constexpr int n1 = 24;
// Width of the matrix A and height of the matrix B
static constexpr int n2 = 8;
// Width of the matrix B
static constexpr int n3 = 9;

void createGridCommunicators() {
  if (p1 * p2 > ProcNum) {
    throw MultiplierException(std::string("There are not enough processes for a launch. Expected: ")
                              + std::to_string(p1 * p2)
                              + std::string(" or more processes, got: ")
                              + std::to_string(ProcNum) + std::string(" processes."));
  }
  // Number of processes in each dimension of the grid
  int dimSize[NUM_OF_DIMS_OF_CARTESIAN_GRID];

  // logical array specifying whether the grid is periodic (true = 1) or not (false = 0) in each dimension
  int periodic[NUM_OF_DIMS_OF_CARTESIAN_GRID];

  // logical array specifying whether the grid dimension should be fixed (true = 1) or not (false = 0)
  int subDimension[NUM_OF_DIMS_OF_CARTESIAN_GRID];

  dimSize[0] = p1;
  dimSize[1] = p2;

  periodic[0] = 1;
  periodic[1] = 1;

  // Creation of the Cartesian communicator
  MPI_Cart_create(MPI_COMM_WORLD, NUM_OF_DIMS_OF_CARTESIAN_GRID, dimSize, periodic, 1, &GridComm);

  // Determination of the cartesian coordinates for every process
  MPI_Cart_coords(GridComm, ProcRank, NUM_OF_DIMS_OF_CARTESIAN_GRID, GridCoords);

  // Creating communicators for rows
  // Dimension is fixed
  subDimension[0] = 0;

  // Dimension belong to the subgrid
  subDimension[1] = 1;

  MPI_Cart_sub(GridComm, subDimension, &RowComm);

  // Creating communicators for columns
  // Dimension belong to the subgrid
  subDimension[0] = 1;

  // Dimension is fixed
  subDimension[1] = 0;

  MPI_Cart_sub(GridComm, subDimension, &ColComm);
}

void initializeProcess(double *&pAMatrix, double *&pBMatrix, double *&pCMatrix,
                       double *&pAblock, double *&pBblock, double *&pCblock, int &ABlockSize, int &BBlockSize) {
  ABlockSize = n1 / p1;
  BBlockSize = n3 / p2;

  pAblock = new double[n2 * ABlockSize];
  pBblock = new double[n2 * BBlockSize];
  pCblock = new double[ABlockSize * BBlockSize];

  if (ProcRank == 0) {
    pAMatrix = new double[n1 * n2];
    pBMatrix = new double[n2 * n3];
    pCMatrix = new double[n1 * n3];
    dataInitialization(pAMatrix, n1, n2);
    dataInitialization(pBMatrix, n2, n3);
    setToZero(pCMatrix, n1, n3);
  }

  setToZero(pCblock, ABlockSize, BBlockSize);
}

void terminateProcess(const double *AMatrix, const double *BMatrix,
                      const double *CMatrix, const double *Ablock, const double *Bblock, const double *Cblock) {

  if (ProcRank == 0) {
    delete[] AMatrix;
    delete[] BMatrix;
    delete[] CMatrix;
  }
  delete[] Ablock;
  delete[] Bblock;
  delete[] Cblock;
}

void dataDistribution(double *AMatrix, double *BMatrix, double *Ablock,
                      double *Bblock, int ABlockSize, int BBlockSize) {

  if (GridCoords[1] == 0) {
    MPI_Scatter(AMatrix, ABlockSize * n2, MPI_DOUBLE, Ablock,
                ABlockSize * n2, MPI_DOUBLE, 0, ColComm);
  }

  MPI_Bcast(Ablock, ABlockSize * n2, MPI_DOUBLE, 0, RowComm);
  MPI_Datatype col, coltype;

  MPI_Type_vector(n2, BBlockSize, n3, MPI_DOUBLE, &col);
  MPI_Type_commit(&col);
  MPI_Type_create_resized(col, 0, BBlockSize * sizeof(double), &coltype);
  MPI_Type_commit(&coltype);

  if (GridCoords[0] == 0) {
    MPI_Scatter(BMatrix, 1, coltype, Bblock, n2 * BBlockSize,
                MPI_DOUBLE, 0, RowComm);
  }

  MPI_Bcast(Bblock, BBlockSize * n2, MPI_DOUBLE, 0, ColComm);
}

int main(int argc, char *argv[]) {
  // First argument of matrix multiplication
  double *AMatrix = nullptr;
  // Second argument of matrix multiplication
  double *BMatrix = nullptr;
  // Result matrix
  double *CMatrix = nullptr;

  int ABlockSize = 0;
  int BBlockSize = 0;

  // Current block of the matrix A
  double *Ablock = nullptr;
  // Current block of the matrix B
  double *Bblock = nullptr;
  // Block of the result matrix C
  double *Cblock = nullptr;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  try {
    if ((n1 % p1 != 0) || (n3 % p2 != 0)) {
      if (ProcRank == 0) {
        throw MultiplierException("Invalid grid size.");
      }
    } else {
      if (ProcRank == 0) {
        printf("Parallel matrix multiplication program, on %d processes\n", ProcNum);
      }
      // Grid communicator creating
      createGridCommunicators();
    }

    // Memory allocation and initialization of matrix elements
    initializeProcess(AMatrix, BMatrix, CMatrix, Ablock, Bblock,
                      Cblock, ABlockSize, BBlockSize);

    double startTime;

    if (ProcRank == 0) {
      startTime = MPI_Wtime();
      printf("Initial matrix A \n");
      printMatrix(AMatrix, n1, n2);
      printf("Initial matrix B \n");
      printMatrix(BMatrix, n2, n3);
    }
    // Distribute data among the processes
    dataDistribution(AMatrix, BMatrix, Ablock, Bblock, ABlockSize, BBlockSize);
    // Multiply matrix blocks of the current process
    matrixMul(Ablock, Bblock, Cblock, ABlockSize, n2, BBlockSize);

    // Gather all data in one matrix

    // the first step is creating a block type and resizing it
    MPI_Datatype block, blocktype;
    MPI_Type_vector(ABlockSize, BBlockSize, n3, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);

    MPI_Type_create_resized(block, 0, BBlockSize * sizeof(double), &blocktype);
    MPI_Type_commit(&blocktype);

    // calculate displ
    int *displ = new int[p1 * p2];
    int *rcount = new int[p1 * p2];
    int blockCount = 0;
    int blockSize = ABlockSize * BBlockSize;
    int numCount = 0;
    int written;
    int j = 0;

    while (numCount < p1 * p2 * blockSize) {
      written = 0;
      for (int i = 0; i < n3; i += BBlockSize) {
        displ[j] = blockCount;
        rcount[j] = 1;
        j++;
        blockCount++;

        written++;
      }
      numCount += written * blockSize;
      blockCount += written * (ABlockSize - 1);
    }

    MPI_Gatherv(Cblock, blockSize, MPI_DOUBLE, CMatrix,
                rcount, displ, blocktype, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) {
      double endTime = MPI_Wtime();
      printf("Matrix C \n");

      printMatrix(CMatrix, n1, n3);
      printf("That took %lf seconds\n", endTime - startTime);
    }

    terminateProcess(AMatrix, BMatrix, CMatrix, Ablock, Bblock, Cblock);
    delete[] displ;
    delete[] rcount;
    MPI_Finalize();
  } catch (std::runtime_error &error) {
    std::cerr << error.what() << std::endl;
    MPI_Finalize();
  }
  return 0;
}
