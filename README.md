# ParallelMatrixMultiplier
My lab for Parallel-Programming course using MPI (lab 3)

## Report
[Detailed report on the implementation of laboratory work.](https://docs.google.com/spreadsheets/d/1XB2V_nefBfjtqDP5qDOAkbld5xyMaRwThIbJSbW4HHs/edit#gid=137491946)
## Advices
Grid parameters and matrix sizes are specified in the program as global variables, and you can change them there.
To compile the program use:

```mpic++ MatrixUtils.cpp ParallelMatrixMultiplier.cpp -o your_exe```

To test the code on a machine that doesn't have enough cores for your grid you can use:

```mpirun --oversubcribe -np your_number your_exe```