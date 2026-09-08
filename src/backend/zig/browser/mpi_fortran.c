// The browser backend has no Fortran callers and needs no Fortran sentinels.
void mpi_get_fort_pointers_(void) {}

#include "mpi.h"
int MPI_Type_dup(MPI_Datatype oldtype, MPI_Datatype *newtype) {
    int result = MPI_Type_contiguous(1, oldtype, newtype);
    if (result == MPI_SUCCESS) result = MPI_Type_commit(newtype);
    return result;
}
