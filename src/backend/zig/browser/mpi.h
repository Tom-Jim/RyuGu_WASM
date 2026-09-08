#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#include "../../../../C++/mpi-serial/mpi.h"
#ifdef __cplusplus
}
#endif
#include <stdlib.h>
#include <stdint.h>

typedef int MPI_File;
typedef int64_t MPI_Offset;
#define MPI_MODE_WRONLY 1
#define MPI_MODE_CREATE 2
#define MPI_MODE_EXCL 4
#define MPI_SEEK_SET 0
// Diagnostic MPI file exports are unsupported in a browser worker.
static inline int MPI_File_open(MPI_Comm c, const char *p, int mode, MPI_Info info, MPI_File *file) {
    (void)c; (void)p; (void)mode; (void)info; (void)file;
    return MPI_ERR_OTHER;
}
static inline int MPI_File_close(MPI_File *file) { (void)file; return MPI_ERR_OTHER; }
static inline int MPI_File_seek(MPI_File file, MPI_Offset off, int whence) {
    (void)file; (void)off; (void)whence; return MPI_ERR_OTHER;
}
static inline int MPI_File_write_at(MPI_File file, MPI_Offset off, const void *data, int count, MPI_Datatype type, MPI_Status *status) {
    (void)file; (void)off; (void)data; (void)count; (void)type; (void)status; return MPI_ERR_OTHER;
}
static inline int MPI_Ialltoallv(const void *send, const int *sc, const int *sd, MPI_Datatype st,
                               void *recv, const int *rc, const int *rd, MPI_Datatype rt,
                               MPI_Comm comm, MPI_Request *request) {
    const int result = MPI_Alltoallv(send, sc, sd, st, recv, rc, rd, rt, comm);
    *request = MPI_REQUEST_NULL;
    return result;
}

#define MPI_CART 1
#define MPI_COMM_SELF MPI_COMM_WORLD
#define MPI_IDENT 0
#define MPI_CONGRUENT 1
static inline int MPI_Comm_compare(MPI_Comm a, MPI_Comm b, int *comparison) {
    if (a == MPI_COMM_NULL || b == MPI_COMM_NULL) return MPI_ERR_OTHER;
    *comparison = a == b ? MPI_IDENT : MPI_CONGRUENT;
    return MPI_SUCCESS;
}
static inline int MPI_Comm_set_name(MPI_Comm comm, const char *name) {
    (void)comm; (void)name;
    return MPI_SUCCESS;
}
static inline int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *result) {
    (void)info;
    *(void **)result = malloc((size_t)size);
    return *(void **)result ? MPI_SUCCESS : MPI_ERR_OTHER;
}
static inline int MPI_Free_mem(void *ptr) { free(ptr); return MPI_SUCCESS; }
static inline int MPI_Topo_test(MPI_Comm comm, int *status) {
    (void)comm;
    *status = MPI_UNDEFINED;
    return MPI_SUCCESS;
}
static inline int MPI_Cart_rank(MPI_Comm comm, const int *coords, int *rank) {
    (void)comm; (void)coords;
    *rank = 0;
    return MPI_SUCCESS;
}
