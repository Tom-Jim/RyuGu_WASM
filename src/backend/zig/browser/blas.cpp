#include <Eigen/Dense>
#include <Eigen/SVD>
#include <complex>
#include <cstdlib>

// ExaFMM's Fortran ABI uses column-major matrices. Eigen supplies the
// numerical operations, including the SVD used to build equivalent surfaces.
template<class T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template<class T>
Matrix<T> matrix(const T* a, int rows, int cols, int stride, char transpose) {
    Eigen::Map<const Matrix<T>, 0, Eigen::OuterStride<>> m(a, rows, cols, Eigen::OuterStride<>(stride));
    if (transpose == 'T') return m.transpose();
    if (transpose == 'C') return m.adjoint();
    return m;
}
template<class T>
void gemm(char ta, char tb, int m, int n, int k, T alpha, T* a, int lda,
          T* b, int ldb, T beta, T* c, int ldc) {
    auto left = matrix(a, ta == 'N' ? m : k, ta == 'N' ? k : m, lda, ta);
    auto right = matrix(b, tb == 'N' ? k : n, tb == 'N' ? n : k, ldb, tb);
    Eigen::Map<Matrix<T>, 0, Eigen::OuterStride<>> out(c, m, n, Eigen::OuterStride<>(ldc));
    if (beta == T(0)) out.noalias() = alpha * (left * right);
    else out = alpha * (left * right) + beta * out;
}
template<class T>
void gemv(char trans, int m, int n, T alpha, T* a, int lda, T* x, int incx,
          T beta, T* y, int incy) {
    auto mat = matrix(a, m, n, lda, trans);
    Matrix<T> input(mat.cols(), 1);
    for (int i = 0; i < mat.cols(); ++i) input(i, 0) = x[i * incx];
    Matrix<T> result = alpha * mat * input;
    for (int i = 0; i < mat.rows(); ++i)
        y[i * incy] = result(i, 0) + (beta == T(0) ? T(0) : beta * y[i * incy]);
}
extern "C" {
void dgemm_(char* ta, char* tb, int* m, int* n, int* k, double* alpha, double* a,
            int* lda, double* b, int* ldb, double* beta, double* c, int* ldc) {
    gemm(*ta, *tb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}
void dgemv_(char* t, int* m, int* n, double* alpha, double* a, int* lda,
            double* x, int* incx, double* beta, double* y, int* incy) {
    gemv(*t, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}
void dgesvd_(char* ju, char* jv, int* m, int* n, double* a, int* lda, double* s,
             double* u, int* ldu, double* vt, int* ldvt, double*, int*, int* info) {
    if (*ju != 'S' || *jv != 'S') { *info = -1; return; }
    auto input = matrix(a, *m, *n, *lda, 'N');
    Eigen::JacobiSVD<Matrix<double>> svd(input, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const int k = std::min(*m, *n);
    Eigen::Map<Eigen::VectorXd>(s, k) = svd.singularValues();
    Eigen::Map<Matrix<double>, 0, Eigen::OuterStride<>>(u, *m, k, Eigen::OuterStride<>(*ldu)) = svd.matrixU();
    Eigen::Map<Matrix<double>, 0, Eigen::OuterStride<>>(vt, k, *n, Eigen::OuterStride<>(*ldvt)) = svd.matrixV().transpose();
    *info = svd.info() == Eigen::Success ? 0 : 1;
}
}
