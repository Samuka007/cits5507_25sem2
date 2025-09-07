#ifndef CONV2D_H
#define CONV2D_H

#include <omp.h>
#include <unistd.h>

/**
 * @brief Serial implementation of 2D convolution
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix
 */
void conv2d_serial(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                   float **restrict output);

/**
 * @brief Parallel implementation of 2D convolution using OpenMP
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix
 */
void conv2d_parallel(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                     float **restrict output);

/**
 * @brief Cache-optimized serial implementation of 2D convolution
 * 
 * This implementation uses several cache optimization techniques:
 * - Loop tiling/blocking for better cache utilization
 * - Kernel reordering for improved spatial locality
 * - Vectorization hints for compiler optimization
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix
 */
void conv2d_serial_cache_optimized(float **f, int H, int W, float **g, int kH, int kW,
                                   float **output);

/**
 * @brief Cache-optimized parallel implementation of 2D convolution
 * 
 * This implementation combines OpenMP parallelization with cache optimization:
 * - Loop tiling/blocking for better cache utilization
 * - Kernel reordering for improved spatial locality
 * - Vectorization hints for compiler optimization
 * - Optimized OpenMP scheduling for cache-friendly parallelization
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix
 */
void conv2d_parallel_cache_optimized(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                                     float **restrict output);

/**
 * @brief Highly optimized parallel convolution with kernel-specific optimizations
 * 
 * This implementation uses multiple acceleration techniques:
 * - Kernel unrolling for small kernels (3x3, 5x5)
 * - SIMD vectorization with proper alignment
 * - Memory prefetching hints
 * - Optimized loop structures
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix
 */
void conv2d_parallel_optimized(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                               float **restrict output);

/**
 * @brief Highly optimized 3x3 kernel convolution
 * 
 * Uses loop unrolling and SIMD optimizations specifically for 3x3 kernels
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix (3x3)
 * @param output Output matrix
 */
void conv2d_3x3_optimized(float **restrict f, int H, int W, float **restrict g, float **restrict output);

/**
 * @brief Highly optimized 5x5 kernel convolution
 * 
 * Uses loop unrolling and SIMD optimizations specifically for 5x5 kernels
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix (5x5)
 * @param output Output matrix
 */
void conv2d_5x5_optimized(float **restrict f, int H, int W, float **restrict g, float **restrict output);

/**
 * @brief SIMD-optimized convolution with vectorization
 * 
 * Uses advanced SIMD techniques and memory prefetching
 *
 * @param f Input matrix
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix
 */
void conv2d_parallel_simd_optimized(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                                    float **restrict output);

/**
 * @brief Allocate a matrix with the specified number of rows and columns
 *
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return Pointer to the allocated matrix
 */
float **allocate_matrix(int rows, int cols);

/**
 * @brief Free a matrix with the specified number of rows
 *
 * @param matrix Pointer to the matrix to free
 * @param rows Number of rows in the matrix
 */
void free_matrix(float **matrix, int rows);

/**
 * @brief Initialize a matrix with the specified value
 *
 * @param matrix Pointer to the matrix to initialize
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @param value Value to initialize the matrix with
 */
void initialize_matrix(float **matrix, int rows, int cols, float value);

// Matrix I/O functions
int read_matrix_from_file(const char *filename, float ***matrix, int *rows,
                          int *cols);
int write_matrix_to_file(const char *filename, float **matrix, int rows,
                         int cols);
void print_matrix(float **matrix, int rows, int cols);

// Matrix generation functions
float **generate_random_matrix(int rows, int cols, float min_val,
                               float max_val);
void generate_padded_matrix(float **input, int height, int width,
                            int kernel_height, int kernel_width,
                            float ***padded, int *padded_height,
                            int *padded_width);

// New functions to read/generate directly into padded matrix
int read_matrix_into_padded(const char *filename, int kernel_height, int kernel_width,
                           float ***padded, int *padded_height, int *padded_width,
                           int *original_height, int *original_width);
float **generate_random_matrix_into_padded(int height, int width, int kernel_height, int kernel_width,
                                          float min_val, float max_val, float ***padded,
                                          int *padded_height, int *padded_width);
int generate_feature_map(const char* filename, int height, int width);
int write_matrix_header(const char *filename, int rows, int cols);
int write_matrix_data_batch(const char *filename, int start_row, int end_row, int cols, 
                           float min_val, float max_val);

/**
 * @brief Compare two matrices element-wise within an absolute tolerance.
 *
 * Compares \`matrix1\` and \`matrix2\` of size \`rows x cols\` and returns
 * whether all corresponding elements differ by no more than \`tolerance\` in
 * absolute value.
 *
 * @param matrix1 Pointer to the first matrix (size rows x cols)
 * @param matrix2 Pointer to the second matrix (size rows x cols)
 * @param rows Number of rows in both matrices
 * @param cols Number of columns in both matrices
 * @param tolerance Maximum allowed absolute difference per element
 * @return int 1 if matrices are equal within tolerance, 0 otherwise
 */
int compare_matrices(float **matrix1, float **matrix2, int rows, int cols,
                     float tolerance);

#endif  // CONV2D_H
