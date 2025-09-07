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

// ===== FLATTENED ARRAY FUNCTIONS =====

/**
 * @brief Serial implementation of 2D convolution using flattened arrays
 *
 * @param f Input matrix (flattened)
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix (flattened)
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix (flattened)
 */
void conv2d_serial_flatten(float *restrict f, int H, int W, float *restrict g, int kH, int kW,
                           float *restrict output);

/**
 * @brief Parallel implementation of 2D convolution using flattened arrays
 *
 * @param f Input matrix (flattened)
 * @param H Number of rows in input matrix
 * @param W Number of columns in input matrix
 * @param g Kernel matrix (flattened)
 * @param kH Number of rows in kernel matrix
 * @param kW Number of columns in kernel matrix
 * @param output Output matrix (flattened)
 */
void conv2d_parallel_flatten(float *restrict f, int H, int W, float *restrict g, int kH, int kW,
                             float *restrict output);

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
int read_matrix_dimensions(const char *filename, int *rows, int *cols);
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

// ===== FLATTENED ARRAY MEMORY MANAGEMENT =====

/**
 * @brief Allocate a flattened matrix with the specified number of rows and columns
 *
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return Pointer to the allocated flattened matrix
 */
float *allocate_matrix_flatten(int rows, int cols);

/**
 * @brief Free a flattened matrix
 *
 * @param matrix Pointer to the flattened matrix to free
 */
void free_matrix_flatten(float *matrix);

/**
 * @brief Initialize a flattened matrix with the specified value
 *
 * @param matrix Pointer to the flattened matrix to initialize
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @param value Value to initialize the matrix with
 */
void initialize_matrix_flatten(float *matrix, int rows, int cols, float value);

/**
 * @brief Compare two flattened matrices element-wise within an absolute tolerance
 *
 * @param matrix1 Pointer to the first flattened matrix
 * @param matrix2 Pointer to the second flattened matrix
 * @param rows Number of rows in both matrices
 * @param cols Number of columns in both matrices
 * @param tolerance Maximum allowed absolute difference per element
 * @return int 1 if matrices are equal within tolerance, 0 otherwise
 */
int compare_matrices_flatten(float *matrix1, float *matrix2, int rows, int cols,
                             float tolerance);

// ===== FLATTENED ARRAY I/O FUNCTIONS =====

/**
 * @brief Read matrix dimensions from file (first line only) - flattened version
 *
 * @param filename Path to the input file
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @return int 0 on success, -1 on failure
 */
int read_matrix_dimensions(const char *filename, int *rows, int *cols);

/**
 * @brief Read matrix from file into flattened format
 *
 * @param filename Path to the input file
 * @param matrix Pointer to store the flattened matrix
 * @param rows Pointer to store number of rows
 * @param cols Pointer to store number of columns
 * @return int 0 on success, -1 on failure
 */
int read_matrix_from_file_flatten(const char *filename, float **matrix, int *rows, int *cols);

/**
 * @brief Read large matrix from file using memory-efficient approach
 *
 * @param filename Path to the input file
 * @param matrix Pointer to store the flattened matrix
 * @param rows Number of rows (already known)
 * @param cols Number of columns (already known)
 * @return int 0 on success, -1 on failure
 */
int read_large_matrix_from_file_flatten(const char *filename, float **matrix, int rows, int cols);

/**
 * @brief Write flattened matrix to file
 *
 * @param filename Path to the output file
 * @param matrix Flattened matrix to write
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @return int 0 on success, -1 on failure
 */
int write_matrix_to_file_flatten(const char *filename, float *matrix, int rows, int cols);

/**
 * @brief Print flattened matrix to stdout
 *
 * @param matrix Flattened matrix to print
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 */
void print_matrix_flatten(float *matrix, int rows, int cols);

/**
 * @brief Generate random matrix in flattened format
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @return Pointer to the generated flattened matrix
 */
float *generate_random_matrix_flatten(int rows, int cols, float min_val, float max_val);

/**
 * @brief Read matrix from file directly into padded flattened format
 *
 * @param filename Path to the input file
 * @param kernel_height Height of the kernel (for padding calculation)
 * @param kernel_width Width of the kernel (for padding calculation)
 * @param padded Pointer to store the padded flattened matrix
 * @param padded_height Pointer to store padded height
 * @param padded_width Pointer to store padded width
 * @param original_height Pointer to store original height
 * @param original_width Pointer to store original width
 * @return int 0 on success, -1 on failure
 */
int read_matrix_into_padded_flatten(const char *filename, int kernel_height, int kernel_width,
                                   float **padded, int *padded_height, int *padded_width,
                                   int *original_height, int *original_width);

/**
 * @brief Generate random matrix directly into padded flattened format
 *
 * @param height Original height
 * @param width Original width
 * @param kernel_height Height of the kernel (for padding calculation)
 * @param kernel_width Width of the kernel (for padding calculation)
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @param padded Pointer to store the padded flattened matrix
 * @param padded_height Pointer to store padded height
 * @param padded_width Pointer to store padded width
 * @return Pointer to the original data area in the padded matrix
 */
float *generate_random_matrix_into_padded_flatten(int height, int width, int kernel_height, int kernel_width,
                                                 float min_val, float max_val, float **padded,
                                                 int *padded_height, int *padded_width);

// ===== STREAMING CONVOLUTION FUNCTIONS =====

/**
 * @brief Streaming convolution structure for managing chunks
 */
typedef struct {
    float *chunk_data;           // Current chunk data
    float *padded_chunk;         // Padded chunk data
    int chunk_height;            // Height of current chunk
    int chunk_width;             // Width of current chunk
    int padded_chunk_height;     // Height of padded chunk
    int padded_chunk_width;      // Width of padded chunk
    int start_row;               // Starting row in original matrix
    int end_row;                 // Ending row in original matrix
    int overlap_top;             // Overlap with previous chunk (top)
    int overlap_bottom;          // Overlap with next chunk (bottom)
    int is_first_chunk;          // Flag for first chunk
    int is_last_chunk;           // Flag for last chunk
} streaming_chunk_t;

/**
 * @brief Initialize streaming convolution context
 *
 * @param filename Input feature map file
 * @param kernel_height Height of kernel
 * @param kernel_width Width of kernel
 * @param chunk_height Height of each chunk (cache-friendly)
 * @param total_height Total height of input matrix
 * @param total_width Total width of input matrix
 * @return streaming_chunk_t* Initialized streaming context
 */
streaming_chunk_t* init_streaming_conv(const char *filename, int kernel_height, int kernel_width, 
                                      int chunk_height, int total_height, int total_width);

/**
 * @brief Read next chunk from file with proper padding
 *
 * @param ctx Streaming context
 * @param filename Input file
 * @return int 1 if chunk read successfully, 0 if no more chunks, -1 on error
 */
int read_next_chunk(streaming_chunk_t *ctx, const char *filename);

/**
 * @brief Process current chunk with parallel convolution
 *
 * @param ctx Streaming context
 * @param kernel Kernel matrix
 * @param kernel_height Height of kernel
 * @param kernel_width Width of kernel
 * @param output Output matrix (full size)
 * @param total_width Width of full output matrix
 * @param use_serial Use serial implementation if 1, parallel if 0
 */
void process_chunk(streaming_chunk_t *ctx, float *kernel, int kernel_height, int kernel_width,
                  float *output, int total_width, int use_serial);

/**
 * @brief Clean up streaming context
 *
 * @param ctx Streaming context to clean up
 */
void cleanup_streaming_conv(streaming_chunk_t *ctx);

/**
 * @brief Calculate optimal chunk height for cache efficiency
 *
 * @param total_height Total height of input matrix
 * @param kernel_height Height of kernel
 * @param available_memory Available memory in bytes
 * @return int Optimal chunk height
 */
int calculate_optimal_chunk_height(int total_height, int kernel_height, size_t available_memory);

/**
 * @brief Read matrix chunk from file with overlap for padding
 *
 * @param filename Input file
 * @param start_row Starting row to read
 * @param end_row Ending row to read (exclusive)
 * @param total_width Total width of matrix
 * @param overlap_top Rows to read before start_row for padding
 * @param overlap_bottom Rows to read after end_row for padding
 * @param chunk_data Buffer to store chunk data
 * @param actual_height Actual height read (including overlaps)
 * @return int 0 on success, -1 on failure
 */
int read_matrix_chunk(const char *filename, int start_row, int end_row, int total_width,
                     int overlap_top, int overlap_bottom, float **chunk_data, int *actual_height);

/**
 * @brief Apply padding to chunk for convolution
 *
 * @param chunk_data Input chunk data
 * @param chunk_height Height of input chunk
 * @param chunk_width Width of input chunk
 * @param kernel_height Height of kernel
 * @param kernel_width Width of kernel
 * @param padded_chunk Output padded chunk
 * @param padded_height Height of padded chunk
 * @param padded_width Width of padded chunk
 * @param is_first_chunk Whether this is the first chunk
 * @param is_last_chunk Whether this is the last chunk
 */
void apply_chunk_padding(float *chunk_data, int chunk_height, int chunk_width,
                        int kernel_height, int kernel_width, float **padded_chunk,
                        int *padded_height, int *padded_width, int is_first_chunk, int is_last_chunk);

#endif  // CONV2D_H
