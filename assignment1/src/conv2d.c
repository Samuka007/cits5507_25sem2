#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/conv2d.h"

#ifndef MATRIX_ALIGNMENT
#define MATRIX_ALIGNMENT 32
#endif

void conv2d_serial(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                   float **restrict output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform convolution producing an output of size out_H x out_W
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            float sum = 0.0f;

            // Apply kernel
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    sum += f[i + ki][j + kj] * g[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }
}

void conv2d_parallel(float **restrict f, int H, int W, float **restrict g, int kH, int kW,
                     float **restrict output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform parallel convolution
#pragma omp parallel for simd collapse(2) schedule(static)
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            float sum = 0.0f;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    sum += f[i + ki][j + kj] * g[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }
}

void conv2d_serial_flatten(float *restrict f, int H, int W, float *restrict g, int kH, int kW, float *restrict output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform serial convolution on flattened arrays
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            float sum = 0.0f;

            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int f_idx = (i + ki) * W + (j + kj);
                    int g_idx = ki * kW + kj;
                    sum += f[f_idx] * g[g_idx];
                }
            }

            output[i * out_W + j] = sum;
        }
    }
}

void conv2d_parallel_flatten(float *restrict f, int H, int W, float *restrict g, int kH, int kW, float *restrict output) {
    // Compute valid output dimensions from padded input and kernel sizes
    const int out_H = H - kH + 1;
    const int out_W = W - kW + 1;

    // Perform parallel convolution on flattened arrays
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < out_H; i++) {
        for (int j = 0; j < out_W; j++) {
            float sum = 0.0f;

            #pragma omp simd collapse(2) reduction(+:sum)
            for (int ki = 0; ki < kH; ki++) {
                for (int kj = 0; kj < kW; kj++) {
                    int f_idx = (i + ki) * W + (j + kj);
                    int g_idx = ki * kW + kj;
                    sum += f[f_idx] * g[g_idx];
                }
            }

            output[i * out_W + j] = sum;
        }
    }
}

/**
 * @brief Allocate a 2D matrix
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return float** Pointer to the allocated matrix
 */
float **allocate_matrix(int rows, int cols) {
    float **matrix = NULL;
    matrix = (float **)aligned_alloc(MATRIX_ALIGNMENT, rows * sizeof(float *));
    if (matrix == NULL) {
        perror("Error: Failed to allocate memory for matrix rows\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] =
            (float *)aligned_alloc(MATRIX_ALIGNMENT, cols * sizeof(float));
        if (matrix[i] == NULL) {
            perror("Error: Failed to allocate memory for matrix columns\n");
            // Free previously allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }

    return matrix;
}

// Free a 2D matrix
void free_matrix(float **matrix, int rows) {
    if (matrix == NULL) return;

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Initialize matrix with a specific value
void initialize_matrix(float **matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows; i++) {
        memset(matrix[i], value, (size_t)cols * sizeof(float));
    }
}

// Compare two matrices with tolerance
int compare_matrices(float **matrix1, float **matrix2, int rows, int cols,
                     float tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(matrix1[i][j] - matrix2[i][j]) > tolerance) {
                return 0;  // Matrices are different
            }
        }
    }
    return 1;  // Matrices are the same
}

void generate_padded_matrix(float **input, int height, int width,
                            int kernel_height, int kernel_width,
                            float ***padded, int *padded_height,
                            int *padded_width) {
    // Asymmetric "same" padding so that output has the same size as input
    // Works for both odd and even kernel sizes
    int pad_top = (kernel_height - 1) / 2;
    int pad_left = (kernel_width - 1) / 2;
    int pad_bottom = kernel_height - 1 - pad_top;
    int pad_right = kernel_width - 1 - pad_left;

    *padded_height = height + pad_top + pad_bottom;
    *padded_width = width + pad_left + pad_right;
    *padded = allocate_matrix(*padded_height, *padded_width);
    initialize_matrix(*padded, *padded_height, *padded_width, 0.0f);
    for (int i = 0; i < height; i++) {
        // Copy each row of the input matrix into the center of the padded
        // matrix
        memcpy((*padded)[i + pad_top] + pad_left, input[i],
               width * sizeof(float));
    }
}

// ===== FLATTENED ARRAY FUNCTIONS =====

// Allocate a flattened matrix
float *allocate_matrix_flatten(int rows, int cols) {
    float *matrix = NULL;
    matrix = (float *)aligned_alloc(MATRIX_ALIGNMENT, rows * cols * sizeof(float));
    if (matrix == NULL) {
        perror("Error: Failed to allocate memory for flattened matrix\n");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

// Free a flattened matrix
void free_matrix_flatten(float *matrix) {
    if (matrix != NULL) {
        free(matrix);
    }
}

// Initialize flattened matrix with a specific value
void initialize_matrix_flatten(float *matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = value;
    }
}

// Compare two flattened matrices with tolerance
int compare_matrices_flatten(float *matrix1, float *matrix2, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(matrix1[i] - matrix2[i]) > tolerance) {
            return 0;  // Matrices are different
        }
    }
    return 1;  // Matrices are the same
}

// ===== STREAMING CONVOLUTION FUNCTIONS =====

// Calculate optimal chunk height for cache efficiency
int calculate_optimal_chunk_height(int total_height, int kernel_height, size_t available_memory) {

    int estimated_chunk_height = 1000; // Start with reasonable default
    
    // Calculate based on available memory
    // Each chunk needs: chunk_height * width * sizeof(float) * 2 (input + padded)
    // Plus kernel and output memory
    size_t bytes_per_row = 1024 * sizeof(float) * 2; // Assume 1024 width, 2 for input+padded
    size_t max_chunk_height = available_memory * 1024 * 1024 * 1024 / bytes_per_row;
    
    if (max_chunk_height > 0 && (int)max_chunk_height < total_height) {
        estimated_chunk_height = (int)max_chunk_height;
    }
    
    // Ensure chunk height is reasonable (not too small, not too large)
    if (estimated_chunk_height < 100) {
        estimated_chunk_height = 100;
    }
    if (estimated_chunk_height > total_height) {
        estimated_chunk_height = total_height;
    }
    
    // Make it cache-friendly (multiple of 64 for better cache line utilization)
    estimated_chunk_height = (estimated_chunk_height / 64) * 64;
    if (estimated_chunk_height < 64) {
        estimated_chunk_height = 64;
    }

    if (estimated_chunk_height < kernel_height) {
        perror("todo!");
        exit(EXIT_FAILURE);
    }
    
    return estimated_chunk_height;
}

// Initialize streaming convolution context
streaming_chunk_t* init_streaming_conv(const char *filename, int kernel_height, int kernel_width, 
                                      int chunk_height, int total_height, int total_width) {
    (void)filename; // Suppress unused parameter warning
    (void)kernel_width; // Suppress unused parameter warning
    (void)total_height; // Suppress unused parameter warning
    
    streaming_chunk_t *ctx = (streaming_chunk_t*)malloc(sizeof(streaming_chunk_t));
    if (!ctx) {
        perror("Error: Failed to allocate streaming context");
        return NULL;
    }
    
    // Initialize context
    ctx->chunk_data = NULL;
    ctx->padded_chunk = NULL;
    ctx->output_chunk = NULL;
    ctx->chunk_height = chunk_height;
    ctx->chunk_width = total_width;
    ctx->padded_chunk_height = 0;
    ctx->padded_chunk_width = 0;
    ctx->output_chunk_height = 0;
    ctx->output_chunk_width = 0;
    ctx->start_row = 0;
    ctx->end_row = 0;
    ctx->total_height = total_height;
    ctx->total_width = total_width;
    ctx->overlap_top = (kernel_height - 1) / 2;
    ctx->overlap_bottom = kernel_height - 1 - ctx->overlap_top;
    ctx->is_first_chunk = 1;
    ctx->is_last_chunk = 0;
    ctx->output_file = NULL;
    ctx->output_written = 0;
    
    return ctx;
}

// Read matrix chunk from file with overlap for padding
int read_matrix_chunk(const char *filename, int start_row, int end_row, int total_width,
                     int overlap_top, int overlap_bottom, float **chunk_data, int *actual_height) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error: Cannot open file for chunk reading");
        return -1;
    }
    
    // Read dimensions (skip first line)
    int file_height, file_width;
    if (fscanf(file, "%d %d", &file_height, &file_width) != 2) {
        perror("Error: Cannot read matrix dimensions");
        fclose(file);
        return -1;
    }
    
    // Calculate actual read range with overlaps
    int read_start = start_row - overlap_top;
    int read_end = end_row + overlap_bottom;
    
    // Clamp to file bounds
    if (read_start < 0) read_start = 0;
    if (read_end > file_height) read_end = file_height;
    
    *actual_height = read_end - read_start;
    
    // Allocate chunk data
    *chunk_data = allocate_matrix_flatten(*actual_height, total_width);
    
    // Skip to start row
    for (int i = 0; i < read_start; i++) {
        for (int j = 0; j < total_width; j++) {
            float dummy;
            if (fscanf(file, "%f", &dummy) != 1) {
                perror("Error: Cannot skip rows");
                free_matrix_flatten(*chunk_data);
                fclose(file);
                return -1;
            }
        }
    }
    
    // Read chunk data
    for (int i = 0; i < *actual_height; i++) {
        for (int j = 0; j < total_width; j++) {
            if (fscanf(file, "%f", &(*chunk_data)[i * total_width + j]) != 1) {
                perror("Error: Cannot read chunk data");
                free_matrix_flatten(*chunk_data);
                fclose(file);
                return -1;
            }
        }
    }
    
    fclose(file);
    return 0;
}

// Apply padding to chunk for convolution
void apply_chunk_padding(float *chunk_data, int chunk_height, int chunk_width,
                        int kernel_height, int kernel_width, float **padded_chunk,
                        int *padded_height, int *padded_width, int is_first_chunk, int is_last_chunk) {
    
    // Calculate padding amounts
    int pad_top = (kernel_height - 1) / 2;
    int pad_left = (kernel_width - 1) / 2;
    int pad_bottom = kernel_height - 1 - pad_top;
    int pad_right = kernel_width - 1 - pad_left;
    
    // Adjust padding for first/last chunks
    if (is_first_chunk) {
        pad_top = 0; // No top padding for first chunk
    }
    if (is_last_chunk) {
        pad_bottom = 0; // No bottom padding for last chunk
    }
    
    *padded_height = chunk_height + pad_top + pad_bottom;
    *padded_width = chunk_width + pad_left + pad_right;
    
    // Allocate padded chunk
    *padded_chunk = allocate_matrix_flatten(*padded_height, *padded_width);
    initialize_matrix_flatten(*padded_chunk, *padded_height, *padded_width, 0.0f);
    
    // Copy data to center of padded chunk
    for (int i = 0; i < chunk_height; i++) {
        for (int j = 0; j < chunk_width; j++) {
            int src_idx = i * chunk_width + j;
            int dst_idx = (i + pad_top) * (*padded_width) + (j + pad_left);
            (*padded_chunk)[dst_idx] = chunk_data[src_idx];
        }
    }
}

// Read next chunk from file with proper padding
int read_next_chunk(streaming_chunk_t *ctx, const char *filename) {
    // Check if we've processed all chunks
    if (ctx->start_row >= ctx->total_height) {
        return 0; // No more chunks
    }
    
    // Calculate chunk boundaries
    ctx->end_row = ctx->start_row + ctx->chunk_height;
    if (ctx->end_row > ctx->total_height) {
        ctx->end_row = ctx->total_height;
        ctx->is_last_chunk = 1;
    }
    
    // Free previous chunk data
    if (ctx->chunk_data) {
        free_matrix_flatten(ctx->chunk_data);
    }
    if (ctx->padded_chunk) {
        free_matrix_flatten(ctx->padded_chunk);
    }
    
    // Read chunk from file
    int actual_height;
    if (read_matrix_chunk(filename, ctx->start_row, ctx->end_row, ctx->total_width,
                         ctx->overlap_top, ctx->overlap_bottom, &ctx->chunk_data, &actual_height) == -1) {
        return -1;
    }
    
    // Update chunk height to actual read height
    ctx->chunk_height = actual_height;
    
    // Apply padding
    apply_chunk_padding(ctx->chunk_data, ctx->chunk_height, ctx->total_width,
                       ctx->overlap_top + ctx->overlap_bottom + 1, ctx->total_width, // kernel dimensions
                       &ctx->padded_chunk, &ctx->padded_chunk_height, &ctx->padded_chunk_width,
                       ctx->is_first_chunk, ctx->is_last_chunk);
    
    // Update for next iteration
    ctx->start_row = ctx->end_row;
    ctx->is_first_chunk = 0;
    
    return 1; // Chunk read successfully
}

// Process current chunk with parallel convolution
void process_chunk(streaming_chunk_t *ctx, float *kernel, int kernel_height, int kernel_width,
                  float *output, int total_width, int use_serial) {
    
    // Calculate output dimensions for this chunk
    int out_H = ctx->padded_chunk_height - kernel_height + 1;
    int out_W = ctx->padded_chunk_width - kernel_width + 1;
    
    // Calculate where to write output in the full output matrix
    int output_start_row = ctx->start_row - ctx->overlap_top;
    if (output_start_row < 0) output_start_row = 0;
    
    // Process chunk with convolution
    if (use_serial) {
        // Serial convolution
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                float sum = 0.0f;
                
                for (int ki = 0; ki < kernel_height; ki++) {
                    for (int kj = 0; kj < kernel_width; kj++) {
                        int f_idx = (i + ki) * ctx->padded_chunk_width + (j + kj);
                        int g_idx = ki * kernel_width + kj;
                        sum += ctx->padded_chunk[f_idx] * kernel[g_idx];
                    }
                }
                
                // Write to full output matrix
                int output_row = output_start_row + i;
                int output_col = j;
                if (output_row >= 0 && output_row < ctx->total_height && output_col >= 0 && output_col < ctx->total_width) {
                    output[output_row * total_width + output_col] = sum;
                }
            }
        }
    } else {
        // Parallel convolution
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                float sum = 0.0f;
                
                #pragma omp simd collapse(2) reduction(+:sum)
                for (int ki = 0; ki < kernel_height; ki++) {
                    for (int kj = 0; kj < kernel_width; kj++) {
                        int f_idx = (i + ki) * ctx->padded_chunk_width + (j + kj);
                        int g_idx = ki * kernel_width + kj;
                        sum += ctx->padded_chunk[f_idx] * kernel[g_idx];
                    }
                }
                
                // Write to full output matrix
                int output_row = output_start_row + i;
                int output_col = j;
                if (output_row >= 0 && output_row < ctx->total_height && output_col >= 0 && output_col < ctx->total_width) {
                    output[output_row * total_width + output_col] = sum;
                }
            }
        }
    }
}

// Process current chunk with streaming output
void process_chunk_streaming(streaming_chunk_t *ctx, float *kernel, int kernel_height, int kernel_width,
                            int use_serial) {
    
    // Calculate output dimensions for this chunk
    int out_H = ctx->padded_chunk_height - kernel_height + 1;
    int out_W = ctx->padded_chunk_width - kernel_width + 1;
    
    // Allocate output chunk if not already allocated
    if (!ctx->output_chunk) {
        ctx->output_chunk_height = out_H;
        ctx->output_chunk_width = out_W;
        ctx->output_chunk = allocate_matrix_flatten(ctx->output_chunk_height, ctx->output_chunk_width);
    }
    
    // Process chunk with convolution
    if (use_serial) {
        // Serial convolution
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                float sum = 0.0f;
                
                for (int ki = 0; ki < kernel_height; ki++) {
                    for (int kj = 0; kj < kernel_width; kj++) {
                        int f_idx = (i + ki) * ctx->padded_chunk_width + (j + kj);
                        int g_idx = ki * kernel_width + kj;
                        sum += ctx->padded_chunk[f_idx] * kernel[g_idx];
                    }
                }
                
                ctx->output_chunk[i * out_W + j] = sum;
            }
        }
    } else {
        // Parallel convolution
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                float sum = 0.0f;
                
                #pragma omp simd collapse(2) reduction(+:sum)
                for (int ki = 0; ki < kernel_height; ki++) {
                    for (int kj = 0; kj < kernel_width; kj++) {
                        int f_idx = (i + ki) * ctx->padded_chunk_width + (j + kj);
                        int g_idx = ki * kernel_width + kj;
                        sum += ctx->padded_chunk[f_idx] * kernel[g_idx];
                    }
                }
                
                ctx->output_chunk[i * out_W + j] = sum;
            }
        }
    }
}

// Write output chunk to file
int write_output_chunk(streaming_chunk_t *ctx) {
    if (!ctx->output_file || !ctx->output_chunk) {
        return -1;
    }
    
    // Write output chunk data
    for (int i = 0; i < ctx->output_chunk_height; i++) {
        for (int j = 0; j < ctx->output_chunk_width; j++) {
            if (fprintf(ctx->output_file, "%.3f", ctx->output_chunk[i * ctx->output_chunk_width + j]) < 0) {
                return -1;
            }
            if (j < ctx->output_chunk_width - 1) {
                if (fprintf(ctx->output_file, " ") < 0) {
                    return -1;
                }
            }
        }
        if (fprintf(ctx->output_file, "\n") < 0) {
            return -1;
        }
    }
    
    ctx->output_written += ctx->output_chunk_height;
    return 0;
}

// Clean up streaming context
void cleanup_streaming_conv(streaming_chunk_t *ctx) {
    if (ctx) {
        if (ctx->chunk_data) {
            free_matrix_flatten(ctx->chunk_data);
        }
        if (ctx->padded_chunk) {
            free_matrix_flatten(ctx->padded_chunk);
        }
        if (ctx->output_chunk) {
            free_matrix_flatten(ctx->output_chunk);
        }
        if (ctx->output_file) {
            fclose(ctx->output_file);
        }
        free(ctx);
    }
}
