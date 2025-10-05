#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

// Global flags to control logging levels, defined in log.c
extern bool g_log_verbose;
extern bool g_log_debug;

// ANSI Color Codes
#define C_NRM  "\x1B[0m"
#define C_RED  "\x1B[31m"
#define C_GRN  "\x1B[32m"
#define C_YEL  "\x1B[33m"
#define C_BLU  "\x1B[34m"
#define C_MAG  "\x1B[35m"
#define C_CYN  "\x1B[36m"
#define C_WHT  "\x1B[37m"

/**
 * @brief Core per-rank logging macro. Prefixes messages with the MPI rank.
 * Do not use directly; use the level-specific macros below.
 */
#define LOG_PER_RANK(color, level_str, format, ...) do { \
    int rank_ = -1; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
    fprintf(stderr, "%s[%s][RANK %d] " C_NRM format "\n", color, level_str, rank_, ##__VA_ARGS__); \
    fflush(stderr); \
} while (0)

/**
 * @brief Core root-only logging macro.
 * Do not use directly; use the level-specific macros below.
 */
#define LOG_ROOT_ONLY(color, level_str, format, ...) do { \
    int rank_ = -1; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
    if (rank_ == 0) { \
        fprintf(stderr, "%s[%s] " C_NRM format "\n", color, level_str, ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while (0)


// ===================================================================
//                          PUBLIC API
// ===================================================================

// --- Per-Rank Logging Macros ---
#define LWARN(format, ...)    LOG_PER_RANK(C_YEL, "WARN", format, ##__VA_ARGS__)
#define LDEBUG(format, ...)   do { if (g_log_debug) { LOG_PER_RANK(C_MAG, "DEBUG", format, ##__VA_ARGS__); } } while (0)


// --- Root-Only Logging Macros ---
#define RINFO(format, ...)     LOG_ROOT_ONLY(C_GRN, "INFO", format, ##__VA_ARGS__)
#define RERR(format, ...)    LOG_ROOT_ONLY(C_RED, "ERROR", format, ##__VA_ARGS__)
#define RVERB(format, ...)  do { if (g_log_verbose) { LOG_ROOT_ONLY(C_CYN, "VERBOSE", format, ##__VA_ARGS__); } } while (0)
