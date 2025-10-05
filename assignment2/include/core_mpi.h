#pragma once
#include "stdbool.h"

extern int RANK;
extern int SIZE;

void init_mpi(int* argc, char*** argv);
inline bool is_root() { return RANK == 0; }

#define ROOT_DO(block) do { if (is_root()) { block; } } while (0);


