//This code is written based on code from https://github.com/mitchellwrosen/rsa-crack-cuda

#include "integer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void gcd(volatile uint32_t *x, volatile uint32_t *y);
__device__ void shiftR1(volatile uint32_t *x);
__device__ void shiftL1(volatile uint32_t *x);
__device__ int geq(volatile uint32_t *x, volatile uint32_t *y);
__device__ void cuSubtract(volatile uint32_t *x, volatile uint32_t *y, volatile uint32_t *z);

__global__ void cuda_crackKeys(const integer *keys, uint16_t *noCoprime, int gridRow, int gridCol, int grid_Dim, int keyNum) {

  // These two shared arrays are used to store key chunks.
  __shared__ volatile uint32_t keyOne[BLOCK_DIM][BLOCK_DIM][32];
  __shared__ volatile uint32_t keyTwo[BLOCK_DIM][BLOCK_DIM][32];

  int keyX = gridCol * grid_Dim + blockIdx.x * BLOCK_DIM + threadIdx.y;
  int keyY = gridRow * grid_Dim + blockIdx.y * BLOCK_DIM + threadIdx.z;

  if (keyX < keyNum && keyY < keyNum && keyX > keyY) {

    keyOne[threadIdx.y][threadIdx.z][threadIdx.x] = keys[keyX].ints[threadIdx.x];
    keyTwo[threadIdx.y][threadIdx.z][threadIdx.x] = keys[keyY].ints[threadIdx.x];

    gcd(keyOne[threadIdx.y][threadIdx.z], keyTwo[threadIdx.y][threadIdx.z]);

    if (threadIdx.x == 31) {
      keyTwo[threadIdx.y][threadIdx.z][threadIdx.x] -= 1;
       //If gcd of two keys is larger than 1.
      if (__any(keyTwo[threadIdx.y][threadIdx.z][threadIdx.x])) {
        int notCoprimeBlockNdx = blockIdx.y * gridDim.x + blockIdx.x;
        noCoprime[notCoprimeBlockNdx] |= 1 << threadIdx.z * BLOCK_DIM + threadIdx.y;
      }
    }
  }
}

void cudaWrapper(dim3 gridDim, dim3 blockDim, integer* block_keys, uint16_t* block_noCoprime,int gridRow, int gridCol, int grid_dim, int keyNum) {
      cuda_crackKeys<<<gridDim, blockDim>>>(block_keys, block_noCoprime, gridRow, gridCol, grid_dim, keyNum);
}

//The following algorithm is referred to Noriyuki's paper.
__device__ void gcd(volatile uint32_t *x, volatile uint32_t *y) {
  int tid = threadIdx.x;

  while (__any(x[tid])) {
    while ((x[31] & 1) == 0)
      shiftR1(x);

    while ((y[31] & 1) == 0)
      shiftR1(y);

    if (geq(x, y)) {
      cuSubtract(x, y, x);
      shiftR1(x);
    }
    else {
      cuSubtract(y, x, y);
      shiftR1(y);
    }
  }
}

__device__ void shiftR1(volatile uint32_t *x) {
  int tid = threadIdx.x;
  uint32_t prevX = tid ? x[tid-1] : 0;
  x[tid] = (x[tid] >> 1) | (prevX << 31);
}

__device__ void shiftL1(volatile uint32_t *x) {
  int tid = threadIdx.x;
  uint32_t nextX = tid != 31 ? x[tid+1] : 0;
  x[tid] = (x[tid] << 1) | (nextX >> 31);
}

__device__ int geq(volatile uint32_t *x, volatile uint32_t *y) {
  /* shared memory to hold the position at which the int of x >= int of y */
  __shared__ unsigned int pos[BLOCK_DIM][BLOCK_DIM];
  int tid = threadIdx.x;

  if (tid == 0)
    pos[threadIdx.y][threadIdx.z] = 31;

  if (x[tid] != y[tid])
    atomicMin(&pos[threadIdx.y][threadIdx.z], tid);

  return x[pos[threadIdx.y][threadIdx.z]] >= y[pos[threadIdx.y][threadIdx.z]];
}

__device__ void cuSubtract(volatile uint32_t *x, volatile uint32_t *y, volatile uint32_t *z) {
  /* shared memory to hold underflow flags */
  __shared__ unsigned char s_borrow[BLOCK_DIM][BLOCK_DIM][32];
  unsigned char *borrow = s_borrow[threadIdx.y][threadIdx.z];
  int tid = threadIdx.x;

  /* set LSB's borrow to 0 */
  if (tid == 0)
    borrow[31] = 0;

  uint32_t t;
  t = x[tid] - y[tid];

  /* set the previous int's underflow flag if the subtraction answer is bigger than the subtractee */
  if(tid)
    borrow[tid - 1] = (t > x[tid]);

  /* keep processing until there's no flags */
  while (__any(borrow[tid])) {
    if (borrow[tid])
      t--;

    /* have to set flag if the new sub answer is 0xFFFFFFFF becuase of an underflow */
    if (tid)
      borrow[tid - 1] = (t == 0xFFFFFFFFu && borrow[tid]);
  }

  z[tid] = t;
}
