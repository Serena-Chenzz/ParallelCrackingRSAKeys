//This code is written based on code from https://github.com/mitchellwrosen/rsa-crack-cuda

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_utils.h"
#include "integer.h"
#include "rsa.h"

#define GRID_DIM 512
#define BLOCKS_PER_GRID GRID_DIM/BLOCK_DIM
#define WARP_DIM 32
#define NUM_GRIDS(N) (N-1)/GRID_DIM+1

using namespace std;

static int *crackedKeys = NULL;
static int crackedLen = 0;
static mpz_t n1, n2, p, q1, q2, d1, d2;
static int X_MASKS[BLOCK_DIM] = { 0x1111, 0x2222, 0x4444, 0x8888 };
static int Y_MASKS[BLOCK_DIM] = { 0x000F, 0x00F0, 0x0F00, 0xF000 };

// This method is to retrieve keys from the given file
inline void retrieveKeys(const char *fileName, integer *keys, int num){
        FILE *file = fopen(fileName, "r");
        // create an integer with multiple precision
        mpz_t n;
        // Initialize the memory space
        mpz_init(n);

        for(int i = 0; i < num; i++){
            //read string from f in the base 10, and assign it to n.
            mpz_inp_str(n, file, 10);
            //export n to key[i]
            mpz_export(keys[i].ints, NULL, 1, sizeof(uint32_t), 0, 0, n);
        }

        fclose(file);

}

void crackPrivateKeys(integer* keys, uint16_t* noCoprime, int gridRow, int gridCol, FILE *stream);

int main(int argc, char* argv[])
{
    // When invoking the program, you need to type in the filename and keyNum
    if (argc < 3){
        printf("The user needs to input the filename and number of keys in this file\n");
        return 0;
    }
    //Read in the filename
    const char* filename = argv[1];
    //Read in the number of keys in the file
    int keyNum = atoi(argv[2]);

    //Initialization
    integer *keys = (integer*) malloc(keyNum * sizeof(integer));
    retrieveKeys(filename, keys, keyNum);

    //Copy keys from host memory to device memory
    integer *block_keys;
    cudaSafe(cudaMalloc((void **) &block_keys, keyNum * sizeof(integer)));
    cudaSafe(cudaMemcpy(block_keys, keys, keyNum * sizeof(integer), cudaMemcpyHostToDevice));

    uint16_t *noCoprime = (uint16_t*) malloc(BLOCKS_PER_GRID * BLOCKS_PER_GRID * sizeof(uint16_t));
    uint16_t *block_noCoprime;
    cudaSafe(cudaMalloc((void **) &block_noCoprime, BLOCKS_PER_GRID * BLOCKS_PER_GRID * sizeof(uint16_t)));

    FILE *file_outputstream = argc == 4 ? fopen(argv[3], "w") : stdout;

    crackedKeys = (int *) malloc(keyNum * sizeof(int));
    mpz_inits(n1, n2, p, q1, q2, d1, d2, NULL);

    //create grids and blocks.
    dim3 grid_dim(GRID_DIM / BLOCK_DIM, GRID_DIM / BLOCK_DIM,1);
    dim3 block_dim(WARP_DIM, BLOCK_DIM, BLOCK_DIM);
    int numOfGrids = NUM_GRIDS(keyNum);

    for(int i=0; i<numOfGrids; i++){
        for(int j=i; j<numOfGrids; j++){
            //record the block which has a pair of keys which are not coprime.
            cudaSafe(cudaMemset(block_noCoprime, 0, BLOCKS_PER_GRID * BLOCKS_PER_GRID * sizeof(uint16_t)));

            cudaWrapper(grid_dim, block_dim, block_keys, block_noCoprime, i, j, GRID_DIM, keyNum);

            cudaSafe(cudaPeekAtLastError());
            cudaSafe(cudaDeviceSynchronize());

            cudaSafe(cudaMemcpy(noCoprime,block_noCoprime,BLOCKS_PER_GRID * BLOCKS_PER_GRID * sizeof(uint16_t),cudaMemcpyDeviceToHost));
            //Invoke the key cracking method to output the cracked keys
            crackPrivateKeys(keys, noCoprime, i, j, file_outputstream);

        }
    }

    cudaSafe(cudaFree(block_keys));
    cudaSafe(cudaFree(block_noCoprime));

    free(keys);
    free(noCoprime);
    free(crackedKeys);

    if (argc == 4)
      fclose(file_outputstream);

    return 0;
}

inline bool checkIfCrackedAlready(int n) {
  for (int i = 0; i < crackedLen; i++) {
    if (n == crackedKeys[i])
      return true;
  }

  return false;
}

void crackPrivateKeys(integer* keys, uint16_t* noCoprime, int gridRow, int gridCol, FILE *stream) {
  for (int i = 0; i < BLOCKS_PER_GRID; i++) {
    for (int j = 0; j < BLOCKS_PER_GRID; j++) {
      uint16_t noCoprimeBlock = noCoprime[i * BLOCKS_PER_GRID + j];

      if (noCoprimeBlock) {
        for (int y = 0; y < BLOCK_DIM; y++) {
          if (noCoprimeBlock & Y_MASKS[y]) {
            for (int x = 0; x < BLOCK_DIM; x++) {
              if (noCoprimeBlock & Y_MASKS[y] & X_MASKS[x]) {
                int n1Ndx = gridRow * GRID_DIM + i * BLOCK_DIM + y;
                int n2Ndx = gridCol * GRID_DIM + j * BLOCK_DIM + x;
                bool crackedN1 = checkIfCrackedAlready(n1Ndx);
                bool crackedN2 = checkIfCrackedAlready(n2Ndx);

                if (!crackedN1 || !crackedN2) {
                  mpz_import(n1, N, 1, sizeof(uint32_t), 0, 0, keys[n1Ndx].ints);
                  mpz_import(n2, N, 1, sizeof(uint32_t), 0, 0, keys[n2Ndx].ints);

                  mpz_gcd(p, n1, n2);

                  if (!crackedN1) {
                    mpz_divexact(q1, n1, p);
                    rsa_compute_d(d1, n1, p, q1);
                    mpz_out_str(stream, 10, n1);
                    fputc(':', stream);
                    mpz_out_str(stream, 10, d1);
                    fputc('\n', stream);

                    crackedKeys[crackedLen++] = n1Ndx;
                  }

                  if (!crackedN2) {
                    mpz_divexact(q2, n2, p);
                    rsa_compute_d(d2, n2, p, q2);
                    mpz_out_str(stream, 10, n2);
                    fputc(':', stream);
                    mpz_out_str(stream, 10, d2);
                    fputc('\n', stream);

                    crackedKeys[crackedLen++] = n2Ndx;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

