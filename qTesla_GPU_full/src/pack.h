#ifndef PACK_H
#define PACK_H

#include "poly.h"
#include <stdint.h>

__device__ __host__ void hash_H(unsigned char *c_bin, poly v, const unsigned char *hm);
__device__ __host__ void encode_sk(unsigned char *sk, const poly s, const poly e, const unsigned char *seeds);
__device__ __host__ void decode_sk(unsigned char *seeds, int16_t *s, int16_t *e, const unsigned char *sk);
__device__ __host__ void encode_pk(unsigned char *pk, const poly t, const unsigned char *seedA);
__device__ __host__ void decode_pk(int32_t *pk, unsigned char *seedA, const unsigned char *pk_in);
__device__ __host__ void encode_sig(unsigned char *sm, unsigned char *c, poly z);
__device__ __host__ void decode_sig(unsigned char *c, poly z, const unsigned char *sm);

#endif
