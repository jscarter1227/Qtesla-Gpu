#ifndef POLY_H
#define POLY_H

#include "params.h"
#include "config.h"
#include <stdint.h>

typedef int32_t poly[PARAM_N];

__device__ __host__ int32_t reduce(int64_t a);
__device__ __host__ int32_t barr_reduce(int32_t a);
__device__ __host__ void ntt(poly a, const poly w);
__device__ __host__ void nttinv(poly a, const poly w);
__device__ __host__ void poly_mul(poly result, const poly x, const poly y);
__device__ __host__ void poly_add(poly result, const poly x, const poly y);
__device__ __host__ void poly_add_correct(poly result, const poly x, const poly y);
__device__ __host__ void poly_sub_correct(poly result, const poly x, const poly y);
__device__ __host__ void poly_sub_reduce(poly result, const poly x, const poly y);
__device__ __host__ void sparse_mul16(poly prod, const int16_t *s, const uint32_t pos_list[PARAM_H], const int16_t sign_list[PARAM_H]);
__device__ __host__ void sparse_mul32(poly prod, const int32_t *pk, const uint32_t pos_list[PARAM_H], const int16_t sign_list[PARAM_H]);
__device__ __host__ void poly_uniform(poly a, const unsigned char *seed);

#endif
