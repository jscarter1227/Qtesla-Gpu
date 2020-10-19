#ifndef FIPS202_H
#define FIPS202_H

#define SHAKE128_RATE 168
#define SHA3_256_RATE 136
#define SHAKE256_RATE 136

#define NROUNDS 24


__device__ __host__ void shake128_absorb(uint64_t *s, const unsigned char *input, unsigned int inputByteLen);
__device__ __host__ void shake128_squeezeblocks(unsigned char *output, unsigned long long nblocks, uint64_t *s);
__device__ __host__ void shake128(unsigned char *output, unsigned long long outputByteLen, const unsigned char *input, unsigned long long inputByteLen);
__device__ __host__ void shake256(unsigned char *output, unsigned long long outlen, const unsigned char *input,  unsigned long long inlen);
__device__ __host__ void sha3256(unsigned char *output, const unsigned char *input, unsigned int inputByteLen);
__device__ __host__ void cshake128_simple(unsigned char *output, unsigned long long outlen, uint16_t cstm, const unsigned char *in, unsigned long long inlen);
__device__ __host__ void cshake256_simple(unsigned char *output, unsigned long long outlen, uint16_t cstm, const unsigned char *in, unsigned long long inlen);

#endif
