/*
 * AVX-512 implementation of Keccak-p[1600] permutation function
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include "KeccakP-1600-SnP.h"
#include "align.h"
#include "brg_endian.h"

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)

// ======== Compiler optimization directives ========
#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline, hot, flatten, optimize("O3"))) inline
#define LIKELY(x)    __builtin_expect(!!(x), 1)
#define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#define RESTRICT     __restrict__
#define VECTORIZE    __attribute__((vectorize))
#elif defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#define LIKELY(x)    (x)
#define UNLIKELY(x)  (x)
#define RESTRICT     __restrict
#define VECTORIZE
#else
#define FORCE_INLINE inline
#define LIKELY(x)    (x)
#define UNLIKELY(x)  (x)
#define RESTRICT
#define VECTORIZE
#endif

// loop unrolling and vectorization
#if defined(__clang__)
#define UNROLL_LOOPS _Pragma("clang loop unroll(full) vectorize(enable)")
#elif defined(__GNUC__)
#define UNROLL_LOOPS _Pragma("GCC unroll 32") _Pragma("GCC ivdep") _Pragma("GCC optimize(\"unroll-loops\")")
#elif defined(_MSC_VER)
#define UNROLL_LOOPS __pragma(loop(full_unroll))
#else
#define UNROLL_LOOPS
#endif

// Cache and memory optimizations
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr)       __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 0)
#define CACHE_ALIGNED        __attribute__((aligned(64)))
#define HOT_FUNCTION         __attribute__((hot, optimize("O3")))
#else
#define PREFETCH(addr)
#define PREFETCH_WRITE(addr)
#define CACHE_ALIGNED        ALIGN(64)
#define HOT_FUNCTION
#endif

// ======== Constants definitions ========
// Round constants - cache line aligned
CACHE_ALIGNED static const uint64_t KeccakF1600RoundConstants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rho rotation values - using uint8_t to reduce memory and cache usage
CACHE_ALIGNED static const uint8_t rhotates[25] = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14
};

// Pi step target index mapping - for fast conversion
CACHE_ALIGNED static const uint8_t pi_lane_map[25] = {
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4
};

// Chi step row indices - optimized for row operations
CACHE_ALIGNED static const uint8_t chi_mask_offsets[5] = {1, 2, 3, 4, 0};

// Precomputed rotation constant pairs - to accelerate rotation operations
CACHE_ALIGNED static const uint8_t row_indices[5][5] = {
    { 0,  1,  2,  3,  4},
    { 5,  6,  7,  8,  9},
    {10, 11, 12, 13, 14},
    {15, 16, 17, 18, 19},
    {20, 21, 22, 23, 24}
};

CACHE_ALIGNED static uint64_t extract_buffer[8];

// ======== Core functions ========
// Efficiently extract 64-bit elements from __m512i
FORCE_INLINE uint64_t extract_epi64(__m512i vec, int pos) {
    _mm512_store_si512((__m512i*)extract_buffer, vec);
    return extract_buffer[pos];
}

// Bit rotation - using lookup tables and precomputed constants
FORCE_INLINE uint64_t rotl64(uint64_t x, int n) {
    switch(n) {
        case 0: return x;
        case 1: return (x << 1) | (x >> 63);
        case 2: return (x << 2) | (x >> 62);
        case 3: return (x << 3) | (x >> 61);
        case 6: return (x << 6) | (x >> 58);
        case 8: return (x << 8) | (x >> 56);
        case 10: return (x << 10) | (x >> 54);
        case 14: return (x << 14) | (x >> 50);
        case 15: return (x << 15) | (x >> 49);
        case 18: return (x << 18) | (x >> 46);
        case 20: return (x << 20) | (x >> 44);
        case 21: return (x << 21) | (x >> 43);
        case 25: return (x << 25) | (x >> 39);
        case 27: return (x << 27) | (x >> 37);
        case 28: return (x << 28) | (x >> 36);
        case 36: return (x << 36) | (x >> 28);
        case 39: return (x << 39) | (x >> 25);
        case 41: return (x << 41) | (x >> 23);
        case 43: return (x << 43) | (x >> 21);
        case 44: return (x << 44) | (x >> 20);
        case 45: return (x << 45) | (x >> 19);
        case 55: return (x << 55) | (x >> 9);
        case 56: return (x << 56) | (x >> 8);
        case 61: return (x << 61) | (x >> 3);
        case 62: return (x << 62) | (x >> 2);
        case 63: return (x << 63) | (x >> 1);
        default: {
            const int normalized_n = n & 63;
            return (x << normalized_n) | (x >> (64 - normalized_n));
        }
    }
}

// Horizontal vector XOR - fast calculation of XOR of all vector elements
FORCE_INLINE uint64_t mm512_horizontal_xor_epi64(__m512i v) {
    __m256i low256 = _mm512_extracti64x4_epi64(v, 0);
    __m256i high256 = _mm512_extracti64x4_epi64(v, 1);
    __m256i reduced = _mm256_xor_si256(low256, high256);
    
    __m128i low128 = _mm256_extracti128_si256(reduced, 0);
    __m128i high128 = _mm256_extracti128_si256(reduced, 1);
    __m128i result = _mm_xor_si128(low128, high128);
    
    uint64_t val = _mm_extract_epi64(result, 0) ^ _mm_extract_epi64(result, 1);
    return val;
}

// High-performance implementation of Theta step - using AVX-512 instruction set
FORCE_INLINE void theta_step_avx512(uint64_t* RESTRICT state, uint64_t* RESTRICT c, uint64_t* RESTRICT d) {

    __m512i col0 = _mm512_set_epi64(0, 0, 0, state[20], state[15], state[10], state[5], state[0]);
    __m512i col1 = _mm512_set_epi64(0, 0, 0, state[21], state[16], state[11], state[6], state[1]);
    __m512i col2 = _mm512_set_epi64(0, 0, 0, state[22], state[17], state[12], state[7], state[2]);
    __m512i col3 = _mm512_set_epi64(0, 0, 0, state[23], state[18], state[13], state[8], state[3]);
    __m512i col4 = _mm512_set_epi64(0, 0, 0, state[24], state[19], state[14], state[9], state[4]);
    
    c[0] = mm512_horizontal_xor_epi64(col0);
    c[1] = mm512_horizontal_xor_epi64(col1);
    c[2] = mm512_horizontal_xor_epi64(col2);
    c[3] = mm512_horizontal_xor_epi64(col3);
    c[4] = mm512_horizontal_xor_epi64(col4);
    
    #define D_CALC(x, x1, x4) d[x] = c[x4] ^ rotl64(c[x1], 1)
    D_CALC(0, 1, 4);
    D_CALC(1, 2, 0);
    D_CALC(2, 3, 1);
    D_CALC(3, 4, 2);
    D_CALC(4, 0, 3);
    #undef D_CALC
}

// Row processing optimized with AVX512 instructions - specialized for Chi step
FORCE_INLINE void chi_process_row_avx512(uint64_t* RESTRICT state, int row_base) {

    __m512i row = _mm512_set_epi64(0, 0, 0, state[row_base+4], state[row_base+3], 
                                   state[row_base+2], state[row_base+1], state[row_base]);
    
    __m512i row_shifted1 = _mm512_permutexvar_epi64(
        _mm512_set_epi64(0, 0, 0, 0, 3, 2, 1, 5), row);
    
    __m512i row_shifted2 = _mm512_permutexvar_epi64(
        _mm512_set_epi64(0, 0, 0, 0, 4, 3, 2, 1), row);
    
    __mmask8 full_mask = 0x1F; 
    __m512i not_row = _mm512_xor_si512(row_shifted1, _mm512_set1_epi64(-1));
    __m512i chi_result = _mm512_and_si512(not_row, row_shifted2);
    
    row = _mm512_mask_xor_epi64(row, full_mask, row, chi_result);
    
    _mm512_mask_storeu_epi64(state + row_base, full_mask, row);
}

// Optimized Keccak-p[1600] single round function
FORCE_INLINE void KeccakP1600_Round_avx512(uint64_t* RESTRICT state, uint64_t rc) {
    CACHE_ALIGNED uint64_t temp[25];
    CACHE_ALIGNED uint64_t c[5], d[5];
    __m512i s0, s1;
    
    // Prefetch state to cache
    PREFETCH(state);
    PREFETCH(state + 16);
    
    // ===================== THETA step =====================
    // Optimized Theta calculation
    theta_step_avx512(state, c, d);
    
    // Load state to vector registers
    s0 = _mm512_loadu_si512((const __m512i *)state);
    s1 = _mm512_loadu_si512((const __m512i *)(state+8));
    
    __m512i d_broadcast;

    // Apply d values to s0 (elements 0-7)
    d_broadcast = _mm512_set_epi64(d[3], d[2], d[1], d[0], d[4], d[3], d[2], d[1]);
    s0 = _mm512_xor_si512(s0, d_broadcast);

    // Apply d values to s1 (elements 8-15)
    d_broadcast = _mm512_set_epi64(d[2], d[1], d[0], d[4], d[3], d[2], d[1], d[0]);
    s1 = _mm512_xor_si512(s1, d_broadcast);

    // Apply d values to remaining elements - fully unrolled loop
    #define APPLY_D(i, col) state[i] ^= d[col]
    APPLY_D(16, 1);
    APPLY_D(17, 2);
    APPLY_D(18, 3);
    APPLY_D(19, 4);
    APPLY_D(20, 0);
    APPLY_D(21, 1);
    APPLY_D(22, 2);
    APPLY_D(23, 3);
    APPLY_D(24, 4);
    #undef APPLY_D
    
    // ===================== RHO and PI steps =====================
    // Store intermediate state
    _mm512_storeu_si512((__m512i*)temp, s0);
    _mm512_storeu_si512((__m512i*)(temp+8), s1);
    memcpy(temp+16, state+16, 9 * sizeof(uint64_t));
    
    // Prefetch write location
    PREFETCH_WRITE(state);
    PREFETCH_WRITE(state + 16);
    
    // Fully unrolled Rho and Pi steps
    state[0] = temp[0]; 
    
    // Use macros to expand all Rho+Pi operations
    #define RHO_PI(dest, src, rot) state[dest] = rotl64(temp[src], rot)
    
    RHO_PI(1*5+0, 1, 1);    // (0,1) -> (1,0)
    RHO_PI(2*5+0, 2, 62);   // (0,2) -> (2,0)
    RHO_PI(3*5+0, 3, 28);   // (0,3) -> (3,0)
    RHO_PI(4*5+0, 4, 27);   // (0,4) -> (4,0)
    
    RHO_PI(0*5+1, 5, 36);   // (1,0) -> (0,1)
    RHO_PI(1*5+1, 6, 44);   // (1,1) -> (1,1)
    RHO_PI(2*5+1, 7, 6);    // (1,2) -> (2,1)
    RHO_PI(3*5+1, 8, 55);   // (1,3) -> (3,1)
    RHO_PI(4*5+1, 9, 20);   // (1,4) -> (4,1)
    
    RHO_PI(0*5+2, 10, 3);   // (2,0) -> (0,2)
    RHO_PI(1*5+2, 11, 10);  // (2,1) -> (1,2)
    RHO_PI(2*5+2, 12, 43);  // (2,2) -> (2,2)
    RHO_PI(3*5+2, 13, 25);  // (2,3) -> (3,2)
    RHO_PI(4*5+2, 14, 39);  // (2,4) -> (4,2)
    
    RHO_PI(0*5+3, 15, 41);  // (3,0) -> (0,3)
    RHO_PI(1*5+3, 16, 45);  // (3,1) -> (1,3)
    RHO_PI(2*5+3, 17, 15);  // (3,2) -> (2,3)
    RHO_PI(3*5+3, 18, 21);  // (3,3) -> (3,3)
    RHO_PI(4*5+3, 19, 8);   // (3,4) -> (4,3)
    
    RHO_PI(0*5+4, 20, 18);  // (4,0) -> (0,4)
    RHO_PI(1*5+4, 21, 2);   // (4,1) -> (1,4)
    RHO_PI(2*5+4, 22, 61);  // (4,2) -> (2,4)
    RHO_PI(3*5+4, 23, 56);  // (4,3) -> (3,4)
    RHO_PI(4*5+4, 24, 14);  // (4,4) -> (4,4)
    
    #undef RHO_PI
    
    // ===================== CHI step =====================
    // Use AVX-512 optimized row processing
    chi_process_row_avx512(state, 0);
    chi_process_row_avx512(state, 5);
    chi_process_row_avx512(state, 10);
    chi_process_row_avx512(state, 15);
    chi_process_row_avx512(state, 20);
    
    // ===================== IOTA step =====================
    state[0] ^= rc;
}

// Round unrolling helper macro
#define KECCAK_ROUND(i) KeccakP1600_Round_avx512(state, KeccakF1600RoundConstants[i])

// Main permutation function - fully unrolled 24 rounds
HOT_FUNCTION
void KeccakP1600_Permute_24rounds_avx512(void *stateArg) {
    uint64_t *state = (uint64_t*)stateArg;
    
    // Prefetch constants to cache
    PREFETCH(KeccakF1600RoundConstants);
    PREFETCH(KeccakF1600RoundConstants + 8);
    PREFETCH(KeccakF1600RoundConstants + 16);
    
    // First 12 transformation rounds
    KECCAK_ROUND(0);
    KECCAK_ROUND(1);
    KECCAK_ROUND(2);
    KECCAK_ROUND(3);
    KECCAK_ROUND(4);
    KECCAK_ROUND(5);
    KECCAK_ROUND(6);
    KECCAK_ROUND(7);
    KECCAK_ROUND(8);
    KECCAK_ROUND(9);
    KECCAK_ROUND(10);
    KECCAK_ROUND(11);
    
    // Last 12 transformation rounds
    KECCAK_ROUND(12);
    KECCAK_ROUND(13);
    KECCAK_ROUND(14);
    KECCAK_ROUND(15);
    KECCAK_ROUND(16);
    KECCAK_ROUND(17);
    KECCAK_ROUND(18);
    KECCAK_ROUND(19);
    KECCAK_ROUND(20);
    KECCAK_ROUND(21);
    KECCAK_ROUND(22);
    KECCAK_ROUND(23);
}

// 12-round permutation function
HOT_FUNCTION
void KeccakP1600_Permute_12rounds_avx512(void *stateArg) {
    uint64_t *state = (uint64_t*)stateArg;
    
    // Prefetch constants
    PREFETCH(KeccakF1600RoundConstants + 12);
    
    // Execute the last 12 transformations
    KECCAK_ROUND(12);
    KECCAK_ROUND(13);
    KECCAK_ROUND(14);
    KECCAK_ROUND(15);
    KECCAK_ROUND(16);
    KECCAK_ROUND(17);
    KECCAK_ROUND(18);
    KECCAK_ROUND(19);
    KECCAK_ROUND(20);
    KECCAK_ROUND(21);
    KECCAK_ROUND(22);
    KECCAK_ROUND(23);
}

#undef KECCAK_ROUND

// Detect if CPU supports AVX-512
int supportsAVX512(void) {
    #if defined(__GNUC__) || defined(__clang__)
        uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
        
        __asm__ __volatile__ (
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(7), "c"(0)
        );
        
        const uint32_t avx512_mask = (1 << 16) | (1 << 17) | (1 << 30) | (1 << 31);
        return ((ebx & avx512_mask) == avx512_mask);
    #elif defined(_MSC_VER)
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        const int avx512_mask = (1 << 16) | (1 << 17) | (1 << 30) | (1 << 31);
        return ((cpuInfo[1] & avx512_mask) == avx512_mask);
    #else
        return 0;
    #endif
}

#endif // AVX-512 conditional compilation 