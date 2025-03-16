/*
 * AVX2 implementation of Keccak-p[1600] permutation function
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include "KeccakP-1600-SnP.h"
#include "align.h"
#include "brg_endian.h"

#if defined(__AVX2__)

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

// Loop unrolling
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
CACHE_ALIGNED static const uint8_t chi_row_indices[5][5] = {
    {0, 1, 2, 3, 4},
    {5, 6, 7, 8, 9},
    {10, 11, 12, 13, 14},
    {15, 16, 17, 18, 19},
    {20, 21, 22, 23, 24}
};

// Precomputed rotation constant pairs - to accelerate rotation operations
CACHE_ALIGNED static const uint8_t rot_pairs[25][2] = {
    {0, 0}, {1, 63}, {62, 2}, {28, 36}, {27, 37},
    {36, 28}, {44, 20}, {6, 58}, {55, 9}, {20, 44},
    {3, 61}, {10, 54}, {43, 21}, {25, 39}, {39, 25},
    {41, 23}, {45, 19}, {15, 49}, {21, 43}, {8, 56},
    {18, 46}, {2, 62}, {61, 3}, {56, 8}, {14, 50}
};

// ======== Work buffers ========
// Buffer for extracting vector elements - cache line aligned
CACHE_ALIGNED static uint64_t extract_buffer[4];

// ======== Core functions ========
// Efficiently extract 64-bit element from __m256i
FORCE_INLINE uint64_t get_epi64(__m256i a, int idx) {
    _mm256_store_si256((__m256i*)extract_buffer, a);
    return extract_buffer[idx];
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
FORCE_INLINE uint64_t mm256_horizontal_xor_epi64(__m256i v) {
    v = _mm256_xor_si256(v, _mm256_permute4x64_epi64(v, 0x4E)); // 2,3,0,1 ^ 0,1,2,3
    v = _mm256_xor_si256(v, _mm256_permute4x64_epi64(v, 0xB1)); // 1,0,3,2 ^ 0,1,2,3
    return _mm256_extract_epi64(v, 0);
}

// Efficient implementation of Theta step - using vector operations to calculate column sums
FORCE_INLINE void theta_step_avx2(uint64_t* RESTRICT state, uint64_t* RESTRICT c, uint64_t* RESTRICT d) {
    __m256i col0 = _mm256_set_epi64x(state[20], state[15], state[10], state[5]);
    col0 = _mm256_xor_si256(col0, _mm256_set1_epi64x(state[0]));
    
    __m256i col1 = _mm256_set_epi64x(state[21], state[16], state[11], state[6]);
    col1 = _mm256_xor_si256(col1, _mm256_set1_epi64x(state[1]));
    
    __m256i col2 = _mm256_set_epi64x(state[22], state[17], state[12], state[7]);
    col2 = _mm256_xor_si256(col2, _mm256_set1_epi64x(state[2]));
    
    __m256i col3 = _mm256_set_epi64x(state[23], state[18], state[13], state[8]);
    col3 = _mm256_xor_si256(col3, _mm256_set1_epi64x(state[3]));
    
    __m256i col4 = _mm256_set_epi64x(state[24], state[19], state[14], state[9]);
    col4 = _mm256_xor_si256(col4, _mm256_set1_epi64x(state[4]));
    
    // Calculate the total XOR value for each column
    c[0] = mm256_horizontal_xor_epi64(col0);
    c[1] = mm256_horizontal_xor_epi64(col1);
    c[2] = mm256_horizontal_xor_epi64(col2);
    c[3] = mm256_horizontal_xor_epi64(col3);
    c[4] = mm256_horizontal_xor_epi64(col4);
    
    // Calculate d values with forced loop unrolling
    #define D_CALC(x, x1, x4) d[x] = c[x4] ^ rotl64(c[x1], 1)
    D_CALC(0, 1, 4);
    D_CALC(1, 2, 0);
    D_CALC(2, 3, 1);
    D_CALC(3, 4, 2);
    D_CALC(4, 0, 3);
    #undef D_CALC
}

// Chi step optimizer - process specific row
FORCE_INLINE void chi_process_row(uint64_t* RESTRICT state, int row_base) {
    const uint64_t a0 = state[row_base + 0];
    const uint64_t a1 = state[row_base + 1];
    const uint64_t a2 = state[row_base + 2];
    const uint64_t a3 = state[row_base + 3];
    const uint64_t a4 = state[row_base + 4];
    
    // Use circular access to avoid conditional branches
    state[row_base + 0] = a0 ^ ((~a1) & a2);
    state[row_base + 1] = a1 ^ ((~a2) & a3);
    state[row_base + 2] = a2 ^ ((~a3) & a4);
    state[row_base + 3] = a3 ^ ((~a4) & a0);
    state[row_base + 4] = a4 ^ ((~a0) & a1);
}

// Single round calculation for Keccak-p[1600]
FORCE_INLINE void KeccakP1600_Round_avx2(uint64_t* RESTRICT state, uint64_t rc) {
    // Use local cache to minimize memory access
    CACHE_ALIGNED uint64_t temp[25];
    CACHE_ALIGNED uint64_t c[5], d[5];
    __m256i lanes0, lanes1, lanes2, lanes3; 

    // ========== THETA step ==========
    // Prefetch data to cache
    PREFETCH(state);
    PREFETCH(state + 16);
    
    // Calculate c and d values needed for theta transformation
    theta_step_avx2(state, c, d);
    
    // Apply theta transformation to first 16 state elements using vector operations
    lanes0 = _mm256_loadu_si256((__m256i*)state);     // 0-3
    lanes1 = _mm256_loadu_si256((__m256i*)(state+4)); // 4-7
    lanes2 = _mm256_loadu_si256((__m256i*)(state+8)); // 8-11
    lanes3 = _mm256_loadu_si256((__m256i*)(state+12)); // 12-15
    
    // Apply d values - efficient vector processing
    lanes0 = _mm256_xor_si256(lanes0, _mm256_set_epi64x(d[3], d[2], d[1], d[0]));
    lanes1 = _mm256_xor_si256(lanes1, _mm256_set_epi64x(d[2], d[1], d[0], d[4]));
    lanes2 = _mm256_xor_si256(lanes2, _mm256_set_epi64x(d[1], d[0], d[4], d[3]));
    lanes3 = _mm256_xor_si256(lanes3, _mm256_set_epi64x(d[0], d[4], d[3], d[2]));
    
    // Process remaining 9 elements - forced loop unrolling
    #define THETA_TAIL(i, col) state[i] ^= d[col]
    THETA_TAIL(16, 1);
    THETA_TAIL(17, 2);
    THETA_TAIL(18, 3);
    THETA_TAIL(19, 4);
    THETA_TAIL(20, 0);
    THETA_TAIL(21, 1);
    THETA_TAIL(22, 2);
    THETA_TAIL(23, 3);
    THETA_TAIL(24, 4);
    #undef THETA_TAIL
    
    // ========== RHO and PI steps ==========
    // Store intermediate state and prefetch write location
    _mm256_storeu_si256((__m256i*)temp, lanes0);
    _mm256_storeu_si256((__m256i*)(temp+4), lanes1);
    _mm256_storeu_si256((__m256i*)(temp+8), lanes2);
    _mm256_storeu_si256((__m256i*)(temp+12), lanes3);
    memcpy(temp+16, state+16, 9 * sizeof(uint64_t));
    
    PREFETCH_WRITE(state);
    PREFETCH_WRITE(state + 16);
    
    state[0] = temp[0];
    
    // Fully unrolled Rho+Pi steps, using manual unrolling to avoid any loop overhead
    #define RHO_PI_STEP(idx, src, rot) state[idx] = rotl64(temp[src], rot)
    
    RHO_PI_STEP(5, 1, 1);    // (0,1) -> (1,0)
    RHO_PI_STEP(10, 2, 62);  // (0,2) -> (2,0)
    RHO_PI_STEP(15, 3, 28);  // (0,3) -> (3,0)
    RHO_PI_STEP(20, 4, 27);  // (0,4) -> (4,0)
    
    RHO_PI_STEP(1, 5, 36);   // (1,0) -> (0,1)
    RHO_PI_STEP(6, 6, 44);   // (1,1) -> (1,1)
    RHO_PI_STEP(11, 7, 6);   // (1,2) -> (2,1)
    RHO_PI_STEP(16, 8, 55);  // (1,3) -> (3,1)
    RHO_PI_STEP(21, 9, 20);  // (1,4) -> (4,1)
    
    RHO_PI_STEP(2, 10, 3);   // (2,0) -> (0,2)
    RHO_PI_STEP(7, 11, 10);  // (2,1) -> (1,2)
    RHO_PI_STEP(12, 12, 43); // (2,2) -> (2,2)
    RHO_PI_STEP(17, 13, 25); // (2,3) -> (3,2)
    RHO_PI_STEP(22, 14, 39); // (2,4) -> (4,2)
    
    RHO_PI_STEP(3, 15, 41);  // (3,0) -> (0,3)
    RHO_PI_STEP(8, 16, 45);  // (3,1) -> (1,3)
    RHO_PI_STEP(13, 17, 15); // (3,2) -> (2,3)
    RHO_PI_STEP(18, 18, 21); // (3,3) -> (3,3)
    RHO_PI_STEP(23, 19, 8);  // (3,4) -> (4,3)
    
    RHO_PI_STEP(4, 20, 18);  // (4,0) -> (0,4)
    RHO_PI_STEP(9, 21, 2);   // (4,1) -> (1,4)
    RHO_PI_STEP(14, 22, 61); // (4,2) -> (2,4)
    RHO_PI_STEP(19, 23, 56); // (4,3) -> (3,4)
    RHO_PI_STEP(24, 24, 14); // (4,4) -> (4,4)
    
    #undef RHO_PI_STEP
    
    // ========== CHI step ==========
    // Efficient row-by-row processing, using function inlining to optimize compiler-generated code
    chi_process_row(state, 0);
    chi_process_row(state, 5);
    chi_process_row(state, 10);
    chi_process_row(state, 15);
    chi_process_row(state, 20);
    
    // ========== IOTA step ==========
    state[0] ^= rc;
}

// Complete Keccak-p[1600] permutation function - applies 24 rounds of transformation
HOT_FUNCTION
void KeccakP1600_Permute_24rounds_avx2(void *stateArg) {
    uint64_t *state = (uint64_t*)stateArg;
    
    // Prefetch key data to cache
    PREFETCH(KeccakF1600RoundConstants);
    PREFETCH(KeccakF1600RoundConstants + 16);
    
    // Use super-unrolled round loop, with macros to enhance readability
    #define KECCAK_ROUND(i) KeccakP1600_Round_avx2(state, KeccakF1600RoundConstants[i])
    
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
    
    #undef KECCAK_ROUND
}

// 12-round permutation function - used in some applications
HOT_FUNCTION
void KeccakP1600_Permute_12rounds_avx2(void *stateArg) {
    uint64_t *state = (uint64_t*)stateArg;
    
    // Prefetch key data
    PREFETCH(KeccakF1600RoundConstants + 12);
    
    // Execute the last 12 transformations (12-23)
    #define KECCAK_ROUND(i) KeccakP1600_Round_avx2(state, KeccakF1600RoundConstants[i+12])
    
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
    
    #undef KECCAK_ROUND
}

// Detect if CPU supports AVX2
int supportsAVX2(void) {
    #if defined(__GNUC__) || defined(__clang__)
        uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
        
        __asm__ __volatile__ (
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(7), "c"(0)
        );
        
        return (ebx & (1 << 5)) != 0;
    #elif defined(_MSC_VER)
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        return (cpuInfo[1] & (1 << 5)) != 0;
    #else
        return 0;
    #endif
}

#endif // defined(__AVX2__) 