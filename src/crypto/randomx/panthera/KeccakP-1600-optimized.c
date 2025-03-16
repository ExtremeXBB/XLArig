/*
 * Optimized wrapper implementation for Keccak-p[1600] permutation function
 * Selects the best implementation based on CPU support
 */

#include <stdint.h>
#include "KeccakP-1600-SnP.h"

// Reference implementation declaration (already exists in KeccakP-1600-reference.c)
void KeccakP1600_Permute_24rounds_reference(void *state);

// AVX2 implementation declaration
#if defined(__AVX2__)
void KeccakP1600_Permute_24rounds_avx2(void *state);
int supportsAVX2(void);
#endif

// AVX-512 implementation declaration
#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
void KeccakP1600_Permute_24rounds_avx512(void *state);
int supportsAVX512(void);
#endif

// Function pointer for dynamically selecting the best implementation
static void (*permute_fn)(void *) = NULL;

// Initialization function - selects the best implementation based on CPU features
static void init_permute_function(void)
{
    if (permute_fn != NULL) {
        return;
    }

    // Use reference implementation by default
    permute_fn = &KeccakP1600_Permute_24rounds_reference;

    // If AVX2 is supported, use AVX2 optimized version
    #if defined(__AVX2__)
    if (supportsAVX2()) {
        permute_fn = &KeccakP1600_Permute_24rounds_avx2;
    }
    #endif

    // If AVX-512 is supported, use AVX-512 optimized version
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    if (supportsAVX512()) {
        permute_fn = &KeccakP1600_Permute_24rounds_avx512;
    }
    #endif
}

// Wrapper function - replaces the original KeccakP1600_Permute_24rounds
void KeccakP1600_Permute_24rounds(void *state)
{
    if (permute_fn == NULL) {
        init_permute_function();
    }
    
    permute_fn(state);
}