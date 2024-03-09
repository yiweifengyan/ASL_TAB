#pragma once

// Please define GCC or CLANG when using these compilers
// Their intrinsic popcnt functions are different from MSVC
//#define GCC
//#define CLANG

#include <cstdint>
#include <vector>
#include <iostream>


#ifdef CLANG
// CLANG on Mac OS use popcntintrin.h
#include <popcntintrin.h>
#define popcnt64(a)      _mm_popcnt_u64(a)
#else
#ifdef GCC
// GCC on Linux uses the nmmintrin.h
#include <nmmintrin.h>
#include <immintrin.h>
#define popcnt64(a)       __builtin_popcountll(a)
#else
// MSVC on Windows uses the intrin.h
#include <intrin.h>	
#define popcnt64(a)       __popcnt64(a)
#endif // GCC
#endif // CLANG


// The bits of the container integer: int64_t
#define cntbits 64
// The bit width of quantized input values
#define BITS 2

// The supported bitwise conv types
enum ConvType {
    TNN = 0, TBN = 1, BTN = 2, BNN = 3, Conv_Types = 4
};