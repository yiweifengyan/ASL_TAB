# ASL Spring 2024 Course Project: TAB Conv2d Optimization

This is the baseline of TAB series Ternary And Binary convolution and quantization functions. It has been verified on MSVC/Windows with a X86 CPU, and probably works well on GCC/Linux and ARM CPU.

It serves as a starting point. You can modify any part of it and add your own optimized versions, or reimplement all functions on your own.

## File Organization

- common.h
  - The common libraries and global const values (Change the popcnt intrinsic functions here)
- main.cpp
  - Verify(): all conv functions must pass the test cases to ensure code correctness.
  - Benchmark(): then you can benchmark the conv functions.
- TAB_CPU.h
- TAB_CPU.cpp
  - The integrated conv function
  - TAB_Conv(): integrate **Quantize - Img2Row/Col - Bitwise GEMM - PReLU** into one function.
- Quantize.h
- Quantize.cpp
  - Ternarize_NCHW_to_NHWCB(): Ternarize the input tensor and reshape it from NCHW to NHWCB.
  - Binarize_NCHW_to_NHWC(): Binarize the input tensor and reshape it from NCHW to NHWC.
  - BTN_CNT_W2(): BTN counts the Weight Bit2 with weight quantization.
- Img2Row.h
  - Img2Row_NHWCB_to_N_OHOW_KHKWC(): Reshape the 5-dimension NHWCB tensor into a 3-dimension (N, OH * OW, KH * KW * C) tensor. It can also be viewed as a 2-dim matrix in (N * OH * OW, KH * KW * C) for bitwise GEMM.
- GEMM.h
- GEMM.cpp
  - TNNGEMM_baseline(): Bitwise GEMM in TNN
  - TBNGEMM_baseline()
  - BTNGEMM_baseline()
  - BNNGEMM_baseline()
- Activation.h
  - PReLU(): A simple parameterized leaky ReLU function
- utility.h
  - DirectPad(): The direct padding function for standard 32-bit float conv
  - DirectConv2d_FP32(): The direct conv function provides the reference correct conv results.
  - generate_array(): Generate ternary or binary tensors for Verify().
  - Compare_Tensor_NHWC(): Compare the conv result tensor for Verify().
  - Compare_Tensor_BNN_Padding(): Compare the conv result tensor for Binary input (BNN & BTN). Zero padding is ineffective on binary (-1, +1) input activations, so this function only compares the central part of the output tensor excluding the padding part.

### Tensor Shapes

GEMM matrix shapes 

- GEMM input a: *M, K*
- GEMM input b: *N, K*
- GEMM result y: *M, N*

As the tensors in conv usually have 4 or 5 dimensions, the Conv-to-GEMM may use equivalent dimension transformation for tensor reshaping. So GEMM part only conducts the bitwise GEMM in a classical manner, and its implementation is decoupled from the convolution workflow. Sorry that GEMM uses MNK for dimension representation, these *N* and *K* are different/independent from the N and Kxx in the following convolution workflow.

Activation X

- Input: N, C, H, W
  - N is Batch_Size, Channel, Height, Width
- After Ternaize/Binarize: N, H, W, C, Bit
  -  N, H + 2 * PaddingH, W + 2 * PaddingW, C/64, BITS (C and B can be fused together)
- After Img2Row/Img2Col: N_OHOW_KNKWC
  - N, OH * OW, KH * kW * C * BITS. (C and BITS are fused. This C has been quantized)
- In GEMM: 2-Dimension (*M, K*)
  -  N * OH * OW, KH * kW * C * BITS
  -  (As GEMM input a: *M* = N * OH * OW, *K* = KH * kW * C * BITS)

Weights W

- Initial: KN, C, KH, KW
- After Ternarize/Binarize: NHWCB
  - KN, KH, KW, C/64, BITS
- In GEMM: 2-Dimension (*N, K*)
  - KN, KH * KW * C * BITS (This C has been quantized)
  - (As GEMM input b: *N* = KN, *K* = KH * KW * C * BITS)

Result Y

- GEMM output: 2-Dimension (*M, N*) = 4-Dimension (N, OH, OW, OC)
  - N * OH * OW, OC (As GEMM output y)
  - Can be viewed as N, OH, OW, OC (Output Channel = KN)
- PReLU: N, OH, OW, OC



## First-try

### Setup

- Change the popcnt64() intrinsic in common.h for GCC/Clang
- Open the .sln in MSVC, or add a makefile for GCC/Clang
- Compile and run it

The bitwise GEMM use popcnt instructions to accelerate quantized convolution. The excution speed will be very slow if current CPU don't have population count instructions.

Intrinsic references:

- [MSVC Compiler intrinsics](https://learn.microsoft.com/en-us/cpp/intrinsics/compiler-intrinsics)
- [MSVC popcnt](https://learn.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64): __popcnt64
- [GCC Compiler builtin](https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html), and in the same page
- GCC popcnt: int __builtin_popcountll (unsigned long long)
- [Clang popcnt](https://clang.llvm.org/doxygen/popcntintrin_8h_source.html)
- [Intel instrinsic guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM intrinsics whole list](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

### Tested Configuration

The MSVC/X86_64 version 
 - Windows 10
 - MS Visual Studio Community 2022

The compiler flags in old TAB versions for reference
- MSVC + X86 CPU w/ AVX2: '/arch:AVX2 /Ot'
- GCC + ARM CPU w/ Neon: '-flax-vector-conversions -march=armv8-a+simd -mfpu=neon -funsafe-math-optimizations -fbuiltin -O3'

### Example ececution results

```
Test Case 0 kernel: 3X3 TAB_TNN Passed!
Test Case 0 kernel: 3X3 TAB_TBN Passed!
Test Case 0 kernel: 3X3 TAB_BTN Passed!
Test Case 0 kernel: 3X3 TAB_BNN Passed!
Test Case 1 kernel: 1X1 TAB_TNN Passed!
Test Case 1 kernel: 1X1 TAB_TBN Passed!
Test Case 1 kernel: 1X1 TAB_BTN Passed!
Test Case 1 kernel: 1X1 TAB_BNN Passed!
Test Case 2 kernel: 1X1 TAB_TNN Passed!
Test Case 2 kernel: 1X1 TAB_TBN Passed!
Test Case 2 kernel: 1X1 TAB_BTN Passed!
Test Case 2 kernel: 1X1 TAB_BNN Passed!
Test Case 3 kernel: 3X3 TAB_TNN Passed!
Test Case 3 kernel: 3X3 TAB_TBN Passed!
Test Case 3 kernel: 3X3 TAB_BTN Passed!
Test Case 3 kernel: 3X3 TAB_BNN Passed!
...
Test Case 8 kernel: 1X1 TAB_BNN Passed!
Test Case 0 TAB_TNN Input NCHW=16,64,56,56, kernel: 64,64,3,3, y_size = 3211264 Average execution time 30962860 ns
Test Case 0 TAB_TBN Input NCHW=16,64,56,56, kernel: 64,64,3,3, y_size = 3211264 Average execution time 30888630 ns
Test Case 0 TAB_BTN Input NCHW=16,64,56,56, kernel: 64,64,3,3, y_size = 3211264 Average execution time 30909680 ns
Test Case 0 TAB_BNN Input NCHW=16,64,56,56, kernel: 64,64,3,3, y_size = 3211264 Average execution time 31426060 ns
Test Case 1 TAB_TNN Input NCHW=16,64,56,56, kernel: 128,64,3,3, y_size = 6422528 Average execution time 50741800 ns
Test Case 1 TAB_TBN Input NCHW=16,64,56,56, kernel: 128,64,3,3, y_size = 6422528 Average execution time 50860390 ns
Test Case 1 TAB_BTN Input NCHW=16,64,56,56, kernel: 128,64,3,3, y_size = 6422528 Average execution time 50583730 ns
Test Case 1 TAB_BNN Input NCHW=16,64,56,56, kernel: 128,64,3,3, y_size = 6422528 Average execution time 51339730 ns
Test Case 2 TAB_TNN Input NCHW=16,128,28,28, kernel: 128,128,3,3, y_size = 1605632 Average execution time 23153320 ns
Test Case 2 TAB_TBN Input NCHW=16,128,28,28, kernel: 128,128,3,3, y_size = 1605632 Average execution time 23116640 ns
Test Case 2 TAB_BTN Input NCHW=16,128,28,28, kernel: 128,128,3,3, y_size = 1605632 Average execution time 23273820 ns
Test Case 2 TAB_BNN Input NCHW=16,128,28,28, kernel: 128,128,3,3, y_size = 1605632 Average execution time 23153430 ns
Test Case 3 TAB_TNN Input NCHW=16,128,28,28, kernel: 256,128,3,3, y_size = 3211264 Average execution time 41445580 ns
Test Case 3 TAB_TBN Input NCHW=16,128,28,28, kernel: 256,128,3,3, y_size = 3211264 Average execution time 41529490 ns
Test Case 3 TAB_BTN Input NCHW=16,128,28,28, kernel: 256,128,3,3, y_size = 3211264 Average execution time 41525410 ns
Test Case 3 TAB_BNN Input NCHW=16,128,28,28, kernel: 256,128,3,3, y_size = 3211264 Average execution time 41495590 ns
...
Test Case 19 TAB_BNN Input NCHW=16,16000,1,1, kernel: 32000,16000,1,1, y_size = 512000 Average execution time 361291720 ns
```
