# This cmakelists works on ARM CPU with GCC
# Tested environment: Rpi 400, Rpi OS, cmake=3.25, GCC=12.2.0 
# It should work for similar GNU/Lunix based OS

cmake_minimum_required(VERSION 3.16) 
project(ASL_TAB)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flax-vector-conversions -march=armv8-a+simd -funsafe-math-optimizations -fbuiltin -O3")
# Reference cmake flags for X86_64 CPU
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -msse4.2 -mavx2 -mpopcnt -march=core-avx2 -O3")

file(GLOB_RECURSE sources TAB/*.cpp TAB/*.h)
add_executable(main ${sources})
