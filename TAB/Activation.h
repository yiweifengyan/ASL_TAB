#pragma once

#include "common.h"

// The simple Parameterized leaky ReLU function
// The tensor shape doesn't matter, because it apply PReLU on each value of the Conv Result
template <typename T>
std::vector<float> PReLU(T* x, int N, int C, int H, int W, float alpha) {

    std::vector<float> y = std::vector<float>(N * C * H * W);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                        T current=x[((n * C + c) * H + h) * W + w];
                        if (current > 0)
                            y[((n * C + c) * H + h) * W + w] = current;
                        else
                            y[((n * C + c) * H + h) * W + w] = current * alpha;
                }
            }
        }
    }

    return y;
}