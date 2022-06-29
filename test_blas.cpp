#include <array>
#include <iostream>

using namespace std;

extern "C" {
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* A, const int* ldA,
            const float* B, const int* ldB, const float* beta, float* C,
            const int* ldC);
}

int main() {
    static constexpr auto o = 3;
    static constexpr auto i = 2;
    static constexpr auto b = 4;
    static constexpr auto alpha = 1.0f;
    static constexpr auto beta = 1.0f;
    const auto input = array<array<float, i>, b>{
        array<float, i>{0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}, {6.0f, 7.0f}};
    const auto weight = array<array<float, i>, o>{
        array<float, i>{0.0f, 2.0f}, {3.0f, 5.0f}, {7.0f, 11.0f}};
    auto output = array<array<float, o>, b>{};

    // O <- I x W.T

    sgemm_("T", "N", &o, &b, &i, &alpha, &weight[0][0], &i, &input[0][0], &i,
           &beta, &output[0][0], &o);

    for (auto out_y : output) {
        for (auto out_yx : out_y)
            cout << out_yx << " ";
        cout << endl;
    }
}

// clang++ test_blas.cpp -l:libblas.so.3
