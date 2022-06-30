#include "../marathon/nn.cpp"

// バッチサイズが 1 じゃないので BLAS 的なもの使う必要がある

extern "C" {
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* A, const int* ldA,
            const float* B, const int* ldB, const float* beta, float* C,
            const int* ldC);
}

template <int in_features, int out_features> struct BatchLinear {
    // パラメータ
    nn::TensorBuffer<float, out_features, in_features> weight;
    nn::TensorBuffer<float, out_features> bias;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        weight.FromFile(f);
        bias.FromFile(f);
    }

    inline void Forward(const int batch_size, const float* const input,
                        float* const output) const {
        // weight: [o, i]
        // input: [b, i]
        // output: [b, o]

        // i * w.T = o

        for (auto i = 0; i < batch_size; i++)
            memcpy(output + i * sizeof(bias), bias.Data(), sizeof(bias));

        static constexpr auto o = out_features;
        static constexpr auto i = in_features;
        static constexpr auto alpha = 1.0f;
        static constexpr auto beta = 1.0f;

        sgemm_("T", "N", &o, &batch_size, &i, &alpha, weight.Data(), &i, input,
               &i, &beta, output, &o);
    }
};

static constexpr auto kMaxBatchSize = 100;

struct SpawnDecoder {
    BatchLinear<256, 12> n_ships_decoder;

    template <bool has_buffer>
    using InTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;
    template <bool has_buffer>
    using NShipsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 12>, has_buffer>;

    template <bool in_has_buffer, bool out_has_buffer>
    void Forward(const int batch_size, const InTensor<in_has_buffer>& input,
                 NShipsTensor<out_has_buffer>& out_n_ships) const {
        n_ships_decoder.Forward(batch_size, input.Data(), out_n_ships.Data());
    }

    // デクオンタイズ、サンプリングもここでやる？微妙
};

struct MoveDecoder {
    BatchLinear<256, 32> n_ships_decoder;
    BatchLinear<256, 448> relative_position_decoder;
    BatchLinear<256, 24> n_steps_decoder; // [1, 21]

    template <bool has_buffer>
    using InTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;
    template <bool has_buffer>
    using NShipsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 32>, has_buffer>;
    template <bool has_buffer>
    using RelativePositionTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 448>, has_buffer>;
    template <bool has_buffer>
    using NStepsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 24>, has_buffer>;

    template <bool in_has_buffer, bool n_ships_has_buffer,
              bool relative_position_has_buffer, bool n_steps_has_buffer>
    void Forward(const int batch_size, const InTensor<in_has_buffer>& input,
                 NShipsTensor<n_ships_has_buffer>& out_n_ships,
                 RelativePositionTensor<relative_position_has_buffer>&
                     out_relative_position,
                 NStepsTensor<n_steps_has_buffer>& out_n_steps) const {
        n_ships_decoder.Forward(batch_size, input.Data(), out_n_ships.Data());
        relative_position_decoder.Forward(batch_size, input.Data(),
                                          out_relative_position.Data());
        n_steps_decoder.Forward(batch_size, input.Data(), out_n_steps.Data());
    }
};

struct AttackDecoder {
    BatchLinear<256, 32> n_ships_decoder;
    BatchLinear<256, 448> relative_position_decoder;
    BatchLinear<256, 4> direction_decoder;

    template <bool has_buffer>
    using InTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;
    template <bool has_buffer>
    using NShipsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 32>, has_buffer>;
    template <bool has_buffer>
    using RelativePositionTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 448>, has_buffer>;
    template <bool has_buffer>
    using DirectionTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 4>, has_buffer>;

    template <bool in_has_buffer, bool n_ships_has_buffer,
              bool relative_position_has_buffer, bool direction_has_buffer>
    void Forward(const int batch_size, const InTensor<in_has_buffer>& input,
                 NShipsTensor<n_ships_has_buffer>& out_n_ships,
                 RelativePositionTensor<relative_position_has_buffer>&
                     out_relative_position,
                 DirectionTensor<direction_has_buffer>& out_direction) const {
        n_ships_decoder.Forward(batch_size, input.Data(), out_n_ships.Data());
        relative_position_decoder.Forward(batch_size, input.Data(),
                                          out_relative_position.Data());
        direction_decoder.Forward(batch_size, input.Data(),
                                  out_direction.Data());
    }
};

struct ConvertDecoder : AttackDecoder {
    // BatchLinear<256, 32> n_ships_decoder;
    // BatchLinear<256, 448> relative_position_decoder;
    // BatchLinear<256, 4> direction_decoder;
};

struct NNUE {
    // vector で持つよりはこっちのほうがいいはず

    static constexpr auto kNGlobalFeatures = 9;
    static constexpr auto kNShipyardFeatures = 10; // 嘘

    // 重み
    BatchLinear<kNGlobalFeatures, 256> global_feature_encoder;
    nn::EmbeddingBag<kNShipyardFeatures + 1, 256> embedding;
    BatchLinear<256, 256> fc1, fc2;
    BatchLinear<256, 1> value_decoder;
    BatchLinear<256, 4> type_decoder;
    // この後のも Batch で処理したほうが良さそう

    template <bool has_buffer>
    using ShipyardFeatureTensor =
        nn::Tensor<int, nn::Shape<kMaxBatchSize, 512>, has_buffer>;
    template <bool has_buffer>
    using GlobalFeatureTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, kNGlobalFeatures>,
                   has_buffer>;
    template <bool has_buffer>
    using OutValueTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize>, has_buffer>;
    template <bool has_buffer>
    using OutActionTypeTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 4>, has_buffer>;

    template <bool shipyard_feature_has_buffer, bool global_feature_has_buffer,
              bool value_has_buffer, bool action_type_has_buffer>
    void Forward(const int batch_size,
                 const ShipyardFeatureTensor<shipyard_feature_has_buffer>&
                     shipyard_feature_tensor,
                 const GlobalFeatureTensor<global_feature_has_buffer>&
                     global_feature_tensor,
                 OutValueTensor<value_has_buffer>& out_value_tensor,
                 OutActionTypeTensor<action_type_has_buffer>&
                     out_action_type_tensor) const {

        static auto x1 = nn::TensorBuffer<float, kMaxBatchSize, 256>();
        for (auto b = 0; b < batch_size; b++) {
            auto r = x1[b]; // 右辺値参照は関数に渡せない、うーん
            embedding.template Forward<-100>(shipyard_feature_tensor[b], r);
        }

        static auto x2 = nn::TensorBuffer<float, kMaxBatchSize, 256>();
        global_feature_encoder.Forward(batch_size, global_feature_tensor.Data(),
                                       x2.Data());
        for (auto b = 0; b < batch_size; b++)
            x1[b] += x2[b];

        static constexpr auto kNegativeSlope = 1.0f / 64.0f;
        for (auto b = 0; b < batch_size; b++)
            x1[b].LeakyRelu_(kNegativeSlope);

        fc1.Forward(batch_size, x1.Data(), x2.Data());
        for (auto b = 0; b < batch_size; b++)
            x2[b].LeakyRelu_(kNegativeSlope);

        fc2.Forward(batch_size, x2.Data(), x1.Data());
        for (auto b = 0; b < batch_size; b++)
            x1[b].LeakyRelu_(kNegativeSlope);

        value_decoder.Forward(batch_size, x1.Data(), out_value_tensor.Data());
        type_decoder.Forward(batch_size, x1.Data(),
                             out_action_type_tensor.Data());

        // 細かい decoder は別の構造体にする
    }
};

void TestNNUE() {
    static auto s = NNUE::ShipyardFeatureTensor<true>();
    static auto g = NNUE::GlobalFeatureTensor<true>();
    static auto v = NNUE::OutValueTensor<true>();
    static auto t = NNUE::OutActionTypeTensor<true>();
    static auto nnue = NNUE();
    nnue.Forward(10, s, g, v, t);
}

// int main() {
//     //
// }
