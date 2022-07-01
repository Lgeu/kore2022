#include "../marathon/nn.cpp"
#include "environment.cpp"
#include <limits>
#include <random>
#include <type_traits>

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

    // static constexpr auto kNGlobalFeatures = 9;
    // static constexpr auto kNShipyardFeatures = 10; // 嘘

    // 重み
    BatchLinear<NNUEFeature::kNGlobalFeatures, 256> global_feature_encoder;
    nn::EmbeddingBag<NNUEFeature::kNFeatureTypes + 1, 256> embedding;
    BatchLinear<256, 256> fc1, fc2;
    BatchLinear<256, 1> value_decoder;
    BatchLinear<256, 4> type_decoder;
    // この後のも Batch で処理したほうが良さそう

    template <bool has_buffer>
    using ShipyardFeatureTensor =
        nn::Tensor<int, nn::Shape<kMaxBatchSize, 512>, has_buffer>;
    template <bool has_buffer>
    using GlobalFeatureTensor =
        nn::Tensor<float,
                   nn::Shape<kMaxBatchSize, NNUEFeature::kNGlobalFeatures>,
                   has_buffer>;
    template <bool has_buffer>
    using OutValueTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize>, has_buffer>;
    template <bool has_buffer>
    using OutActionTypeTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 4>, has_buffer>;
    template <bool has_buffer>
    using OutCodeTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;

    template <bool shipyard_feature_has_buffer, bool global_feature_has_buffer,
              bool value_has_buffer, bool action_type_has_buffer,
              bool code_has_buffer>
    void
    Forward(const int batch_size,
            const ShipyardFeatureTensor<shipyard_feature_has_buffer>&
                shipyard_feature_tensor,
            const GlobalFeatureTensor<global_feature_has_buffer>&
                global_feature_tensor,
            OutValueTensor<value_has_buffer>& out_value_tensor,
            OutActionTypeTensor<action_type_has_buffer>& out_action_type_tensor,
            OutCodeTensor<code_has_buffer>& out_code_tensor) const {

        const auto x1 =
            nn::TensorSlice<float, kMaxBatchSize, 256>(out_code_tensor);
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

struct ActionSampler {};

struct NNUEGreedyAgent : Agent {
    NNUE nnue;
    SpawnDecoder spawn_decoder;
    MoveDecoder move_decoder;
    AttackDecoder attack_decoder;
    ConvertDecoder convert_decoder;

    Action ComputeNextMove(const State& state, const PlayerId = -1) const {
        // TODO
        static auto global_feature_tensor = NNUE::GlobalFeatureTensor<true>();
        static auto shipyard_feature_tensor =
            NNUE::ShipyardFeatureTensor<true>();
        static auto value_tensor = NNUE::OutValueTensor<true>();
        static auto action_type_tensor = NNUE::OutActionTypeTensor<true>();
        static auto code_tensor = NNUE::OutCodeTensor<true>();

        // 特徴抽出
        const auto features = NNUEFeature(state);

        // 特徴を NN に入れる形に変形
        auto shipyard_ids = array<ShipyardId, kMaxBatchSize>();
        auto batch_size = 0;
        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            const auto& player_shipyard_features =
                features.shipyard_features[player_id];
            const auto& player_global_features =
                features.global_features[player_id];
            for (const auto& [shipyard_id, shipyard_feature] :
                 player_shipyard_features) {
                shipyard_ids[batch_size] = shipyard_id;
                auto idx_feature = 0;
                for (; idx_feature < min(512, (int)shipyard_feature.size());
                     idx_feature++)
                    shipyard_feature_tensor[batch_size][idx_feature] =
                        shipyard_feature[idx_feature];
                if (idx_feature != 512)
                    shipyard_feature_tensor[batch_size][idx_feature] = -100;

                static_assert(
                    is_same_v<float,
                              decltype(global_feature_tensor)::value_type>);
                static_assert(
                    is_same_v<
                        float,
                        remove_reference_t<
                            decltype(player_global_features)>::value_type>);
                memcpy(global_feature_tensor[batch_size].Data(),
                       player_global_features.data(),
                       sizeof(float) * NNUEFeature::kNGlobalFeatures);

                batch_size++;
            }
        }
        assert(batch_size == (int)state.shipyards_.size());
        assert(batch_size == (int)(state.players_[0].shipyard_ids_.size() +
                                   state.players_[1].shipyard_ids_.size()));

        // NN で推論
        nnue.Forward(batch_size, shipyard_feature_tensor, global_feature_tensor,
                     value_tensor, action_type_tensor, code_tensor);

        // 集計
        auto mean_value = 0.0f; // player 0 視点
        auto action_types = array<int, kMaxBatchSize>();
        for (auto b = 0; b < batch_size; b++) {
            if (b < (int)state.players_[0].shipyard_ids_.size())
                mean_value += value_tensor[b];
            else
                mean_value -= value_tensor[b];
            nn::F::SampleFromLogit(action_type_tensor[b], action_types[b]);
        }

        // 各 Action を具体的にしていく
        auto player_spawn_capacities = array<int, 2>();
        for (auto i = 0; i < 2; i++)
            player_spawn_capacities[i] =
                (int)(state.players_[i].kore_ / kSpawnCost);
        auto result = Action();
        for (auto action_id = 0; action_id < 4; action_id++) {
            static auto action_decoder_in_tensor = NNUE::OutCodeTensor<true>();
            static auto action_shipyard_ids =
                array<ShipyardId, kMaxBatchSize>();
            auto action_batch_size = 0;
            for (auto b = 0; b < batch_size; b++)
                if (action_types[b] == action_id) {
                    action_decoder_in_tensor[action_batch_size++] =
                        code_tensor[b];
                }
            switch (action_id) {
            case 0: { // Spawn
                // 順伝播
                static auto n_ships_tensor = SpawnDecoder::NShipsTensor<true>();
                spawn_decoder.Forward(action_batch_size,
                                      action_decoder_in_tensor, n_ships_tensor);
                // shipyard ごとに処理
                for (auto ab = 0; ab < action_batch_size; ab++) {
                    const auto shipyard_id = action_shipyard_ids[ab];
                    const auto& shipyard = state.shipyards_.at(shipyard_id);
                    const auto max_spawn =
                        min(shipyard.MaxSpawn(),
                            player_spawn_capacities[shipyard.player_id_]);

                    for (auto i = max_spawn + 1; i < 12; i++)
                        n_ships_tensor[ab][i] = numeric_limits<float>::min();
                    auto n_ships = 0;
                    // spawn に関しては quantize の前後が同じ
                    nn::F::Argmax(n_ships_tensor[ab], n_ships);
                    // const auto n_ships =
                    //     kDequantizationTable[quantized_n_ships];
                    // const auto n_ships_r =
                    //     kDequantizationTable[quantized_n_ships + 1];
                    // const auto n_ships = uniform_int_distribution<>(
                    //     n_ships_l, n_ships_r - 1)(rng);

                    // TODO
                    result.actions.insert(make_pair(
                        shipyard_id,
                        ShipyardAction(ShipyardActionType::kSpawn, n_ships)));
                    player_spawn_capacities[shipyard.player_id_] -= n_ships;
                    assert(player_spawn_capacities[shipyard.player_id_] >= 0);
                }

            } break;
            case 1: { // Move
                //
                // まず DP する

                // dp[step][plan_length][y][x] 最大の kore
                // 441 * 200 = 90000
                // 10 マス以内まで見る -> 221 x 200 ~ 50000
                // 偶奇で半分のマスは使用されない 25000
                // 3 方向に 10 マスずつ 7.5 * 10^5
                // step >= plan_length
                // plan_length は高々 11 としていい

                static constexpr auto kMaxPlanLength = 11;

                // NN: 256 * (960) ~ 2.5 * 10^5
                static auto dp = array<
                    array<array<array<float, 11>, 11>, kMaxPlanLength + 1>,
                    22>();
                static auto dp_from =
                    array<array<array<array<signed char, 11>, 11>,
                                kMaxPlanLength + 1>,
                          22>();

                fill((float*)dp.begin(), (float*)dp.end(), -100.0f);

                dp[0][0][5][5] = 0.0f;
                dp_from[0][0][5][5] =
                    7; // 下 3 ビットで、0-3 が N-W 、7 が None
                for (auto step = 0; step < 21; step++) {
                    for (auto plan_length = 0;
                         plan_length < min(kMaxPlanLength, step + 1);
                         plan_length++) {
                        const auto first_uv = max(step & 1, 10 - step);
                        const auto last_uv = min(20 - (step & 1), 10 + step);
                        for (auto u = first_uv; u <= last_uv; u += 2) {
                            for (auto v = first_uv; v <= last_uv; v += 2) {
                                const auto k =
                                    dp[step][plan_length][u >> 1][v >> 1];
                                if (k == -100.0f)
                                    continue;
                                const auto y = (u + v) >> 1;
                                const auto x = (v - u) >> 1;

                                const auto last_direction =
                                    dp_from[step][plan_length][u >> 1][v >> 1] &
                                    0b111;
                                if (last_direction != 0) {
                                    const auto max_distance =
                                        10 - max(0, max(10 - u, 10 - v));
                                    if (max_distance >= 1) {
                                        const auto u2 = u - 1;
                                        const auto v2 = v - 1;
                                        auto gain = 0.0f;
                                        // TODO: ここで y と x 使って state
                                        // から量を求めたりする
                                        // 衝突判定とかも必要
                                        auto& k2 = dp[step + 1][plan_length + 1]
                                                     [u2 >> 1][v2 >> 1];
                                        if (k2 < k + gain)
                                    }
                                }
                                if (last_direction != 1) {
                                }
                                // TODO
                            }
                        }
                    }
                }

                // n_ships と n_steps_decoder を独立に決める
                // -> max_flight_plan_len が

                // relative_position_decoder は最後　いや、わからん

                // n_ships_decoder;
                // BatchLinear<256, 448> relative_position_decoder;
                // BatchLinear<256, 24> n_steps_decoder
            } break;
            case 2:
                break;
            case 3:
                break;
            }
        }

        // TODO

        (void)mean_value;

        return Action();
    }

    auto SampleAction() {
        // TODO
    }
};
struct NNUEMCTSAgent : Agent {
    NNUE nnue;
    SpawnDecoder spawn_decoder;
    MoveDecoder move_decoder;
    AttackDecoder attack_decoder;
    ConvertDecoder convert_decoder;

    Action ComputeNextMove(const State& /*state*/, const PlayerId = -1) const {
        // TODO
        return Action();
    }

    auto SampleAction() {
        // TODO
    }
};

void TestNNUE() {
    static auto s = NNUE::ShipyardFeatureTensor<true>();
    static auto g = NNUE::GlobalFeatureTensor<true>();
    static auto v = NNUE::OutValueTensor<true>();
    static auto t = NNUE::OutActionTypeTensor<true>();
    static auto c = NNUE::OutCodeTensor<true>();
    static auto nnue = NNUE();

    // TODO: パラメータ読み込み
    nnue.Forward(10, s, g, v, t, c);
}

// int main() {
//     //
// }
