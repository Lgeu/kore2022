#include "../marathon/nn.cpp"
#include "environment.cpp"
#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

// バッチサイズが 1 じゃないので BLAS 的なもの使う必要がある

extern "C" {
void sgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const float* alpha, const float* A, const int* ldA,
            const float* B, const int* ldB, const float* beta, float* C,
            const int* ldC);
}

inline auto TranslatePosition221ToUV(const int p) {
    const auto u = p < 121 ? p / 11 : (p - 121) / 10;
    const auto v = p < 121 ? p % 11 : (p - 121) % 10;
    return make_pair(u, v);
}

inline auto TranslatePosition221ToYX(const int p) {
    const auto [u, v] = TranslatePosition221ToUV(p);
    const auto y = u + v - (p < 121 ? 10 : 9);
    const auto x = v - u;
    return make_pair(y, x);
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

        for (auto b = 0; b < batch_size; b++)
            memcpy(output + b * bias.n_data, bias.Data(),
                   sizeof(float) * bias.n_data);

        static constexpr auto o = out_features;
        static constexpr auto i = in_features;
        static constexpr auto alpha = 1.0f;
        static constexpr auto beta = 1.0f;

        sgemm_("T", "N", &o, &batch_size, &i, &alpha, weight.Data(), &i, input,
               &i, &beta, output, &o);
    }
};

void ReadRelativePositionDecoderParameters(
    BatchLinear<256, 224>& relative_position_decoder, FILE* const f) {
    static auto raw_weight = nn::TensorBuffer<float, 448, 256>();
    static auto raw_bias = nn::TensorBuffer<float, 448>();
    raw_weight.FromFile(f);
    raw_bias.FromFile(f);
    for (auto i = 0; i < 221; i++) {
        auto [y, x] = TranslatePosition221ToYX(i);
        if (y < 0)
            y += kSize;
        if (x < 0)
            x += kSize;
        const auto yx = y * kSize + x;
        relative_position_decoder.weight[i] = raw_weight[yx];
        relative_position_decoder.bias[i] = raw_bias[yx];
    }
}

static constexpr auto kMaxBatchSize = 100;

struct SpawnDecoder {
    BatchLinear<256, 12> n_ships_decoder;

    template <bool has_buffer>
    using InTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;
    template <bool has_buffer>
    using NShipsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 12>, has_buffer>;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) { n_ships_decoder.ReadParameters(f); }

    template <bool in_has_buffer, bool out_has_buffer>
    void Forward(const int batch_size, const InTensor<in_has_buffer>& input,
                 NShipsTensor<out_has_buffer>& out_n_ships) const {
        n_ships_decoder.Forward(batch_size, input.Data(), out_n_ships.Data());
    }
};

struct MoveDecoder {
    BatchLinear<256, 32> n_ships_decoder;
    BatchLinear<256, 224> relative_position_decoder;
    BatchLinear<256, 24> n_steps_decoder; // [1, 21]

    template <bool has_buffer>
    using InTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;
    template <bool has_buffer>
    using NShipsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 32>, has_buffer>;
    template <bool has_buffer>
    using RelativePositionTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 224>, has_buffer>;
    template <bool has_buffer>
    using NStepsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 24>, has_buffer>;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        n_ships_decoder.ReadParameters(f);
        ReadRelativePositionDecoderParameters(relative_position_decoder, f);
        n_steps_decoder.ReadParameters(f);
    }

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
    BatchLinear<256, 224> relative_position_decoder;
    BatchLinear<256, 4> direction_decoder;

    template <bool has_buffer>
    using InTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 256>, has_buffer>;
    template <bool has_buffer>
    using NShipsTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 32>, has_buffer>;
    template <bool has_buffer>
    using RelativePositionTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 224>, has_buffer>;
    template <bool has_buffer>
    using DirectionTensor =
        nn::Tensor<float, nn::Shape<kMaxBatchSize, 4>, has_buffer>;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        n_ships_decoder.ReadParameters(f);
        ReadRelativePositionDecoderParameters(relative_position_decoder, f);
        direction_decoder.ReadParameters(f);
    }

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
    // 重み
    BatchLinear<NNUEFeature::kNGlobalFeatures, 256> global_feature_encoder;
    nn::EmbeddingBag<NNUEFeature::kNFeatureTypes + 1, 256> embedding;
    BatchLinear<256, 256> fc1, fc2;
    BatchLinear<256, 1> value_decoder;
    BatchLinear<256, 4> type_decoder;

    void ReadParameters(FILE* const f) {
        global_feature_encoder.ReadParameters(f);
        embedding.ReadParameters(f);
        fc1.ReadParameters(f);
        fc2.ReadParameters(f);
        value_decoder.ReadParameters(f);
        type_decoder.ReadParameters(f);
    }

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

    void ReadParameters(FILE* const f) {
        nnue.ReadParameters(f);
        spawn_decoder.ReadParameters(f);
        move_decoder.ReadParameters(f);
        attack_decoder.ReadParameters(f);
        convert_decoder.ReadParameters(f);
    }

    void ReadParameters(const string& filename) {
        const auto f = fopen(filename.c_str(), "rb");
        if (f == NULL) {
            cerr << filename << " を開けないよ" << endl;
            abort();
        }
        ReadParameters(f);
        if (!(getc(f) == EOF && feof(f))) {
            cerr << "読み込むファイルが大きすぎるよ" << endl;
            abort();
        }
        fclose(f);
    }

    NNUEGreedyAgent(const string parameters_filename) {
        ReadParameters(parameters_filename);
    }

    inline auto FeatureToTensor(
        const NNUEFeature& features,
        NNUE::ShipyardFeatureTensor<true>& shipyard_feature_tensor,
        NNUE::GlobalFeatureTensor<true>& global_feature_tensor) const {
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
                    is_same_v<
                        float,
                        remove_reference_t<
                            decltype(global_feature_tensor)>::value_type>);
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
        return make_pair(batch_size, shipyard_ids);
    }

    Action ComputeNextMove(const State& state, const PlayerId = -1) const {
        static auto global_feature_tensor = NNUE::GlobalFeatureTensor<true>();
        static auto shipyard_feature_tensor =
            NNUE::ShipyardFeatureTensor<true>();
        static auto value_tensor = NNUE::OutValueTensor<true>();
        static auto action_type_tensor = NNUE::OutActionTypeTensor<true>();
        static auto code_tensor = NNUE::OutCodeTensor<true>();

        // 特徴抽出
        const auto features = NNUEFeature(state);

        // 特徴を NN に入れる形に変形
        const auto [batch_size, shipyard_ids] = FeatureToTensor(
            features, shipyard_feature_tensor, global_feature_tensor);
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

            // 艦数が足りないなら convert は弾く
            if (state.shipyards_.at(shipyard_ids[b]).ship_count_ < kConvertCost)
                action_type_tensor[b][3] = -1e30f;

            nn::F::SampleFromLogit(action_type_tensor[b], action_types[b]);
        }
        mean_value /= batch_size;

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
                    action_shipyard_ids[action_batch_size] = shipyard_ids[b];
                    action_decoder_in_tensor[action_batch_size++] =
                        code_tensor[b];
                }
            const auto normalize = [](const int x) {
                return x < 0 ? x + kSize : x >= kSize ? x - kSize : x;
            };
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
                    result.actions.insert(make_pair(
                        shipyard_id,
                        ShipyardAction(ShipyardActionType::kSpawn, n_ships)));
                    player_spawn_capacities[shipyard.player_id_] -= n_ships;
                    assert(player_spawn_capacities[shipyard.player_id_] >= 0);
                }

            } break;
            case 1: { // Move
                // 順伝播
                static auto n_ships_tensor = MoveDecoder::NShipsTensor<true>();
                static auto relative_position_tensor =
                    MoveDecoder::RelativePositionTensor<true>();
                static auto n_steps_tensor = MoveDecoder::NStepsTensor<true>();
                move_decoder.Forward(action_batch_size,
                                     action_decoder_in_tensor, n_ships_tensor,
                                     relative_position_tensor, n_steps_tensor);

                // まず DP する

                // dp[step][plan_length][y][x] 最大の kore
                // 441 * 200 = 90000
                // 10 マス以内まで見る -> 221 x 200 ~ 50000
                // 偶奇で半分のマスは使用されない 25000
                // 3 方向に 10 マスずつ 7.5 * 10^5
                // step >= plan_length
                // plan_length は高々 9 とする
                // - plan_length を 10 にするには 91 隻必要

                static constexpr auto kMaxPlanLength = 9;

                // NN: 256 * (960) ~ 2.5 * 10^5
                static auto dp = array<
                    array<array<array<float, 11>, 11>, kMaxPlanLength + 2>,
                    22>();
                static auto dp_from =
                    array<array<array<array<signed char, 11>, 11>,
                                kMaxPlanLength + 2>,
                          22>();

                // shipyard ごとに処理
                for (auto ab = 0; ab < action_batch_size; ab++) {
                    const auto shipyard_id = action_shipyard_ids[ab];
                    const auto& shipyard = state.shipyards_.at(shipyard_id);
                    const auto mask_opponent =
                        shipyard.player_id_ == 0
                            ? NNUEFeature::kPlayer1Shipyard |
                                  NNUEFeature::kPlayer1Fleet |
                                  NNUEFeature::kPlayer1FleetAdjacent
                            : NNUEFeature::kPlayer0Shipyard |
                                  NNUEFeature::kPlayer0Fleet |
                                  NNUEFeature::kPlayer0FleetAdjacent;
                    const auto mask_plan_length_1 =
                        shipyard.player_id_ == 0
                            ? NNUEFeature::kPlayer0Shipyard |
                                  NNUEFeature::kPlayer0Fleet
                            : NNUEFeature::kPlayer1Shipyard |
                                  NNUEFeature::kPlayer1Fleet;

                    fill((float*)dp.begin(), (float*)dp.end(), -1e30f);
                    fill((char*)dp_from.begin(), (char*)dp_from.end(), 255);

                    dp[0][0][5][5] = 0.0f;
                    // 下 3 ビットで、0-3 が N-W 、7 が None
                    dp_from[0][0][5][5] = 7;
                    // 各マスについて、目的地たりえるか集計する配列
                    auto can_be_destination = array<bool, 224>();
                    for (auto step = 0; step < 21; step++) {
                        for (auto plan_length = 0;
                             plan_length < min(kMaxPlanLength, step + 1);
                             plan_length++) {
                            const auto first_uv = max(step & 1, 10 - step);
                            const auto last_uv =
                                min(20 - (step & 1), 10 + step);
                            for (auto u = first_uv; u <= last_uv; u += 2) {
                                for (auto v = first_uv; v <= last_uv; v += 2) {
                                    const auto k =
                                        dp[step][plan_length][u >> 1][v >> 1];
                                    if (k == -100.0f)
                                        continue;
                                    // ここの u, v は 2 倍してあるので注意
                                    const auto dy = ((u + v) >> 1) - 10;
                                    const auto y =
                                        normalize(shipyard.position_.y + dy);
                                    const auto dx = ((v - u) >> 1);
                                    const auto x =
                                        normalize(shipyard.position_.x + dx);
                                    if (features.future_info[step][{y, x}]
                                                .flags != 0 &&
                                        step >= 1)
                                        continue;

                                    const auto last_direction =
                                        dp_from[step][plan_length][u >> 1]
                                               [v >> 1] &
                                        0b111;

                                    // North y-, u-, v-
                                    auto cumulative_kore = k;
                                    do {
                                        if (last_direction == (int)Direction::N)
                                            break;
                                        const auto max_distance = min(
                                            21 - step,
                                            10 - max(0, max(10 - u, 10 - v)));

                                        for (auto distance = 1;
                                             distance <= max_distance;
                                             distance++) {

                                            const auto y2 =
                                                y >= distance
                                                    ? y - distance
                                                    : y - distance + kSize;
                                            const auto x2 = x;
                                            const auto& target_info =
                                                features.future_info[step +
                                                                     distance]
                                                                    [{y2, x2}];

                                            // 相手の造船所、相手の艦隊、相手の隣接艦隊がある場合、計算しない
                                            if (target_info.flags &
                                                mask_opponent)
                                                break;

                                            const auto u2 = (u - distance) >> 1;
                                            const auto v2 = (v - distance) >> 1;
                                            cumulative_kore += target_info.kore;
                                            const auto d_plan_length =
                                                distance == 1 ||
                                                        (target_info.flags &
                                                         mask_plan_length_1)
                                                    ? 1
                                                    : 2;

                                            auto& k2 =
                                                dp[step + distance]
                                                  [plan_length + d_plan_length]
                                                  [u2][v2];
                                            if (k2 < cumulative_kore) {
                                                k2 = cumulative_kore;
                                                dp_from[step + distance]
                                                       [plan_length +
                                                        d_plan_length][u2][v2] =
                                                           (distance - 1) << 3 |
                                                           (int)Direction::N;
                                            }

                                            // 回収する場合も直帰でいい
                                            if (target_info.flags &
                                                mask_plan_length_1) {
                                                can_be_destination
                                                    [(step + distance) % 2
                                                         ? 121 + u2 * 10 + v2
                                                         : u2 * 11 + v2] = true;
                                                break;
                                            }
                                        }
                                    } while (false);
                                    // East x+, u-, v+
                                    cumulative_kore = k;
                                    do {
                                        if (last_direction == (int)Direction::E)
                                            break;
                                        const auto max_distance = min(
                                            21 - step,
                                            10 - max(0, max(10 - u, v - 10)));

                                        for (auto distance = 1;
                                             distance <= max_distance;
                                             distance++) {

                                            const auto y2 = y;
                                            const auto x2 =
                                                x + distance < kSize
                                                    ? x + distance
                                                    : x + distance - kSize;
                                            const auto& target_info =
                                                features.future_info[step +
                                                                     distance]
                                                                    [{y2, x2}];

                                            // 相手の造船所、相手の艦隊、相手の隣接艦隊がある場合、計算しない
                                            if (target_info.flags &
                                                mask_opponent)
                                                break;

                                            const auto u2 = (u - distance) >> 1;
                                            const auto v2 = (v + distance) >> 1;
                                            cumulative_kore += target_info.kore;
                                            const auto d_plan_length =
                                                distance == 1 ||
                                                        (target_info.flags &
                                                         mask_plan_length_1)
                                                    ? 1
                                                    : 2;

                                            auto& k2 =
                                                dp[step + distance]
                                                  [plan_length + d_plan_length]
                                                  [u2][v2];
                                            if (k2 < cumulative_kore) {
                                                k2 = cumulative_kore;
                                                dp_from[step + distance]
                                                       [plan_length +
                                                        d_plan_length][u2][v2] =
                                                           (distance - 1) << 3 |
                                                           (int)Direction::E;
                                            }

                                            // 回収する場合も直帰でいい
                                            if (target_info.flags &
                                                mask_plan_length_1) {
                                                can_be_destination
                                                    [(step + distance) % 2
                                                         ? 121 + u2 * 10 + v2
                                                         : u2 * 11 + v2] = true;
                                                break;
                                            }
                                        }
                                    } while (false);
                                    // South y+, u+, v+
                                    cumulative_kore = k;
                                    do {
                                        if (last_direction == (int)Direction::S)
                                            break;
                                        const auto max_distance = min(
                                            21 - step,
                                            10 - max(0, max(u - 10, v - 10)));

                                        for (auto distance = 1;
                                             distance <= max_distance;
                                             distance++) {

                                            const auto y2 =
                                                y + distance < kSize
                                                    ? y + distance
                                                    : y + distance - kSize;
                                            const auto x2 = x;
                                            const auto& target_info =
                                                features.future_info[step +
                                                                     distance]
                                                                    [{y2, x2}];

                                            // 相手の造船所、相手の艦隊、相手の隣接艦隊がある場合、計算しない
                                            if (target_info.flags &
                                                mask_opponent)
                                                break;

                                            const auto u2 = (u + distance) >> 1;
                                            const auto v2 = (v + distance) >> 1;
                                            cumulative_kore += target_info.kore;
                                            const auto d_plan_length =
                                                distance == 1 ||
                                                        (target_info.flags &
                                                         mask_plan_length_1)
                                                    ? 1
                                                    : 2;

                                            auto& k2 =
                                                dp[step + distance]
                                                  [plan_length + d_plan_length]
                                                  [u2][v2];
                                            if (k2 < cumulative_kore) {
                                                k2 = cumulative_kore;
                                                dp_from[step + distance]
                                                       [plan_length +
                                                        d_plan_length][u2][v2] =
                                                           (distance - 1) << 3 |
                                                           (int)Direction::S;
                                            }

                                            // 回収する場合も直帰でいい
                                            if (target_info.flags &
                                                mask_plan_length_1) {
                                                can_be_destination
                                                    [(step + distance) % 2
                                                         ? 121 + u2 * 10 + v2
                                                         : u2 * 11 + v2] = true;
                                                break;
                                            }
                                        }
                                    } while (false);
                                    // West x-, u+, v-
                                    cumulative_kore = k;
                                    do {
                                        if (last_direction == (int)Direction::W)
                                            break;
                                        const auto max_distance = min(
                                            21 - step,
                                            10 - max(0, max(u - 10, 10 - v)));

                                        for (auto distance = 1;
                                             distance <= max_distance;
                                             distance++) {

                                            const auto y2 = y;
                                            const auto x2 =
                                                x >= distance
                                                    ? x - distance
                                                    : x - distance + kSize;
                                            ;
                                            const auto& target_info =
                                                features.future_info[step +
                                                                     distance]
                                                                    [{y2, x2}];

                                            // 相手の造船所、相手の艦隊、相手の隣接艦隊がある場合、計算しない
                                            if (target_info.flags &
                                                mask_opponent)
                                                break;

                                            const auto u2 = (u + distance) >> 1;
                                            const auto v2 = (v - distance) >> 1;
                                            cumulative_kore += target_info.kore;
                                            const auto d_plan_length =
                                                distance == 1 ||
                                                        (target_info.flags &
                                                         mask_plan_length_1)
                                                    ? 1
                                                    : 2;

                                            auto& k2 =
                                                dp[step + distance]
                                                  [plan_length + d_plan_length]
                                                  [u2][v2];
                                            if (k2 < cumulative_kore) {
                                                k2 = cumulative_kore;
                                                dp_from[step + distance]
                                                       [plan_length +
                                                        d_plan_length][u2][v2] =
                                                           (distance - 1) << 3 |
                                                           (int)Direction::W;
                                            }

                                            // 回収する場合も直帰でいい
                                            if (target_info.flags &
                                                mask_plan_length_1) {
                                                can_be_destination
                                                    [(step + distance) % 2
                                                         ? 121 + u2 * 10 + v2
                                                         : u2 * 11 + v2] = true;
                                                break;
                                            }
                                        }
                                    } while (false);
                                }
                            }
                        }
                    }

                    // 最初に目的地を決める
                    for (auto i = 0; i < 224; i++)
                        if (!can_be_destination[i])
                            relative_position_tensor[ab][i] = -1e30f;
                    auto relative_position = 0;
                    nn::F::Argmax(relative_position_tensor[ab],
                                  relative_position);

                    // n_steps を、position から定まる可能なものから決める
                    auto n_steps_candidates = array<bool, 24>();
                    const auto [target_u, target_v] =
                        TranslatePosition221ToUV(relative_position);
                    for (auto step = relative_position < 121 ? 2 : 1;
                         step <= 21; step += 2)
                        for (auto plan_length = 1;
                             plan_length <= min(kMaxPlanLength, step);
                             plan_length++)
                            n_steps_candidates[step] |=
                                dp[step][plan_length][target_u][target_v] >=
                                0.0f;
                    for (auto i = 0; i < 24; i++)
                        if (!n_steps_candidates[i])
                            n_steps_tensor[ab][i] = -1e30f;
                    auto n_steps = 0;
                    nn::F::Argmax(n_steps_tensor[ab], n_steps);

                    // n_ships を、可能なものに決める
                    auto min_n_ships = 999;
                    for (auto plan_length = 1;
                         plan_length <= min(kMaxPlanLength, n_steps);
                         plan_length++) {
                        if (dp[n_steps][plan_length][target_u][target_v] >=
                            0.0f) {
                            // plan_length から n_ships への変換
                            static constexpr auto mapping =
                                array<int, kMaxPlanLength + 1>{
                                    0, 1, 2, 3, 5, 8, 13, 21, 34, 55,
                                };
                            min_n_ships = mapping[plan_length];
                            break;
                        }
                    }
                    const auto max_n_ships = shipyard.ship_count_;
                    if (min_n_ships > max_n_ships) {
                        // 失敗処理はこれでいいのか？
                        cerr << "失敗しちゃったよ move" << endl;
                        continue;
                    }
                    const auto n_ships = DetermineNShips(
                        min_n_ships, max_n_ships, n_ships_tensor[ab]);

                    // もっとも良い plan_length をテーブルを走査して見つける
                    auto best_plan_length = 0;
                    auto best_score = -100.0f;
                    for (auto plan_length = 1;
                         plan_length <=
                         Fleet::MaxFlightPlanLenForShipCount(n_ships);
                         plan_length++) {
                        const auto& score =
                            dp[n_steps][plan_length][target_u][target_v];
                        if (best_score < score) {
                            best_score = score;
                            best_plan_length = plan_length;
                        }
                    }
                    if (best_plan_length == 0) {
                        cerr << "失敗 2" << endl;
                        continue;
                    }

                    // 経路復元
                    // 回収する場合は、戻る文字を残すようにする // TODO
                    auto flight_plan = string();
                    auto path_y = target_u + target_v + n_steps % 2 - 10;
                    auto path_x = target_v - target_u;
                    auto path_step = n_steps;
                    auto path_plan_length = best_plan_length;
                    do {
                        const auto u = (10 + path_y - path_x) >> 1;
                        const auto v = (10 + path_y + path_x) >> 1;
                        assert(dp[path_step][path_plan_length][u][v] >= 0.0f);
                        const auto number_direction =
                            dp_from[path_step][path_plan_length][u][v];
                        const auto direction = number_direction & 0b111;
                        if (direction == 0b111) {
                            assert(path_plan_length == 0);
                            assert(path_step == 0);
                            assert(path_y == 0);
                            assert(path_x == 0);
                            break;
                        }
                        assert(path_step > 0);
                        const auto number = number_direction >> 3;
                        if (number)
                            flight_plan += to_string(number);
                        flight_plan += "NESW"[direction];
                        switch ((Direction)direction) {
                        case Direction::N:
                            path_y += 1 + number;
                            break;
                        case Direction::E:
                            path_x -= 1 + number;
                            break;
                        case Direction::S:
                            path_y -= 1 + number;
                            break;
                        case Direction::W:
                            path_x += 1 + number;
                            break;
                        }
                        path_step -= 1 + number;
                        path_plan_length -= number == 0 ? 1 : 2;
                    } while (true);
                    reverse(flight_plan.begin(), flight_plan.end());
                    if ('0' <= flight_plan.back() && flight_plan.back() <= '9')
                        flight_plan.pop_back();
                    result.actions.insert(make_pair(
                        shipyard_id, ShipyardAction(ShipyardActionType::kLaunch,
                                                    n_ships, flight_plan)));
                }
            } break;
            case 2: { // Attack
                static auto n_ships_tensor =
                    AttackDecoder::NShipsTensor<true>();
                static auto relative_position_tensor =
                    AttackDecoder::RelativePositionTensor<true>();
                static auto direction_tensor =
                    AttackDecoder::DirectionTensor<true>();
                attack_decoder.Forward(
                    action_batch_size, action_decoder_in_tensor, n_ships_tensor,
                    relative_position_tensor, direction_tensor);

                // shipyard ごとに処理
                for (auto ab = 0; ab < action_batch_size; ab++) {
                    const auto shipyard_id = action_shipyard_ids[ab];
                    const auto& shipyard = state.shipyards_.at(shipyard_id);
                    const auto mask_opponent =
                        shipyard.player_id_ == 0
                            ? NNUEFeature::kPlayer1Shipyard |
                                  NNUEFeature::kPlayer1Fleet |
                                  NNUEFeature::kPlayer1FleetAdjacent
                            : NNUEFeature::kPlayer0Shipyard |
                                  NNUEFeature::kPlayer0Fleet |
                                  NNUEFeature::kPlayer0FleetAdjacent;

                    // 最初に目標位置を決める
                    // 10 マス以内の敵関連のマス
                    for (auto dy = -10; dy <= 10; dy++) {
                        for (auto dx = -10 + abs(dy); dx <= 10 - abs(dy);
                             dx++) {
                            const auto y = normalize(shipyard.position_.y + dy);
                            const auto x = normalize(shipyard.position_.x + dx);
                            if ((dy != 0 || dx != 0) &&
                                (features.future_info[abs(y) + abs(x)][{y, x}]
                                     .flags &
                                 mask_opponent))
                                continue;
                            const auto u = (10 + y - x) >> 1;
                            const auto v = (10 + y + x) >> 1;
                            const auto idx_relative_position =
                                ((y ^ x) & 1) ? 121 + u * 10 + v : u * 11 + v;
                            relative_position_tensor[ab]
                                                    [idx_relative_position] =
                                                        -1e30f;
                        }
                    }

                    auto target_position = 0;
                    nn::F::Argmax(relative_position_tensor[ab],
                                  target_position);
                    if (relative_position_tensor[ab][target_position] <=
                        -1e30f) {
                        cerr << "失敗 attack" << endl;
                        continue;
                    }
                    const auto [target_y, target_x] =
                        TranslatePosition221ToYX(target_position);

                    // 初手の方向を決める
                    if (target_y >= 0)
                        direction_tensor[ab][0] = -1e30f;
                    if (target_x <= 0)
                        direction_tensor[ab][1] = -1e30f;
                    if (target_y <= 0)
                        direction_tensor[ab][2] = -1e30f;
                    if (target_x >= 0)
                        direction_tensor[ab][3] = -1e30f;
                    auto best_direction = 0;
                    nn::F::Argmax(direction_tensor[ab], best_direction);

                    // 艦数を決める
                    const auto min_n_ships = 21;
                    const auto max_n_ships = shipyard.ship_count_;
                    const auto n_ships = DetermineNShips(
                        min_n_ships, max_n_ships, n_ships_tensor[ab]);

                    // 航路を決める
                    const auto flight_plan =
                        DetermineFlightPlan(target_y, target_x, best_direction);

                    // TODO: 経路に障害物がないか検証

                    result.actions.insert(make_pair(
                        shipyard_id, ShipyardAction(ShipyardActionType::kLaunch,
                                                    n_ships, flight_plan)));
                }
            } break;
            case 3: { // Convert
                static auto n_ships_tensor =
                    ConvertDecoder::NShipsTensor<true>();
                static auto relative_position_tensor =
                    ConvertDecoder::RelativePositionTensor<true>();
                static auto direction_tensor =
                    ConvertDecoder::DirectionTensor<true>();
                convert_decoder.Forward(
                    action_batch_size, action_decoder_in_tensor, n_ships_tensor,
                    relative_position_tensor, direction_tensor);

                // shipyard ごとに処理
                for (auto ab = 0; ab < action_batch_size; ab++) {
                    const auto shipyard_id = action_shipyard_ids[ab];
                    const auto& shipyard = state.shipyards_.at(shipyard_id);

                    // 最初に目標を定める
                    auto target_position = 0;
                    nn::F::Argmax(relative_position_tensor[ab],
                                  target_position);
                    const auto [target_y, target_x] =
                        TranslatePosition221ToYX(target_position);

                    // 初手の方向を決める
                    if (target_y >= 0)
                        direction_tensor[ab][0] = -1e30f;
                    if (target_x <= 0)
                        direction_tensor[ab][1] = -1e30f;
                    if (target_y <= 0)
                        direction_tensor[ab][2] = -1e30f;
                    if (target_x >= 0)
                        direction_tensor[ab][3] = -1e30f;
                    auto best_direction = 0;
                    nn::F::Argmax(direction_tensor[ab], best_direction);

                    // 艦数を決める
                    const auto min_n_ships = 0;
                    const auto max_n_ships = shipyard.ship_count_ - 50;
                    const auto n_ships =
                        DetermineNShips(min_n_ships, max_n_ships,
                                        n_ships_tensor[ab]) +
                        50;

                    // 航路を決める
                    const auto flight_plan = DetermineFlightPlan<true>(
                        target_y, target_x, best_direction);

                    // TODO: 経路に障害物がないか検証

                    result.actions.insert(make_pair(
                        shipyard_id, ShipyardAction(ShipyardActionType::kLaunch,
                                                    n_ships, flight_plan)));
                }
            } break;
            }
        }

        // TODO

        (void)mean_value;

        return result;
    }

    int DetermineNShips(const int min_n_ships, const int max_n_ships,
                        nn::TensorSlice<float, 32>&& t) const {
        const auto min_quantized_n_ships =
            kNShipsQuantizationTable[min_n_ships];
        const auto max_quantized_n_ships =
            max_n_ships < (int)kNShipsQuantizationTable.size()
                ? kNShipsQuantizationTable[max_n_ships]
                : kNShipsQuantizationTable.back() + 1;

        for (auto i = 0; i < 32; i++)
            if (i < min_quantized_n_ships || max_quantized_n_ships < i)
                t[i] = 1e-30f;
        auto quantized_n_ships = -100;
        nn::F::Argmax(t, quantized_n_ships);
        const auto n_ships = uniform_int_distribution<>(
            max((int)kDequantizationTable[quantized_n_ships], min_n_ships),
            min((int)kDequantizationTable[quantized_n_ships + 1],
                (int)max_n_ships))(rng);
        return n_ships;
    }

    // 行って戻ってくる
    template <bool convert_type = false>
    string DetermineFlightPlan(int target_y, int target_x,
                               int first_direction) const {
        auto flight_plan = string();
        auto flight_plan_last = 'a';
        auto remaining_y = target_y;
        auto remaining_x = target_x;
        switch (first_direction) {
        case 0:
            flight_plan += 'N';
            if (target_y < -1)
                flight_plan += to_string(-1 - target_y);
            remaining_y = 0;
            flight_plan_last = 'S';
            break;
        case 1:
            flight_plan += 'E';
            if (target_x > 1)
                flight_plan += to_string(target_x - 1);
            remaining_x = 0;
            flight_plan_last = 'W';
            break;
        case 2:
            flight_plan += 'S';
            if (target_y > 1)
                flight_plan += to_string(target_y - 1);
            remaining_y = 0;
            flight_plan_last = 'N';
            break;
        case 3:
            flight_plan += 'W';
            if (target_x < -1)
                flight_plan += to_string(-1 - target_x);
            remaining_x = 0;
            flight_plan_last = 'E';
            break;
        }
        if (remaining_y < 0) {
            flight_plan += 'N';
            if (remaining_y < -1)
                flight_plan += to_string(-1 - remaining_y);
            if (!convert_type) {
                flight_plan += 'S';
                flight_plan += flight_plan[flight_plan.size() - 2];
            }
        } else if (remaining_x > 0) {
            flight_plan += 'E';
            if (remaining_x > 1)
                flight_plan += to_string(remaining_x - 1);
            if (!convert_type) {
                flight_plan += 'W';
                flight_plan += flight_plan[flight_plan.size() - 2];
            }
        } else if (remaining_y > 0) {
            flight_plan += 'S';
            if (remaining_y > 1)
                flight_plan += to_string(remaining_y - 1);
            if (!convert_type) {
                flight_plan += 'N';
                flight_plan += flight_plan[flight_plan.size() - 2];
            }
        } else if (remaining_x < 0) {
            flight_plan += 'W';
            if (remaining_x < -1)
                flight_plan += to_string(-1 - remaining_x);
            if (!convert_type) {
                flight_plan += 'E';
                flight_plan += flight_plan[flight_plan.size() - 2];
            }
        }
        if (convert_type)
            flight_plan += 'C';
        else
            flight_plan += flight_plan_last;
        return flight_plan;
    }

    auto SampleAction() {
        // TODO
    }
};

struct MCTSShipyardAction {
    // ハッシュが同じ行動なら探索を深める
    // 基本的に艦数をまとめるくらいで良い？
    // 子の数にも制限を設ける

    // convert の位置とかもまとめたい感じはある

    unsigned long long hash;
};

struct MCTSAction {
    unsigned long long hash;
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

#include <fstream>

static void TestPrediction(const string kif_filename,
                           const string parameter_filename) {
    static auto shipyard_feature_tensor = NNUE::ShipyardFeatureTensor<true>();
    static auto global_feature_tensor = NNUE::GlobalFeatureTensor<true>();
    static auto value_tensor = NNUE::OutValueTensor<true>();
    static auto type_tensor = NNUE::OutActionTypeTensor<true>();
    static auto code_tensor = NNUE::OutCodeTensor<true>();

    static auto spawn_n_ships_tensor = SpawnDecoder::NShipsTensor<true>();
    static auto move_n_ships_tensor = MoveDecoder::NShipsTensor<true>();
    static auto move_relative_position_tensor =
        MoveDecoder::RelativePositionTensor<true>();
    static auto move_n_steps_tensor = MoveDecoder::NStepsTensor<true>();
    static auto attack_n_ships_tensor = AttackDecoder::NShipsTensor<true>();
    static auto attack_relative_position_tensor =
        AttackDecoder::RelativePositionTensor<true>();
    static auto attack_direction_tensor =
        AttackDecoder::DirectionTensor<true>();
    static auto convert_n_ships_tensor = ConvertDecoder::NShipsTensor<true>();
    static auto convert_relative_position_tensor =
        ConvertDecoder::RelativePositionTensor<true>();
    static auto convert_direction_tensor =
        ConvertDecoder::DirectionTensor<true>();
    spawn_n_ships_tensor.Fill_(-1e30f);
    move_n_ships_tensor.Fill_(-1e30f);
    move_relative_position_tensor.Fill_(-1e30f);
    move_n_steps_tensor.Fill_(-1e30f);
    attack_n_ships_tensor.Fill_(-1e30f);
    attack_relative_position_tensor.Fill_(-1e30f);
    attack_direction_tensor.Fill_(-1e30f);
    convert_n_ships_tensor.Fill_(-1e30f);
    convert_relative_position_tensor.Fill_(-1e30f);
    convert_direction_tensor.Fill_(-1e30f);

    static auto agent = NNUEGreedyAgent(parameter_filename);

    // ==============================
    auto is = ifstream(kif_filename);
    if (!is) {
        throw runtime_error(string("ファイル ") + kif_filename +
                            string("を開けなかったよ"));
    }
    KifHeader().Read(is);

    // 罫線を読み捨てる
    string line;
    is >> line; // "==="

    // 0 ターン目の行動を読み捨てる
    int zero0, zero1;
    is >> zero0 >> zero1;

    struct NNUEData {
        vector<int> shipyard_feature_;
        array<float, NNUEFeature::kNGlobalFeatures> global_feature_;
        ActionTarget target_;
        PlayerId player_id_;
    };

    auto data = vector<NNUEData>();

    while (true) {
        // 状態を読み取る
        const auto state = State().Read(is);

        is >> line; // "---"
        if (line[0] == '=')
            break;

        // 行動を読み取る
        const auto action = Action().Read(state.shipyard_id_mapper_, is);

        // 接戦なら特徴抽出
        const auto approx_scores = state.ComputeApproxScore();
        if (state.step_ < 100 ||
            max(approx_scores[0], approx_scores[1]) <
                3.0 * min(approx_scores[0], approx_scores[1])) {

            const auto features = NNUEFeature(state);

            const auto [batch_size, shipyard_ids] = agent.FeatureToTensor(
                features, shipyard_feature_tensor, global_feature_tensor);
            assert(batch_size == (int)state.shipyards_.size());
            assert(batch_size == (int)(state.players_[0].shipyard_ids_.size() +
                                       state.players_[1].shipyard_ids_.size()));

            agent.nnue.Forward(batch_size, shipyard_feature_tensor,
                               global_feature_tensor, value_tensor, type_tensor,
                               code_tensor);
            agent.spawn_decoder.Forward(batch_size, code_tensor,
                                        spawn_n_ships_tensor);
            agent.move_decoder.Forward(
                batch_size, code_tensor, move_n_ships_tensor,
                move_relative_position_tensor, move_n_steps_tensor);
            agent.attack_decoder.Forward(
                batch_size, code_tensor, attack_n_ships_tensor,
                attack_relative_position_tensor, attack_direction_tensor);
            agent.convert_decoder.Forward(
                batch_size, code_tensor, convert_n_ships_tensor,
                convert_relative_position_tensor, convert_direction_tensor);

            cout << kTextBold << "--- State ---" << kResetTextStyle << endl;
            state.Print();
            cout << kTextBold << "--- Features ---" << kResetTextStyle << endl;
            cout << "-- global features --" << endl;
            for (auto i = 0; i < 2; i++) {
                for (const auto v : features.global_features[i])
                    cout << v << " ";
                cout << endl;
            }
            cout << "-- shipyard features --" << endl;
            for (auto i = 0; i < 2; i++) {
                for (const auto& [shipyard_id, feats] :
                     features.shipyard_features[i]) {
                    for (const auto v : feats)
                        cout << v << " ";
                    cout << endl;
                }
            }
            cout << kTextBold << "--- Predicted actions ---" << kResetTextStyle
                 << endl;
            const auto predicted_actions = agent.ComputeNextMove(state);
            for (const auto& [shipyard_id, shipyard_action] :
                 predicted_actions.actions) {
                cout << shipyard_id << ": " << shipyard_action.Str() << endl;
            }

            cout << kTextBold << "--- Predictions ---" << kResetTextStyle
                 << endl;
            cout << "-- value --" << endl;
            for (auto b = 0; b < batch_size; b++)
                cout << value_tensor[b] << " ";
            cout << endl;

            cout << "-- type --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                for (const auto& v : type_tensor[b])
                    cout << v << " ";
                cout << endl;
            }
            cout << endl;

            const auto get_top_n = [](const auto& tensor, const int n) {
                auto result = vector<int>(tensor.size());
                iota(result.begin(), result.end(), 0);
                partial_sort(result.begin(),
                             result.begin() + min((int)result.size(), n),
                             result.end(), [&](const int l, const int r) {
                                 return tensor[l] > tensor[r];
                             });
                result.resize(min((int)result.size(), n));
                return result;
            };

            const auto top_n = 5;
            cout << "-- spawn n_ships --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(spawn_n_ships_tensor[b], top_n);
                for (const auto i : top_indices) {
                    cout << i << "(" << spawn_n_ships_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- move n_ships --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(move_n_ships_tensor[b], top_n);
                for (const auto i : top_indices) {
                    const auto l = kDequantizationTable[i];
                    const auto r = kDequantizationTable[i + 1];
                    cout << "[" << l << "," << r - 1 << "]("
                         << move_n_ships_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- move relative_position --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(move_relative_position_tensor[b], top_n);
                for (const auto i : top_indices) {
                    const auto [y, x] = TranslatePosition221ToYX(i);
                    cout << y << "," << x << "("
                         << move_relative_position_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- move n_steps --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(move_n_ships_tensor[b], top_n);
                for (const auto i : top_indices) {
                    cout << i << "(" << move_n_ships_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- attack n_ships --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(attack_n_ships_tensor[b], top_n);
                for (const auto i : top_indices) {
                    const auto l = kDequantizationTable[i];
                    const auto r = kDequantizationTable[i + 1];
                    cout << "[" << l << "," << r - 1 << "]("
                         << attack_n_ships_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- attack relative_position --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(attack_relative_position_tensor[b], top_n);
                for (const auto i : top_indices) {
                    const auto [y, x] = TranslatePosition221ToYX(i);
                    cout << y << "," << x << "("
                         << attack_relative_position_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- attack direction --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(attack_direction_tensor[b], top_n);
                for (const auto i : top_indices) {
                    cout << "NESW"[i] << "(" << attack_direction_tensor[b][i]
                         << ") ";
                }
                cout << endl;
            }
            cout << "-- convert n_ships --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(convert_n_ships_tensor[b], top_n);
                for (const auto i : top_indices) {
                    const auto l = kDequantizationTable[i];
                    const auto r = kDequantizationTable[i + 1];
                    cout << "[" << 50 + l << "," << 50 + r - 1 << "]("
                         << convert_n_ships_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- convert relative_position --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(convert_relative_position_tensor[b], top_n);
                for (const auto i : top_indices) {
                    const auto [y, x] = TranslatePosition221ToYX(i);
                    cout << y << "," << x << "("
                         << convert_relative_position_tensor[b][i] << ") ";
                }
                cout << endl;
            }
            cout << "-- convert direction --" << endl;
            for (auto b = 0; b < batch_size; b++) {
                const auto p = state.shipyards_.at(shipyard_ids[b]).position_;
                cout << shipyard_ids[b] << " (" << (int)p.y << "," << (int)p.x
                     << ")" << endl;
                const auto top_indices =
                    get_top_n(convert_direction_tensor[b], top_n);
                for (const auto i : top_indices) {
                    cout << "NESW"[i] << "(" << convert_direction_tensor[b][i]
                         << ") ";
                }
                cout << endl;
            }
            cout << endl;

            if (state.step_ >= 115)
                return;

            for (const auto& [shipyard_id, shipyard_action] : action.actions) {
                const auto action_target =
                    ActionTarget(state, shipyard_id, shipyard_action);
                if (action_target.action_target_type == ActionTargetType::kNull)
                    continue;
                const auto shipyard = state.shipyards_.at(shipyard_id);

                data.push_back(
                    {features.shipyard_features[shipyard.player_id_].at(
                         shipyard_id),
                     features.global_features[shipyard.player_id_],
                     action_target, shipyard.player_id_});
            }
        }
    }
}

int main() {
    //
    TestPrediction("36385265.kif", "parameters.bin");
}
// clang-format off
// clang++ nnue.cpp -std=c++17 -Wall -Wextra -march=native -l:libblas.so.3 -fsanitize=address -g
// clang-format on
