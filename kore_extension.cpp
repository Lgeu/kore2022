#include <boost/python/numpy.hpp>

#include <fstream>
#include <stdexcept>

#include "kore_fleets.cpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

template <typename T, typename... S> auto NpEmpty(const S... shape) {
    return np::empty(p::make_tuple(shape...), np::dtype::get_builtin<T>());
}

auto MakeNNUEFeature(const string& filename) {
    // TODO: フォーマットを考える
    constexpr auto kMaxNShipyardFeatures = 512;
    // shipyard_feature: int[many, kMaxNShipyardFeatures]
    // global_feature: float[many, kNGlobalFeatures]

    // target_values: bool[many]
    // target_action_types: signed char[many]
    // target_action_n_ships: short[many]
    // target_action_relative_position: short[many]
    // target_action_n_steps: signed char[many]
    // target_action_direction: signed char[many]

    // PyTorch との相性も良さそう

    // Action は null かもしれないことに留意

    auto is = ifstream(filename);
    if (!is) {
        throw runtime_error(string("ファイル ") + filename +
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

    // 結果の読み取り
    is >> line; // "-1"
    double reward0, reward1;
    is >> reward0 >> reward1;
    const auto winner = (PlayerId)(reward1 > reward0);

    // numpy のデータつくる
    const auto n_data = (int)data.size();
    const auto np_shipyard_features =
        NpEmpty<int>(n_data, kMaxNShipyardFeatures);
    auto np_global_features =
        NpEmpty<float>(n_data, NNUEFeature::kNGlobalFeatures);
    auto np_target_values = NpEmpty<bool>(n_data);
    auto np_target_action_types = NpEmpty<signed char>(n_data);
    auto np_target_action_n_ships = NpEmpty<short>(n_data);
    auto np_target_action_relative_position = NpEmpty<short>(n_data);
    auto np_target_action_n_steps = NpEmpty<signed char>(n_data);
    auto np_target_action_direction = NpEmpty<signed char>(n_data);

    for (auto idx_data = 0; idx_data < n_data; idx_data++) {
        const auto& d = data[idx_data];
        for (auto i = 0; i < kMaxNShipyardFeatures; i++) {
            ((int*)np_shipyard_features
                 .get_data())[idx_data * kMaxNShipyardFeatures + i] =
                i < (int)d.shipyard_feature_.size() ? d.shipyard_feature_[i]
                                                    : -100;
        }
        for (auto i = 0; i < NNUEFeature::kNGlobalFeatures; i++) {
            ((float*)np_shipyard_features
                 .get_data())[idx_data * NNUEFeature::kNGlobalFeatures + i] =
                d.global_feature_[i];
        }
        const auto& a = d.target_;
        ((bool*)np_target_values.get_data())[idx_data] = d.player_id_ == winner;
        ((signed char*)np_target_action_types.get_data())[idx_data] =
            (signed char)a.action_target_type;
        ((short*)np_target_action_n_ships.get_data())[idx_data] =
            (signed char)a.NShips();
        ((signed char*)
             np_target_action_relative_position.get_data())[idx_data] =
            a.action_target_type == ActionTargetType::kSpawn
                ? -100
                : RelativeAll(a.RelativePosition());

        ((signed char*)np_target_action_n_steps.get_data())[idx_data] =
            a.action_target_type == ActionTargetType::kMove ? a.NSteps() : -100;

        ((signed char*)np_target_action_direction.get_data())[idx_data] =
            a.action_target_type == ActionTargetType::kAttack ||
                    a.action_target_type == ActionTargetType::kConvert
                ? (int)a.InitialMove()
                : -100;
    }

    // 行動なしをかんがえてなかった

    // const auto shipyard_features = nn::TensorSlice<typename T, int dims>

    // auto np_shipyard_features =
    //     np::empty(p::make_tuple(n_data, kMaxNShipyardFeatures),
    //     np::dtype::get_builtin<int>());
    // auto np_global_features =
    //     np::empty(p::make_tuple(n_data, NNUEFeature::kNGlobalFeatures),
    //               np::dtype::get_builtin<float>());
    // auto np_target_values =
    //     np::empty(p::make_tuple(n_data), np::dtype::get_builtin<bool>());
    // auto np_action_types =
    //     np::empty(p::make_tuple(n_data), np::dtype::get_builtin<signed
    //     char>());
    // auto np_target_action_n_ships =
    //     np::empty(p::make_tuple(n_data), np::dtype::get_builtin<short>());
    // auto np_target_action_relative_position =
    //     np::empty(p::make_tuple(n_data), np::dtype::get_builtin<short>());
    // auto np_target_action_n_steps =
    //     np::empty(p::make_tuple(n_data), np::dtype::get_builtin<signed
    //     char>());
    // auto np_target_action_direction =
    //     np::empty(p::make_tuple(n_data), np::dtype::get_builtin<signed
    //     char>());

    return p::make_tuple(np_shipyard_features, np_global_features,
                         np_target_values, np_target_action_types,
                         np_target_action_n_ships,
                         np_target_action_relative_position,
                         np_target_action_n_steps, np_target_action_direction);
}

// ファイル名を受け取って、特徴と勝敗を返す
auto MakeFeature(const string& filename, np::ndarray& out_local_features,
                 np::ndarray& out_global_features, np::ndarray& out_targets) {
    // 接戦なとこだけ返す

    if (out_local_features.get_nd() != 4 ||
        out_local_features.shape(0) != 400 ||
        out_local_features.shape(1) != Feature::kNLocalFeatures ||
        out_local_features.shape(2) != kSize ||
        out_local_features.shape(3) != kSize) {
        throw runtime_error(string("local_features の shape が (400, ") +
                            to_string(Feature::kNLocalFeatures) + string(", ") +
                            to_string(kSize) + string(", ") + to_string(kSize) +
                            string(") じゃないよ"));
    }
    if (out_local_features.get_dtype() != np::dtype::get_builtin<float>()) {
        throw runtime_error("local_features の dtype が float32 じゃないよ");
    }
    if (out_global_features.get_nd() != 2 ||
        out_global_features.shape(0) != 400 ||
        out_global_features.shape(1) != Feature::kNGlobalFeatures) {
        throw runtime_error(string("global_features の shape が (400, ") +
                            to_string(Feature::kNGlobalFeatures) +
                            string(") じゃないよ"));
    }
    if (out_global_features.get_dtype() != np::dtype::get_builtin<float>()) {
        throw runtime_error("global_features の dtype が float32 じゃないよ");
    }
    if (out_targets.get_nd() != 1 || out_targets.shape(0) != 400) {
        throw runtime_error(string("targets の shape が (400,) じゃないよ"));
    }
    if (out_targets.get_dtype() != np::dtype::get_builtin<bool>()) {
        throw runtime_error("targets の dtype が bool じゃないよ");
    }

    auto is = ifstream(filename);
    if (!is) {
        throw runtime_error(string("ファイル ") + filename +
                            string("を開けなかったよ"));
    }
    KifHeader().Read(is);

    // 罫線を読み捨てる
    string line;
    is >> line; // "==="

    // 0 ターン目の行動を読み捨てる
    int zero0, zero1;
    is >> zero0 >> zero1;

    // 最初の状態を読み取る
    State state;
    state.Read(is);

    auto n_data = 0;
    auto local_features =
        nn::TensorSlice<float, 800, Feature::kNLocalFeatures, kSize, kSize>(
            (float*)out_local_features.get_data());
    auto global_features =
        nn::TensorSlice<float, 800, Feature::kNGlobalFeatures>(
            (float*)out_global_features.get_data());
    auto targets = (bool*)out_targets.get_data();

    while (true) {
        is >> line; // "---"
        if (line[0] == '=')
            break;

        // action を読み飛ばす
        Action().Read(state.shipyard_id_mapper_, is);

        // state を読み取る
        state = State().Read(is);

        // 接戦なら feature 抽出
        const auto approx_scores = state.ComputeApproxScore();
        if (state.step_ < 100 ||
            max(approx_scores[0], approx_scores[1]) <
                3.0 * min(approx_scores[0], approx_scores[1])) {
            const auto player_id = (PlayerId)0;
            const auto feature = Feature(state, player_id);
            local_features[n_data] = feature.local_features;
            global_features[n_data] = feature.global_features;
            n_data++;
        }
    }

    // 結果の読み取り
    is >> line; // "-1"
    double reward0, reward1;
    is >> reward0 >> reward1;
    for (auto i = 0; i < n_data; i++) {
        targets[i] = reward0 > reward1;
    }

    return n_data;
}

// from https://qiita.com/termoshtt/items/81eeb0467d9087958f7f
/*
// 2倍にする
void mult_two(np::ndarray a) {
    int nd = a.get_nd();
    if (nd != 1)
        throw std::runtime_error("a must be 1-dimensional");
    size_t N = a.shape(0);
    if (a.get_dtype() != np::dtype::get_builtin<double>())
        throw std::runtime_error("a must be float64 array");
    double* p = reinterpret_cast<double*>(a.get_data());
    std::transform(p, p + N, p, [](double x) { return 2 * x; });
}

BOOST_PYTHON_MODULE(mymod1) {
    Py_Initialize();
    np::initialize();
    p::def("mult_two", mult_two);
}
*/

BOOST_PYTHON_MODULE(kore_extension) {
    Py_Initialize();
    np::initialize();
    p::def("make_feature", MakeFeature);
    p::def("make_nnue_feature", MakeNNUEFeature);
}

#ifdef TEST_KORE_EXTENSION
int main() {
    Py_Initialize();
    np::initialize();

    auto np_local_features =
        np::empty(p::make_tuple(400, Feature::kNLocalFeatures, kSize, kSize),
                  np::dtype::get_builtin<float>());
    auto np_global_features =
        np::empty(p::make_tuple(400, Feature::kNGlobalFeatures),
                  np::dtype::get_builtin<float>());
    auto np_targets =
        np::empty(p::make_tuple(400), np::dtype::get_builtin<bool>());

    for (auto lap = 0; lap < 10; lap++) {
        cout << "lap " << lap << endl;
        MakeFeature("36385265.kif", np_local_features, np_global_features,
                    np_targets);
    }
}
// clang-format off
// clang++ -std=c++17 -Wall -Wextra -O3 -DTEST_KORE_EXTENSION -fPIC kore_extension.cpp -I/home/user/anaconda3/include/python3.8 /usr/local/lib/libboost_numpy38.a /usr/local/lib/libboost_python38.a -lpython3.8 -L/home/user/anaconda3/lib -g
// clang-format on
#endif

#ifdef TEST_MAKE_NNUE_FEATURE
int main() {
    Py_Initialize();
    np::initialize();

    // {
    //     const auto arr = NpEmpty<float>(10);
    //     ((float*)arr.get_data())[0] = 1.0f;
    //     const auto t = make_tuple(arr);
    //     const auto extracted = (np::ndarray)p::extract<np::ndarray>(t[0]);
    //     cout << ((float*)extracted.get_data())[0] << endl;
    // }

    const auto t = MakeNNUEFeature("36385265.kif");
    const auto np_shipyard_features =
        (np::ndarray)p::extract<np::ndarray>(t[0]);

    const auto n_data = (int)np_shipyard_features.shape(0);
    cout << "n_data=" << n_data << endl;
    const auto max_n_shipyard_features = (int)np_shipyard_features.shape(1);
    for (auto i = 0; i < n_data; i++) {
        for (auto f = 0; f < max_n_shipyard_features; f++) {
            const auto feature =
                ((int*)np_shipyard_features
                     .get_data())[i * max_n_shipyard_features + f];
            if (feature == -100)
                break;
            cout << feature << " ";
        }
        cout << endl;
    }
}
// clang-format off
// clang++ -std=c++17 -Wall -Wextra -O3 -DTEST_MAKE_NNUE_FEATURE -fPIC kore_extension.cpp -I/home/user/anaconda3/include/python3.8 /usr/local/lib/libboost_numpy38.a /usr/local/lib/libboost_python38.a -lpython3.8 -L/home/user/anaconda3/lib -g
// clang-format on
#endif