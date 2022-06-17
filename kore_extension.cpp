// 最終的にやることは、state から特徴量の numpy array への変換

// from https://qiita.com/termoshtt/items/81eeb0467d9087958f7f
#include <boost/python/numpy.hpp>

#include <algorithm>
#include <boost/python/numpy/dtype.hpp>
#include <boost/python/numpy/ndarray.hpp>
#include <boost/python/tuple.hpp>
#include <fstream>
#include <stdexcept>
#include <tuple>

#include "kore_fleets.cpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

// ファイル名を受け取って、特徴と勝敗を返す
auto MakeFeature(const string& filename) {
    // 接戦なとこだけ返す

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
    static auto local_features =
        nn::TensorBuffer<float, 800, Feature::kNLocalFeatures, kSize, kSize>();
    static auto global_features =
        nn::TensorBuffer<float, 800, Feature::kNGlobalFeatures>();
    static auto targets = array<bool, 800>();

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
            for (PlayerId player_id = 0; player_id < 2; player_id++) {
                const auto feature = Feature(state, 0);
                local_features[n_data] = feature.local_features;
                global_features[n_data] = feature.global_features;
                n_data++;
            }
        }
    }

    // 結果の読み取り
    is >> line; // "-1"
    double reward0, reward1;
    is >> reward0 >> reward1;
    for (auto i = 0; i < n_data; i += 2) {
        if (reward0 > reward1) {
            targets[i] = true;
            targets[i + 1] = false;
        } else {
            targets[i] = false;
            targets[i + 1] = true;
        }
    }

    // TODO: Boost との接続
    auto np_local_features =
        np::empty(p::make_tuple(n_data, Feature::kNLocalFeatures, kSize, kSize),
                  np::dtype::get_builtin<float>());
    auto np_global_features =
        np::empty(p::make_tuple(n_data, Feature::kNGlobalFeatures),
                  np::dtype::get_builtin<float>());
    auto np_targets =
        np::empty(p::make_tuple(n_data), np::dtype::get_builtin<bool>());
    memcpy(np_local_features.get_data(), local_features.Data(),
           sizeof(float) * Feature::kNLocalFeatures * kSize * kSize * n_data);
    memcpy(np_global_features.get_data(), global_features.Data(),
           sizeof(float) * Feature::kNGlobalFeatures * n_data);
    memcpy(np_targets.get_data(), targets.data(), sizeof(bool) * n_data);
    return p::make_tuple(np_local_features, np_global_features, np_targets);
}

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
}
