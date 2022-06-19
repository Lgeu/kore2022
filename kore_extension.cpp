// 最終的にやることは、state から特徴量の numpy array への変換

// from https://qiita.com/termoshtt/items/81eeb0467d9087958f7f
#include <boost/python/numpy.hpp>

#include <fstream>
#include <stdexcept>

#include "kore_fleets.cpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

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
