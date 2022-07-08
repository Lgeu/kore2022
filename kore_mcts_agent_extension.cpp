#include <boost/python/numpy.hpp>

#include "environment.cpp"
#include "nnue.cpp"

namespace p = boost::python;
namespace np = boost::python::numpy;

NNUEMCTSAgent agent;

void LoadParameters(const string& parameters_filename) {
    agent.ReadParameters(parameters_filename);
}

auto ComputeNextMove(const string& state_string) {
    auto is = istringstream(state_string);
    const auto state = State().Read(is);
    const auto action = agent.ComputeNextMove(state);
    auto os = ostringstream();
    action.Write(state.shipyard_id_mapper_, os);
    return string(os.str());
}

BOOST_PYTHON_MODULE(kore_mcts_agent_extension) {
    Py_Initialize();
    np::initialize();
    p::def("load_parameters", LoadParameters);
    p::def("compute_next_move", ComputeNextMove);
}

// clang-format off
// clang++ -std=c++17 -Wall -Wextra -O3 --shared -fPIC kore_mcts_agent_extension.cpp -o kore_mcts_agent_extension.so -I/home/user/anaconda3/include/python3.8 /usr/local/lib/libboost_numpy38.a /usr/local/lib/libboost_python38.a -lpython3.8 -L/home/user/anaconda3/lib -march=broadwell -l:libblas.so.3
// clang-format on
