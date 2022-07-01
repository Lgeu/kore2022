#include "environment.cpp"

// 対戦するときのインタフェース、チェックするときの、次の一手するときの
struct Game {
    State state;
    vector<Agent*> agents;

    // 状態を読み込む
    void ReadState(istream& is) { state.Read(is); }

    // 行動を読み込んで次の状態に遷移する
    void ReadAction(istream& is) {
        state = state.Next(Action().Read(state.shipyard_id_mapper_, is));
    }

    // 棋譜の行動によって次の状態に正しく遷移できるか確認
    void ValidateKif(istream& is) {
        // ヘッダを読み捨てる
        KifHeader().Read(is);

        // 罫線を読み捨てる
        string line;
        is >> line; // "==="

        // 0 ターン目の行動を読み捨てる
        int zero0, zero1;
        is >> zero0 >> zero1;

        // 最初の状態を読み取る
        state.Read(is);

        while (true) {
            is >> line; // "---"
            if (line[0] == '=')
                break;
            auto action = Action().Read(state.shipyard_id_mapper_, is);
            state = state.Next(action);
            if (state.step_ % 10 == 1)
                state.Print();
            const auto input_state = State().Read(is);
            assert(state.Same(input_state));
            state = input_state;
        }
    }

    // 対戦する
    void Match() {
        while (true) {
            state.Read(cin);
            auto action = agents[0]->ComputeNextMove(state, 0);
            action.Write(state.shipyard_id_mapper_, cout);
        }
    }
};