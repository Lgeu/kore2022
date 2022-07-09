#pragma once
#include "../marathon/library.hpp"
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <istream>
#include <map>
#include <ostream>
#include <random>
#include <set>
#include <string>

#ifdef NDEBUG
static constexpr auto kDebug = false;
#else
static constexpr auto kDebug = true;
#endif

static constexpr auto kSize = 21;
static constexpr auto kSpawnCost = 10.0;
static constexpr auto kConvertCost = 50;
static constexpr auto kRegenRate = 0.02;
static constexpr auto kMaxRegenCellKore = 500;
static constexpr auto kAgentTimeout = 60.0;
static constexpr auto kActTimeout = 3.0;
static constexpr auto kRunTimeout = 9600.0;
static constexpr auto kSpawnValues =
    array<int, 9>{2, 7, 17, 34, 60, 97, 147, 212, 294};

auto rng = mt19937(42);

using Point = Vec2<signed char>;
using FleetId = short;
using ShipyardId = short;
using PlayerId = short;

struct Kore90Percentiles {
    array<double, 400> data;
    Kore90Percentiles() {
        for (auto step = 0; step < 400; step++) {
            auto result = 0.0;
            auto step_pow = 1.0;
            for (const auto coef :
                 {44.263940407419646, 0.8352519775398193, -0.032542548363244446,
                  0.0005796380595064503, -3.0853302781649527e-06,
                  6.548497742662611e-09, -4.905086438656963e-12}) {
                result += step_pow * coef;
                step_pow *= step;
            }
            data[step] = result;
        }
    }
    const double& operator[](const int i) { return data[i]; }
} kore_90_percentiles;

#include "n_ships_quantization_table.cpp"

template <> struct std::hash<Point> {
    size_t operator()(const Point& key) const { return key.y * 256 + key.x; }
};

enum struct Direction { N, E, S, W };
enum struct ShipyardActionType { kSpawn, kLaunch };

const auto kTextColors =
    array<string, 8>{"\x1B[30m", "\x1B[31m", "\x1B[32m", "\x1B[33m",
                     "\x1B[34m", "\x1B[35m", "\x1B[36m", "\x1B[37m"};
constexpr auto kTextBold = "\x1B[1m";
constexpr auto kTextDim = "\x1B[2m";
constexpr auto kResetTextStyle = "\x1B[0m";

static auto IsClose(const double a, const double b) {
    return abs(a - b) <= max(abs(a), abs(b)) * 1e-9;
}

// cell の kore が x.x25 か x.x75 のとき、1.02 倍するとちょうど x.xxx5 になって
// round がやばい
// Python の round は正確にやるために一旦文字列に起こしてるっぽい
// https://github.com/python/cpython/blob/3.7/Objects/floatobject.c
static auto Round3(const double a) {
    assert(a >= 0.0);
    if ((unsigned)(a * 1e5 + 0.5) % 100 != 50) {
        return round(a * 1e3) / 1e3;
    } else {
        const auto y = (long double)a * 1e3;
        auto z = round(y);
        if (abs(y - z) == 0.5)
            z = 2.0 * round(y / 2.0);
        return (double)(z * 1e-3l);
    }
}

static auto CharToDirection(const char c) {
    switch (c) {
    case 'N':
        return Direction::N;
    case 'E':
        return Direction::E;
    case 'S':
        return Direction::S;
    case 'W':
        return Direction::W;
    }
    cerr << "CharToDirection: " << c << " なんて方向は無いよ" << endl;
    abort();
}

// static Point GetColRow(const int pos) {
//     return {(Point::value_type)(pos / kSize), (Point::value_type)(pos %
//     kSize)};
// }

static Point GetToPos(const Point v, const Direction direction) {
    switch (direction) {
    case Direction::N:
        return v.y != 0 ? Point{(Point::value_type)(v.y - 1), v.x}
                        : Point{(Point::value_type)(kSize - 1), v.x};
    case Direction::S:
        return v.y != kSize - 1 ? Point{(Point::value_type)(v.y + 1), v.x}
                                : Point{0, v.x};
    case Direction::E:
        return v.x != kSize - 1 ? Point{v.y, (Point::value_type)(v.x + 1)}
                                : Point{v.y, 0};
    case Direction::W:
        return v.x != 0 ? Point{v.y, (Point::value_type)(v.x - 1)}
                        : Point{v.y, (Point::value_type)(kSize - 1)};
    default:
        assert(false);
    }
}

struct Cell {
    double kore_;
    ShipyardId shipyard_id_;
    FleetId fleet_id_;
    Cell() : kore_(), shipyard_id_(-1), fleet_id_(-1) {}
};

struct Fleet {
    FleetId id_;
    short ship_count_;
    Direction direction_;
    Point position_;
    double kore_; // int でええんか
    string flight_plan_;
    PlayerId player_id_;

    Fleet(const FleetId id, const short ship_count, const Direction direction,
          const Point position, const double kore, const string flight_plan,
          const PlayerId player_id)
        : id_(id), ship_count_(ship_count), direction_(direction),
          position_(position), kore_(kore), flight_plan_(flight_plan),
          player_id_(player_id) {}

    Fleet(const FleetId id, const PlayerId player_id, istream& is)
        : id_(id), player_id_(player_id) {
        Read(is);
    }

    double CollectionRate() const { return min(log(ship_count_) * 0.05, 0.99); }

    static auto MaxFlightPlanLenForShipCount(const int ship_count) {
        return 1 + (int)(2.0 * log(ship_count));
    }

    bool LessThanOtherAlliedFleet(const Fleet& other) const {
        if (ship_count_ != other.ship_count_)
            return ship_count_ < other.ship_count_;
        if (kore_ != other.kore_)
            return kore_ < other.kore_;
        return (int)direction_ > (int)other.direction_;
    }

    auto Same(const Fleet& rhs) const {
        return ship_count_ == rhs.ship_count_ && direction_ == rhs.direction_ &&
               position_ == rhs.position_ && IsClose(kore_, rhs.kore_) &&
               flight_plan_ == rhs.flight_plan_ && player_id_ == rhs.player_id_;
    }

    Fleet& Read(istream& is) {
        // id_ と player_id_ は設定されない
        auto position_raw = 0;
        auto direction_raw = 0;
        is >> position_raw >> kore_ >> ship_count_ >> direction_raw >>
            flight_plan_;
        position_ = Point{(Point::value_type)(position_raw / kSize),
                          (Point::value_type)(position_raw % kSize)};
        direction_ = (Direction)direction_raw;
        if (strcmp(flight_plan_.c_str(), "null") == 0)
            flight_plan_ = "";
        return *this;
    }
};

struct ShipyardAction {
    ShipyardActionType type_;
    short num_ships_;
    string flight_plan_;

    // コンストラクタ
    ShipyardAction(const ShipyardActionType type, const int num_ships,
                   const string flight_plan = "")
        : type_(type), num_ships_(num_ships), flight_plan_(flight_plan) {
        switch (type) {
        case ShipyardActionType::kLaunch: {
            assert(num_ships > 0);
            assert(flight_plan_.size() > 0);
            assert(string("NESW").find(flight_plan[0]) != string::npos);
            for (const auto c : flight_plan)
                assert(string("NESWC0123456789").find(c) != string::npos);
            const auto max_flight_plan_len =
                Fleet::MaxFlightPlanLenForShipCount(num_ships);
            if ((int)flight_plan.size() > max_flight_plan_len) {
                flight_plan_ = flight_plan.substr(0, max_flight_plan_len);
            }
            break;
        }
        case ShipyardActionType::kSpawn:
            break;
        }
    }
    ShipyardAction(string raw)
        : ShipyardAction(
              (assert(raw.size()), raw[0] == 'S') ? ShipyardActionType::kSpawn
                                                  : ShipyardActionType::kLaunch,
              raw[0] == 'S' ? stoi(raw.substr(6))
                            : stoi(raw.substr(7, raw.find('_', 8) - 7)),
              raw[0] == 'S' ? "" : raw.substr(raw.find('_', 8) + 1)) {
        // if (raw.size() == 0) {
        //     assert(false);
        // } else if (raw[0] == 'S') {
        //     // SPAWN
        //     ShipyardAction(ShipyardActionType::kSpawn, stoi(raw.substr(6)));
        // } else {
        //     // LAUNCH
        //     const auto idx_underscore = raw.find('_', 8);
        //     ShipyardAction(ShipyardActionType::kLaunch,
        //                    stoi(raw.substr(7, idx_underscore - 7)),
        //                    raw.substr(idx_underscore + 1));
        // }
    }

    // 文字列化
    string Str() const {
        switch (type_) {
        case ShipyardActionType::kSpawn:
            return string("SPAWN_") + to_string(num_ships_);
        case ShipyardActionType::kLaunch:
            return string("LAUNCH_") + to_string(num_ships_) + "_" +
                   flight_plan_;
        }
    }
};

struct Shipyard {
    ShipyardId id_;
    short ship_count_;
    Point position_;
    PlayerId player_id_;
    short turns_controlled_;

    // 以下は Read では無視する

    // 帰還/生産でそのターンに増えた艦数。新しくできたり乗っ取ったりした時は全部
    short last_ship_increment_;
    // 出撃/外部からの攻撃で減った艦数
    short last_ship_decrement_;

    Shipyard(const ShipyardId id, const int ship_count, const Point position,
             const PlayerId player_id, const int turns_controlled)
        : id_(id), ship_count_(ship_count), position_(position),
          player_id_(player_id), turns_controlled_(turns_controlled),
          last_ship_increment_(ship_count), last_ship_decrement_(0) {}

    Shipyard(const ShipyardId id, const PlayerId player_id, istream& is)
        : id_(id), player_id_(player_id), last_ship_decrement_(0) {
        Read(is);
        last_ship_increment_ = ship_count_;
    }

    int MaxSpawn() const {
        for (int i = 0; i < (int)kSpawnValues.size(); i++) {
            if (turns_controlled_ < kSpawnValues[i]) {
                return i + 1;
            }
        }
        return kSpawnValues.size() + 1;
    }

    auto Same(const Shipyard& rhs) const {
        return ship_count_ == rhs.ship_count_ && position_ == rhs.position_ &&
               player_id_ == rhs.player_id_ &&
               turns_controlled_ == rhs.turns_controlled_;
    }

    Shipyard& Read(istream& is) {
        // id_ と player_id_ は設定されない
        auto position_raw = 0;
        is >> position_raw >> ship_count_ >> turns_controlled_;
        position_ = Point{(Point::value_type)(position_raw / kSize),
                          (Point::value_type)(position_raw % kSize)};
        return *this;
    }
};

struct Player {
    PlayerId id_;
    double kore_;
    set<ShipyardId> shipyard_ids_;
    set<FleetId> fleet_ids_;
};

template <typename InternalId> struct IdMapper {
    unordered_map<string, InternalId> external_to_internal_;
    unordered_map<InternalId, string> internal_to_external_;

    void AddData(const string& external_id, const InternalId& internal_id) {
        external_to_internal_[external_id] = internal_id;
        internal_to_external_[internal_id] = external_id;
    }
    auto ExternalToInternal(const string& external_id) const {
        return external_to_internal_.at(
            external_id); // 外部の番号違うのあたりまえなんだよな、どうしよ
    }
    auto InternalToExternal(const InternalId& internal_id) const {
        return internal_to_external_.at(internal_id);
    }
};

// 2 プレイヤー分を同時に持つこともある
struct Action {
    map<ShipyardId, ShipyardAction> actions;

    auto& Read(const IdMapper<ShipyardId>& shipyard_id_mapper, istream& is) {
        assert(actions.size() == 0);
        for (auto player_id = 0; player_id < 2; player_id++) {
            int n_shipyards;
            is >> n_shipyards;
            for (auto i = 0; i < n_shipyards; i++) {
                string external_shipyard_id, action_raw;
                is >> external_shipyard_id >> action_raw;
                if (shipyard_id_mapper.external_to_internal_.find(
                        external_shipyard_id) ==
                    shipyard_id_mapper.external_to_internal_.end())
                    continue;
                const auto shipyard_id =
                    shipyard_id_mapper.ExternalToInternal(external_shipyard_id);
                const auto [it, emplaced] =
                    actions.try_emplace(shipyard_id, action_raw);
                assert(emplaced);
            }
        }
        return *this;
    }

    void Write(const IdMapper<ShipyardId>& shipyard_id_mapper,
               ostream& os) const {
        os << actions.size() << endl;
        for (const auto& [shipyard_id, action] : actions) {
            const auto external_shipyard_id =
                shipyard_id_mapper.InternalToExternal(shipyard_id);
            os << external_shipyard_id << " " << action.Str() << endl;
        }
    }

    void Merge(Action& rhs) { actions.merge(rhs.actions); }
};

enum struct FleetReportType : signed char {
    kArrived,
    kCollided,
    kMerged,
    kConverted,
};

struct FleetReport {
    FleetReportType type_;
    Point position_;
    bool deleted;
};

struct State {
    int step_;
    array<Player, 2> players_;
    map<FleetId, Fleet> fleets_;
    map<ShipyardId, Shipyard> shipyards_;
    map<FleetId, FleetReport> fleet_reports_;
    Board<Cell, kSize, kSize> board_;

    IdMapper<ShipyardId> shipyard_id_mapper_;
    IdMapper<FleetId> fleet_id_mapper_;
    FleetId next_fleet_id_;
    ShipyardId next_shipyard_id_;

    auto CountShips() const {
        auto counts = array<short, 2>();
        for (const auto& [_, shipyard] : shipyards_)
            counts[shipyard.player_id_] += shipyard.ship_count_;
        for (const auto& [_, fleet] : fleets_)
            counts[fleet.player_id_] += fleet.ship_count_;
        return counts;
    }

    auto CountCargo() const {
        auto counts = array<double, 2>();
        for (const auto& [_, fleet] : fleets_)
            counts[fleet.player_id_] += fleet.kore_;
        return counts;
    }

    auto ComputeApproxScore() const {
        auto result = array<double, 2>();
        result[0] += players_[0].kore_;
        result[1] += players_[1].kore_;
        const auto cargo_counts = CountCargo();
        result[0] += cargo_counts[0];
        result[1] += cargo_counts[1];
        const auto ship_counts = CountShips();
        result[0] += kSpawnCost * ship_counts[0];
        result[1] += kSpawnCost * ship_counts[1];
        result[0] +=
            kConvertCost * kSpawnCost * players_[0].shipyard_ids_.size();
        result[1] +=
            kConvertCost * kSpawnCost * players_[1].shipyard_ids_.size();
        return result;
    }

    void Initialize() {}

    auto& Read(istream& is) {
        // 初期状態を仮定
        is >> step_;
        for (auto y = 0; y < kSize; y++) {
            for (auto x = 0; x < kSize; x++) {
                is >> board_[{y, x}].kore_;
            }
        }
        for (auto player_id = 0; player_id < 2; player_id++) {
            auto& player = players_[player_id];
            double remaining_time; // 使わない？
            is >> remaining_time;
            is >> player.kore_;
            auto n_shipyards = 0;
            is >> n_shipyards;
            for (auto i = 0; i < n_shipyards; i++) {
                string external_shipyard_id;
                is >> external_shipyard_id;
                const auto shipyard_id = next_shipyard_id_++;
                shipyard_id_mapper_.AddData(external_shipyard_id, shipyard_id);
                AddShipyard(Shipyard(shipyard_id, player_id, is));
            }
            auto n_fleets = 0;
            is >> n_fleets;
            for (auto i = 0; i < n_fleets; i++) {
                string external_fleet_id;
                is >> external_fleet_id;
                const auto fleet_id = next_fleet_id_++;
                fleet_id_mapper_.AddData(external_fleet_id, fleet_id);
                AddFleet(Fleet(fleet_id, player_id, is));
            }
        }
        return *this;
    }

    void Print() const {
        auto initial_precision = cout.precision();
        cout << fixed << setprecision(1);
        const auto player_colors = array<int, 2>{5, 6};
        cout << kResetTextStyle;
        cout << "step: " << step_ << endl;
        cout << "player    / kore      / cargo     / ships     / shipyards / "
                "max spawn / kore+cargo+ships*10+shipyards*500"
             << endl;
        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            const auto& player = players_[player_id];
            const auto kore = player.kore_;
            auto cargo = 0.0;
            auto ships = 0;
            auto max_spawn = 0;
            auto n_shipyards = (int)player.shipyard_ids_.size();
            for (const auto& [_, fleet] : fleets_) {
                if (fleet.player_id_ == player_id) {
                    cargo += fleet.kore_;
                    ships += fleet.ship_count_;
                }
            }
            for (const auto& [_, shipyard] : shipyards_) {
                if (shipyard.player_id_ == player_id) {
                    ships += shipyard.ship_count_;
                    max_spawn += shipyard.MaxSpawn();
                }
            }
            cout << kTextColors[player_colors[player_id]];
            cout << setw(9) << player_id << " / ";
            cout << setw(9) << kore << " / ";
            cout << setw(9) << cargo << " / ";
            cout << setw(9) << ships << " / ";
            cout << setw(9) << n_shipyards << " / ";
            cout << setw(9) << max_spawn << " / ";
            cout << setw(9) << kore + cargo + ships * 10 + n_shipyards * 500
                 << endl;
        }
        cout << kResetTextStyle;
        cout << "                      ";
        cout << kTextBold << "[^]" << kResetTextStyle
             << "ships/kore/flight_plan" << kTextBold << " [@]"
             << kResetTextStyle << "ships/max_spawn" << endl;
        for (auto y = 0; y < kSize; y++) {
            for (auto x = 0; x < kSize; x++) {
                cout << kResetTextStyle;
                const auto& cell = board_[{y, x}];
                if (cell.fleet_id_ != -1) {
                    const auto& fleet = fleets_.at(cell.fleet_id_);
                    cout << kTextColors[player_colors[fleet.player_id_]];
                    cout << kTextBold << "^>v<"[(int)fleet.direction_];
                } else if (cell.shipyard_id_ != -1) {
                    const auto& shipyard = shipyards_.at(cell.shipyard_id_);
                    cout << kTextColors[player_colors[shipyard.player_id_]];
                    cout << kTextBold << "@";
                } else {
                    cout << kTextDim;
                    cout << ".,-~:;!*#$"[min(
                        9, (int)(9.0 * cell.kore_ / kMaxRegenCellKore))];
                }
            }

            for (auto x = 0; x < kSize; x++) {
                cout << kResetTextStyle;
                const auto& cell = board_[{y, x}];
                if (cell.fleet_id_ != -1) {
                    const auto& fleet = fleets_.at(cell.fleet_id_);
                    cout << kTextColors[player_colors[fleet.player_id_]];
                    cout << kTextBold << " ["
                         << "^>v<"[(int)fleet.direction_] << "]";
                    cout << kResetTextStyle;
                    cout << kTextColors[player_colors[fleet.player_id_]];
                    cout << fleet.ship_count_ << "/" << fleet.kore_ << "/"
                         << (fleet.flight_plan_.size() ? fleet.flight_plan_
                                                       : "_");
                } else if (cell.shipyard_id_ != -1) {
                    const auto& shipyard = shipyards_.at(cell.shipyard_id_);
                    cout << kTextColors[player_colors[shipyard.player_id_]];
                    cout << kTextBold << " [@]";
                    cout << kResetTextStyle;
                    cout << kTextColors[player_colors[shipyard.player_id_]];
                    cout << shipyard.ship_count_ << "/" << shipyard.MaxSpawn();
                }
            }
            cout << endl;
        }
        cout << kResetTextStyle << defaultfloat
             << setprecision(initial_precision);
    }

    auto Same(const State& rhs) const {
        if (step_ != rhs.step_)
            return false;

        for (auto y = 0; y < kSize; y++) {
            for (auto x = 0; x < kSize; x++) {
                const auto& cell = board_[{y, x}];
                const auto& rhs_cell = rhs.board_[{y, x}];
                // cout << "y,x=" << y << "," << x << endl;
                if (!IsClose(cell.kore_, rhs_cell.kore_))
                    return false;
                if ((cell.fleet_id_ == -1) != (rhs_cell.fleet_id_ == -1))
                    return false;
                if (cell.fleet_id_ != -1 &&
                    !fleets_.at(cell.fleet_id_)
                         .Same(rhs.fleets_.at(rhs_cell.fleet_id_))) {
                    cerr << "fleets not same: "
                         << fleets_.at(cell.fleet_id_).flight_plan_ << " "
                         << rhs.fleets_.at(rhs_cell.fleet_id_).flight_plan_
                         << endl;
                    return false;
                }

                if ((cell.shipyard_id_ == -1) != (rhs_cell.shipyard_id_ == -1))
                    return false;
                if (cell.shipyard_id_ != -1 &&
                    !shipyards_.at(cell.shipyard_id_)
                         .Same(rhs.shipyards_.at(rhs_cell.shipyard_id_)))
                    return false;
            }
        }
        return true;
    }

    void AddFleet(const Fleet& fleet) {
        players_[fleet.player_id_].fleet_ids_.insert(fleet.id_);
        board_[fleet.position_].fleet_id_ = fleet.id_;
        const auto [it, inserted] = fleets_.insert(make_pair(fleet.id_, fleet));
        assert(inserted);
    }

    void AddShipyard(const Shipyard& shipyard) {
        players_[shipyard.player_id_].shipyard_ids_.insert(shipyard.id_);
        assert(board_[shipyard.position_].shipyard_id_ == -1);
        board_[shipyard.position_].shipyard_id_ = shipyard.id_;
        board_[shipyard.position_].kore_ = 0.0;
        const auto [it, inserted] =
            shipyards_.insert(make_pair(shipyard.id_, shipyard));
        assert(inserted);
    }

    void DeleteFleet(const Fleet& fleet, const FleetReportType cause) {
        players_[fleet.player_id_].fleet_ids_.erase(fleet.id_);
        if (board_[fleet.position_].fleet_id_ == fleet.id_) {
            board_[fleet.position_].fleet_id_ = -1;
        }
        if constexpr (kDebug) {
            fleet_reports_.insert(make_pair(
                fleet.id_, FleetReport{cause, fleet.position_, true}));
        }
        fleets_.erase(fleet.id_);
    }

    void DeleteShipyard(const Shipyard& shipyard) {
        players_[shipyard.player_id_].shipyard_ids_.erase(shipyard.id_);
        if (board_[shipyard.position_].shipyard_id_ == shipyard.id_) {
            board_[shipyard.position_].shipyard_id_ = -1;
        } else {
            cerr << "DeleteShipyard: そんなことある？" << endl;
        }
        shipyards_.erase(shipyard.id_);
    }

    auto Next(istream& is) const {
        return Next(Action().Read(shipyard_id_mapper_, is));
    }

    State Next(const Action& action) const {
        auto state = *this;

        // shipyard の情報をリセット
        for (auto& [_, shipyard] : state.shipyards_) {
            shipyard.last_ship_increment_ = 0;
            shipyard.last_ship_decrement_ = 0;
        }

        // reports の情報をリセット
        state.fleet_reports_.clear();

        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            auto& player = state.players_[player_id];
            for (const auto& shipyard_id : player.shipyard_ids_) {
                auto& shipyard = state.shipyards_.at(shipyard_id);
                const auto it = action.actions.find(shipyard_id);
                if (it == action.actions.end())
                    continue;
                const auto& shipyard_action = it->second;
                if (shipyard_action.num_ships_ == 0) {
                    continue;
                } else if (shipyard_action.type_ ==
                           ShipyardActionType::kSpawn) {
                    if (player.kore_ >=
                            kSpawnCost * shipyard_action.num_ships_ &&
                        shipyard_action.num_ships_ <= shipyard.MaxSpawn()) {
                        player.kore_ -= kSpawnCost * shipyard_action.num_ships_;
                        shipyard.ship_count_ += shipyard_action.num_ships_;
                        shipyard.last_ship_increment_ +=
                            shipyard_action.num_ships_;
                    } else {
                        cerr << "Next: その Spawn は無理" << endl;
                    }
                } else if (shipyard_action.type_ ==
                           ShipyardActionType::kLaunch) {
                    if (shipyard.ship_count_ >= shipyard_action.num_ships_) {
                        const auto& flight_plan = shipyard_action.flight_plan_;
                        shipyard.ship_count_ -= shipyard_action.num_ships_;
                        shipyard.last_ship_decrement_ +=
                            shipyard_action.num_ships_;
                        const auto direction = CharToDirection(flight_plan[0]);
                        const auto max_flight_plan_len =
                            Fleet::MaxFlightPlanLenForShipCount(
                                shipyard_action.num_ships_);
                        if ((int)flight_plan.size() > max_flight_plan_len) {
                            // 元の実装だと縮めてるけどこの実装だとスキップする
                            cerr << "Next: flight_plan が長すぎ" << endl;
                        } else {
                            state.AddFleet(Fleet{state.next_fleet_id_++,
                                                 shipyard_action.num_ships_,
                                                 direction, shipyard.position_,
                                                 0, flight_plan, player_id});
                        }
                    }
                } else {
                    assert(false);
                }
            }
        }

        // 元の実装だとプレイヤーごと　上のも一度にやってよいのでは？？
        for (auto&& [_, shipyard] : state.shipyards_) {
            shipyard.turns_controlled_++;
        }

        const auto find_first_non_digit = [](const string& s) {
            for (auto i = 0; i < (int)s.size(); i++)
                if (s[i] > '9')
                    return i;
            return (int)s.size(); // 元の実装、1 を足す意味無いよね
        };

        // 元の実装だとプレイヤーごと
        for (auto it = state.fleets_.begin(); it != state.fleets_.end();) {
            // 0 を無視
            auto& fleet = it->second;
            while (fleet.flight_plan_.size() > 0 &&
                   fleet.flight_plan_[0] == '0') {
                fleet.flight_plan_ = fleet.flight_plan_.substr(1);
            }
            // 造船所を建造
            if (fleet.flight_plan_.size() > 0 && fleet.flight_plan_[0] == 'C' &&
                fleet.ship_count_ >= kConvertCost &&
                state.board_[fleet.position_].shipyard_id_ == -1) {
                state.players_[fleet.player_id_].kore_ += fleet.kore_;
                state.board_[fleet.position_].kore_ = 0;
                state.AddShipyard(Shipyard{
                    state.next_shipyard_id_++, fleet.ship_count_ - kConvertCost,
                    fleet.position_, fleet.player_id_, 0});
                it++; // 削除する前にイテレータを進める
                state.DeleteFleet(fleet, FleetReportType::kConverted);
                continue;
            }
            // C を無視
            while (fleet.flight_plan_.size() > 0 &&
                   fleet.flight_plan_[0] == 'C') {
                fleet.flight_plan_ = fleet.flight_plan_.substr(1);
            }
            if (fleet.flight_plan_.size() && fleet.flight_plan_[0] >= 'A') {
                // 方向転換
                fleet.direction_ = CharToDirection(fleet.flight_plan_[0]);
                fleet.flight_plan_ = fleet.flight_plan_.substr(1);
            } else if (fleet.flight_plan_.size()) {
                // 直進
                const auto idx = find_first_non_digit(fleet.flight_plan_);
                auto digits = stoi(fleet.flight_plan_.substr(0, idx));
                const auto rest = fleet.flight_plan_.substr(idx);
                digits -= 1;
                if (digits > 0) {
                    fleet.flight_plan_ = to_string(digits) + rest;
                } else {
                    fleet.flight_plan_ = rest;
                }
            }
            // 進む
            state.board_[fleet.position_].fleet_id_ = -1;
            fleet.position_ = GetToPos(fleet.position_, fleet.direction_);
            // NOTE: ここではまだ cell にはセットしない

            it++;
        }

        const auto combine_fleets = [&state](const FleetId fid1,
                                             const FleetId fid2) {
            auto& f1 = state.fleets_.at(fid1);
            auto& f2 = state.fleets_.at(fid2);
            if (f1.LessThanOtherAlliedFleet(f2)) {
                assert(false);
                f2.kore_ += f1.kore_;
                f2.ship_count_ += f1.ship_count_;
                state.DeleteFleet(f1, FleetReportType::kMerged);
                return fid2;
            } else {
                f1.kore_ += f2.kore_;
                f1.ship_count_ += f2.ship_count_;
                state.DeleteFleet(f2, FleetReportType::kMerged);
                return fid1;
            }
        };

        // プレイヤーごと場所ごとに fleet を集める
        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            const auto& player = state.players_[player_id];
            auto fleets_by_loc = unordered_map<Point, vector<FleetId>>();
            for (const auto& fleet_id : player.fleet_ids_)
                fleets_by_loc[state.fleets_.at(fleet_id).position_].push_back(
                    fleet_id);
            for (auto&& [_, value] : fleets_by_loc) {
                // これソートいらないかと思いきや 3 つ同時とかだと必要になる
                sort(value.begin(), value.end(),
                     [&state](const FleetId& l, const FleetId& r) {
                         return state.fleets_.at(r).LessThanOtherAlliedFleet(
                             state.fleets_.at(l));
                     });
                const auto fid = value[0];
                for (auto i = 1; i < (int)value.size(); i++) {
                    const auto res_fid = combine_fleets(fid, value[i]);
                    assert(fid == res_fid);
                    if constexpr (kDebug)
                        state.fleet_reports_.insert(make_pair(
                            fid,
                            FleetReport{FleetReportType::kMerged,
                                        state.fleets_.at(res_fid).position_,
                                        false}));
                }
            }
        }

        const auto resolve_collision = [&state](const vector<FleetId>& fleets) {
            struct Result {
                FleetId winner;
                vector<FleetId> deleted;
            };
            if (fleets.size() == 1)
                return Result{fleets[0], {}};
            assert(fleets.size() == 2);
            const auto& fid0 = fleets[0];
            const auto& fid1 = fleets[1];
            const auto ship_count_0 = state.fleets_.at(fid0).ship_count_;
            const auto ship_count_1 = state.fleets_.at(fid1).ship_count_;
            if (ship_count_0 < ship_count_1)
                return Result{fid1, {fid0}};
            else if (ship_count_0 > ship_count_1)
                return Result{fid0, {fid1}};
            else
                return Result{-1, fleets};
        };

        // 艦隊同士の衝突
        auto fleet_collision_groups = unordered_map<Point, vector<FleetId>>();
        for (const auto& [fleet_id, fleet] : state.fleets_) {
            fleet_collision_groups[fleet.position_].push_back(fleet_id);
        }
        for (const auto& [position, collided_fleets] : fleet_collision_groups) {
            const auto [winner, deleted] = resolve_collision(collided_fleets);
            // shipyard を変な方法で取得する意味とは・・・
            // shipyard、同時に同じ箇所に 2 つあるわけないよね？
            const auto shipyard_id = state.board_[position].shipyard_id_;
            if (winner != -1) {
                state.board_[position].fleet_id_ = winner;
                const auto max_enemy_size =
                    deleted.size() == 0
                        ? (short)0
                        : state.fleets_.at(deleted[0]).ship_count_;
                state.fleets_.at(winner).ship_count_ -= max_enemy_size;
                if constexpr (kDebug)
                    if (max_enemy_size)
                        state.fleet_reports_.insert(make_pair(
                            winner, FleetReport{FleetReportType::kCollided,
                                                position, false}));
            }
            for (const auto& fleet_id : deleted) {
                const auto fleet = state.fleets_.at(fleet_id);
                state.DeleteFleet(fleet, FleetReportType::kCollided);
                if (winner != -1) {
                    state.fleets_.at(winner).kore_ += fleet.kore_;
                } else if (shipyard_id != -1 &&
                           state.shipyards_.at(shipyard_id).player_id_ !=
                               -1) { // player の条件いる？
                    state.players_[state.shipyards_.at(shipyard_id).player_id_]
                        .kore_ += fleet.kore_;
                } else {
                    state.board_[position].kore_ += fleet.kore_;
                }
            }
        }

        // 艦隊と造船所の衝突
        for (auto it = state.shipyards_.begin();
             it != state.shipyards_.end();) {
            auto& shipyard = it->second;
            it++;
            const auto& fleet_id = state.board_[shipyard.position_].fleet_id_;
            if (fleet_id != -1 &&
                state.fleets_.at(fleet_id).player_id_ != shipyard.player_id_) {
                const auto& fleet = state.fleets_.at(fleet_id);
                if (fleet.ship_count_ > shipyard.ship_count_) {
                    const auto count = fleet.ship_count_ - shipyard.ship_count_;
                    assert(fleet.position_ == shipyard.position_);
                    state.DeleteShipyard(shipyard);
                    state.AddShipyard({state.next_shipyard_id_++, count,
                                       fleet.position_, fleet.player_id_, 1});
                    state.players_[fleet.player_id_].kore_ += fleet.kore_;
                    state.DeleteFleet(fleet, FleetReportType::kCollided);
                } else {
                    shipyard.ship_count_ -= fleet.ship_count_;
                    shipyard.last_ship_decrement_ += fleet.ship_count_;
                    state.players_[shipyard.player_id_].kore_ += fleet.kore_;
                    state.DeleteFleet(fleet, FleetReportType::kCollided);
                }
            }
        }

        // 帰港
        for (auto&& [_, shipyard] : state.shipyards_) {
            const auto& fleet_id = state.board_[shipyard.position_].fleet_id_;
            if (fleet_id != -1) {
                const auto& fleet = state.fleets_.at(fleet_id);
                assert(fleet.player_id_ == shipyard.player_id_);
                state.players_[shipyard.player_id_].kore_ += fleet.kore_;
                shipyard.ship_count_ += fleet.ship_count_;
                shipyard.last_ship_increment_ += fleet.ship_count_;
                state.DeleteFleet(fleet, FleetReportType::kArrived);
            }
        }

        // 隣接した艦隊同士のダメージ
        auto incoming_fleet_dmg = map<FleetId, map<FleetId, int>>();
        for (const auto& [_, fleet] : state.fleets_) {
            for (const auto direction :
                 {Direction::N, Direction::E, Direction::S, Direction::W}) {
                const auto curr_pos = GetToPos(fleet.position_, direction);
                const auto& fleet_id_at_pos = state.board_[curr_pos].fleet_id_;
                if (fleet_id_at_pos != -1 &&
                    state.fleets_.at(fleet_id_at_pos).player_id_ !=
                        fleet.player_id_) {
                    incoming_fleet_dmg[fleet_id_at_pos][fleet.id_] =
                        fleet.ship_count_;
                }
            }
        }

        // 隣接した攻撃で撃沈したら、kore の 1/2 をその場に落とし、
        // 残りを与えたダメージ量に比例して周りの艦隊に分配する
        auto to_distribute =
            unordered_map<FleetId, unordered_map<Point, double>>();
        for (const auto& [fleet_id, fleet_dmg_dict] : incoming_fleet_dmg) {
            auto& fleet = state.fleets_.at(fleet_id);
            auto damage = 0;
            for (const auto& [_, dmg] : fleet_dmg_dict)
                damage += dmg;
            if (damage >= fleet.ship_count_) {
                state.board_[fleet.position_].kore_ +=
                    (double)fleet.kore_ * 0.5;
                const auto to_split = (double)fleet.kore_ * 0.5;
                for (const auto& [f_id, dmg] : fleet_dmg_dict)
                    to_distribute[f_id][fleet.position_] =
                        to_split * (double)dmg / (double)damage;
                state.DeleteFleet(fleet, FleetReportType::kCollided);
            } else {
                fleet.ship_count_ -= damage;
                if constexpr (kDebug)
                    if (damage)
                        state.fleet_reports_.insert(make_pair(
                            fleet.id_, FleetReport{FleetReportType::kCollided,
                                                   fleet.position_, false}));
            }
        }

        // 艦隊が生き残っていたら分配された kore を獲得する
        // そうでなければ撃沈したところに戻る
        for (const auto& [fleet_id, loc_kore_dict] : to_distribute) {
            auto fleet_it = state.fleets_.find(fleet_id);
            if (fleet_it != state.fleets_.end()) {
                auto sum_kore = 0.0;
                for (const auto& [_, kore] : loc_kore_dict)
                    sum_kore += kore;
                fleet_it->second.kore_ += sum_kore; // これ int じゃだめな気が
            } else {
                for (const auto& [loc_idx, kore] : loc_kore_dict)
                    state.board_[loc_idx].kore_ += kore;
            }
        }

        // 艦隊が kore を集める
        for (auto& [_, fleet] : state.fleets_) {
            auto& cell = state.board_[fleet.position_];
            const auto delta_kore =
                round(cell.kore_ * fleet.CollectionRate() * 1e3) * 1e-3;
            if (delta_kore > 0.0) {
                fleet.kore_ += delta_kore;
                cell.kore_ -= delta_kore;
            }
        }

        // kore の生成
        for (auto&& cell : state.board_.data) {
            if (cell.fleet_id_ == -1 && cell.shipyard_id_ == -1) {
                if (cell.kore_ < kMaxRegenCellKore) {
                    cell.kore_ = Round3(cell.kore_ * (1.0 + kRegenRate));
                }
            }
        }

        state.step_++;

        return state;
    }
};

struct KifHeader {
    string format_version;
    string kif_id;
    array<string, 2> players_info;

    auto& Read(istream& is) {
        getline(is, format_version);
        getline(is, kif_id);
        for (auto i = 0; i < 2; i++)
            getline(is, players_info[i]);
        return *this;
    }
};

struct Agent {
    // player_id が -1 で両方
    virtual Action ComputeNextMove(const State&, const PlayerId) const = 0;
};

struct SpawnAgent : Agent {
    Action ComputeNextMove(const State& state, const PlayerId player_id) const {
        auto kore = state.players_[player_id].kore_;
        auto action = Action();
        for (const auto& [shipyard_id, shipyard] : state.shipyards_) {
            if (shipyard.player_id_ != player_id)
                continue;
            if (kore < 10.0)
                break;
            const auto max_spawn = shipyard.MaxSpawn();
            const auto n_spawn = min(max_spawn, (int)(kore / kSpawnCost));
            kore -= n_spawn * kSpawnCost;
            const auto [it, inserted] = action.actions.insert(
                make_pair(shipyard_id,
                          ShipyardAction(ShipyardActionType::kSpawn, n_spawn)));
            assert(inserted);
        }
        return action;
    }
};

#include "../marathon/nn.cpp"

constexpr auto Relative(const Point p) {
    constexpr auto kHalf = kSize / 2;
    return Point(p.y > kHalf    ? p.y - kSize
                 : p.y < -kHalf ? p.y + kSize
                                : p.y,
                 p.x > kHalf    ? p.x - kSize
                 : p.x < -kHalf ? p.x + kSize
                                : p.x);
}

constexpr auto RelativeAll(const Point p) {
    const auto y = p.y >= 0 ? p.y : p.y + kSize;
    const auto x = p.x >= 0 ? p.x : p.x + kSize;
    return y * kSize + x;
}

static constexpr int PointTimeIndexOffset(const int n) {
    return (n * n * n * 2 + n * n * 6 + n * 4) / 3;
}

struct NNUEFeature {
    static constexpr auto kNGlobalFeatures = 9;

    static constexpr auto kFutureSteps = 10;
    static constexpr auto kNPointTimeIndices =
        PointTimeIndexOffset(kFutureSteps + 1);

    static constexpr auto kFleetResolution = 10;
    static constexpr auto kFieldKoreResolution = 5;

    static inline auto PointToIndex(const Point relative_point) {
        const auto [y, x] = relative_point;
        const auto d = relative_point.l1_norm();
        const auto offset = 2 * d * (d - 1);
        if (x >= 0) {
            if (y < 0)
                return offset + x;
            else
                return offset + d + y;
        } else {
            if (y >= 0)
                return offset + d * 2 - x;
            else
                return offset + d * 3 - y;
        }
    }

    static inline auto PointTimeToIndex(const Point relative_point,
                                        const int n) {
        assert(0 <= n && n <= kFutureSteps);
        const auto offset = PointTimeIndexOffset(n);
        const auto point_index = PointToIndex(relative_point);
        const auto result = offset + point_index;
        assert(offset <= result && result < PointTimeIndexOffset(n + 1));
        return result;
    }

    auto QuantizeNShips(const int n_ships) {
        if (n_ships == 0)
            return 0;
        const auto result =
            clamp((int)(2 * log(n_ships)) - 2, 0, kFleetResolution - 1);

        assert(0 <= result && result < kFleetResolution);
        return result;
    }

    static constexpr auto kNFleetFeatureTypes =
        2 * kNPointTimeIndices * kFleetResolution;

    static constexpr auto kNShipyardFeatureTypes =
        kSize * kSize * 2 * kFleetResolution;

    static constexpr auto kNKoreFeatureTypes =
        2 * kFutureSteps * (kFutureSteps + 1) * kFieldKoreResolution;

    static constexpr auto kNFeatureTypes =
        kNFleetFeatureTypes + kNShipyardFeatureTypes + kNKoreFeatureTypes;

    static array<string, kNFeatureTypes> feature_names;

    static void SetFeatureNames() {
        // fleet
        for (auto t = 0; t <= kFutureSteps; t++) {
            for (auto y = -10; y <= 10; y++)
                for (auto x = -10; x <= 10; x++) {
                    const auto p = Point(y, x);
                    if (p.l1_norm() > t + 1 || (y == 0 && x == 0))
                        continue;
                    const auto relative_point_index = PointTimeToIndex(p, t);
                    for (auto away = 0; away < 2; away++)
                        for (auto n = 0; n < kFleetResolution; n++) {
                            const auto feature =
                                (relative_point_index * 2 + away) *
                                    kFleetResolution +
                                n;
                            feature_names[feature] =
                                string("fleet_step") + to_string(t) +
                                string("_y") + to_string(y) + string("_x") +
                                to_string(x) +
                                string(away ? "_away" : "_home") +
                                string("_ships") + to_string(n);
                        }
                }
        }

        // shipyard
        auto offset = kNFleetFeatureTypes;
        for (auto y = -10; y <= 10; y++)
            for (auto x = -10; x <= 10; x++) {
                const auto p = Point(y, x);
                const auto relative_point_index = RelativeAll(p);
                for (auto away = 0; away < 2; away++)
                    for (auto n = 0; n < kFleetResolution; n++) {
                        const auto feature = offset +
                                             (relative_point_index * 2 + away) *
                                                 kFleetResolution +
                                             n;
                        feature_names[feature] =
                            string("shipyard_y") + to_string(y) + string("_x") +
                            to_string(x) + string(away ? "_away" : "_home") +
                            string("_ships") + to_string(n);
                    }
            }

        // kore
        offset += kNShipyardFeatureTypes;
        for (auto y = -10; y <= 10; y++)
            for (auto x = -10; x <= 10; x++) {
                const auto p = Point(y, x);
                const auto relative_point_index = PointToIndex(p);
                if (p.l1_norm() > kFutureSteps || (p.y == 0 && p.x == 0))
                    continue;
                for (auto n = 0; n < kFieldKoreResolution; n++) {
                    const auto feature =
                        offset + relative_point_index * kFieldKoreResolution +
                        n;
                    feature_names[feature] = string("kore_y") + to_string(y) +
                                             string("_x") + to_string(x) +
                                             string("_amount") + to_string(n);
                }
            }
    }

    auto QuantizeFieldKore(const double kore, const int step) {
        const auto result = clamp((int)(kore / kore_90_percentiles[step] *
                                        kFieldKoreResolution),
                                  0, kFieldKoreResolution) -
                            1;
        assert(-1 <= result && result < kFieldKoreResolution);
        return result;
    }

    // [player][shipyard][idx_features]
    array<unordered_map<int, vector<int>>, 2> shipyard_features;
    array<array<float, kNGlobalFeatures>, 2> global_features;

    struct FeatureInfo {
        unsigned char flags;
        float kore;
    };
    array<Board<FeatureInfo, kSize, kSize>, 22> future_info;
    enum : unsigned char {
        kPlayer0Shipyard = 1,
        kPlayer0Fleet = 2,
        kPlayer0FleetAdjacent = 4,
        kPlayer1Shipyard = 8,
        kPlayer1Fleet = 16,
        kPlayer1FleetAdjacent = 32,
    };

    NNUEFeature(const State& state) {
        // 共通した global feature と、shipyard ごとの feature を作る
        // 2 人分

        auto state_i = state;

        fill((FeatureInfo*)future_info.begin(), (FeatureInfo*)future_info.end(),
             FeatureInfo());

        // fleet
        for (auto i = 0; i <= 21; i++) {
            // future_info に書き込む
            for (const auto& [_, fleet] : state_i.fleets_) {
                const auto& p = fleet.position_;
                future_info[i][p].flags |=
                    fleet.player_id_ == 0 ? kPlayer0Fleet : kPlayer1Fleet;
                future_info[i][{(int)p.y, p.x == 0 ? kSize - 1 : p.x - 1}]
                    .flags |= fleet.player_id_ == 0 ? kPlayer0FleetAdjacent
                                                    : kPlayer1FleetAdjacent;
                future_info[i][{(int)p.y, p.x == kSize - 1 ? 0 : p.x + 1}]
                    .flags |= fleet.player_id_ == 0 ? kPlayer0FleetAdjacent
                                                    : kPlayer1FleetAdjacent;
                future_info[i][{p.y == 0 ? kSize - 1 : p.y - 1, (int)p.x}]
                    .flags |= fleet.player_id_ == 0 ? kPlayer0FleetAdjacent
                                                    : kPlayer1FleetAdjacent;
                future_info[i][{p.y == kSize - 1 ? 0 : p.y + 1, (int)p.x}]
                    .flags |= fleet.player_id_ == 0 ? kPlayer0FleetAdjacent
                                                    : kPlayer1FleetAdjacent;
            }
            for (const auto& [_, shipyard] : state_i.shipyards_) {
                const auto& p = shipyard.position_;
                future_info[i][{p.y, p.x}].flags |= shipyard.player_id_ == 0
                                                        ? kPlayer0Shipyard
                                                        : kPlayer1Shipyard;
            }
            for (auto y = 0; y < kSize; y++)
                for (auto x = 0; x < kSize; x++)
                    future_info[i][{y, x}].kore = state_i.board_[{y, x}].kore_;
            if (i > kFutureSteps)
                continue;

            // shipyard feature
            for (const auto& [_, center_shipyard] : state.shipyards_) {

                for (const auto& [_, fleet] : state_i.fleets_) {
                    const auto relative_point =
                        Relative(fleet.position_ - center_shipyard.position_);
                    if (relative_point.l1_norm() > i + 1)
                        continue;
                    const auto away =
                        fleet.player_id_ != center_shipyard.player_id_;

                    const auto feature =
                        (PointTimeToIndex(relative_point, i) * 2 + away) *
                            kFleetResolution +
                        QuantizeNShips(fleet.ship_count_);
                    shipyard_features[center_shipyard.player_id_]
                                     [center_shipyard.id_]
                                         .push_back(feature);
                }
            }

            // 次のターンへ
            auto action = SpawnAgent().ComputeNextMove(state_i, 0);
            auto action1 = SpawnAgent().ComputeNextMove(state_i, 1);
            action.Merge(action1);
            assert(action1.actions.size() == 0);
            state_i = state_i.Next(action);
        }

        // shipyard
        // shipyard は全箇所必要
        auto offset = kNFleetFeatureTypes;
        for (const auto& [_, center_shipyard] : state.shipyards_) {
            for (const auto& [_, shipyard] : state.shipyards_) {
                const auto relative_point_index =
                    RelativeAll(shipyard.position_ - center_shipyard.position_);

                const auto away =
                    shipyard.player_id_ != center_shipyard.player_id_;

                const auto feature =
                    offset +
                    (relative_point_index * 2 + away) * kFleetResolution +
                    QuantizeNShips(shipyard.ship_count_);

                shipyard_features[center_shipyard.player_id_]
                                 [center_shipyard.id_]
                                     .push_back(feature);
            }
        }

        offset += kNShipyardFeatureTypes;

        // kore
        for (auto y = 0; y < kSize; y++) {
            for (auto x = 0; x < kSize; x++) {
                const auto p = Point(y, x);
                if (state.board_[{y, x}].shipyard_id_ >= 0)
                    continue;
                const auto kore = state.board_[{y, x}].kore_;
                const auto encoded = QuantizeFieldKore(kore, state.step_);
                if (encoded == -1)
                    continue;
                for (const auto& [_, center_shipyard] : state.shipyards_) {
                    const auto relative_point =
                        Relative(p - center_shipyard.position_);
                    if (relative_point.l1_norm() > kFutureSteps)
                        continue;
                    const auto relative_point_index =
                        PointToIndex(relative_point);
                    const auto feature =
                        offset + relative_point_index * kFieldKoreResolution +
                        encoded;
                    shipyard_features[center_shipyard.player_id_]
                                     [center_shipyard.id_]
                                         .push_back(feature);
                }
            }
        }

        offset += kNKoreFeatureTypes;

        // global features
        const auto sum_cargo = state.CountCargo();
        const auto n_ships = state.CountShips();
        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            auto idx_global_features = 0;
            auto& g = global_features[player_id];
            g[idx_global_features++] = state.step_ * 1e-2;
            g[idx_global_features++] = state.players_[player_id].kore_ * 1e-3;
            g[idx_global_features++] =
                state.players_[1 - player_id].kore_ * 1e-3;
            g[idx_global_features++] = sum_cargo[player_id] * 1e-3;
            g[idx_global_features++] = sum_cargo[1 - player_id] * 1e-3;
            g[idx_global_features++] = n_ships[player_id] * 1e-2;
            g[idx_global_features++] = n_ships[1 - player_id] * 1e-2;
            g[idx_global_features++] =
                state.players_[player_id].shipyard_ids_.size() * 0.1;
            g[idx_global_features++] =
                state.players_[1 - player_id].shipyard_ids_.size() * 0.1;

            assert(idx_global_features == kNGlobalFeatures);
        }
    }
};
array<string, NNUEFeature::kNFeatureTypes> NNUEFeature::feature_names;

enum struct ActionTargetType : short {
    kNull = -1,
    kSpawn,
    kMove,
    kAttack,
    kConvert
};

struct ActionTarget {

    ActionTargetType action_target_type;

  private:
    short n_ships_;
    Point relative_position_;
    short n_steps_;
    Direction initial_move_;

  public:
    // spawn => 艦数
    // move => 移動先の場所、かけるターン数、艦数
    // attack => 攻撃先の場所、艦数、初手
    // convert => 建設場所、艦数、初手

    // move は機体の回収を含むとする

    // attack と convert が最短距離でなかった場合、無視する

    ActionTarget(const State& state, const ShipyardId shipyard_id,
                 const ShipyardAction shipyard_action) {
        n_ships_ = shipyard_action.num_ships_;
        const auto& center_shipyard = state.shipyards_.at(shipyard_id);
        if (n_ships_ >= 256) {
            action_target_type = ActionTargetType::kNull;
            goto ok;
        }
        if (shipyard_action.type_ == ShipyardActionType::kSpawn) {
            action_target_type = ActionTargetType::kSpawn;
        } else {
            auto initial_move_char = shipyard_action.flight_plan_[0];
            if (initial_move_char == 'C') {
                action_target_type = ActionTargetType::kNull;
                goto ok;
            }
            initial_move_ = CharToDirection(initial_move_char);
            auto action = Action();
            action.actions.insert(make_pair(shipyard_id, shipyard_action));
            auto state_i = state.Next(action);
            const auto fleet_id = FleetId(state_i.next_fleet_id_ - 1);
            for (auto i = 0; i < 21; i++) {
                const auto it = state_i.fleet_reports_.find(fleet_id);
                if (it != state_i.fleet_reports_.end()) {
                    const auto& report = it->second;
                    relative_position_ =
                        Relative(report.position_ - center_shipyard.position_);
                    n_steps_ = state_i.step_ - state.step_;
                    assert(n_steps_ == i + 1);
                    switch (report.type_) {
                    case FleetReportType::kArrived:
                    case FleetReportType::kMerged:
                        action_target_type = ActionTargetType::kMove;
                        goto ok;
                    case FleetReportType::kCollided:
                        if (relative_position_.l1_norm() != n_steps_)
                            break;
                        action_target_type = ActionTargetType::kAttack;
                        goto ok;
                    case FleetReportType::kConverted:
                        if (relative_position_.l1_norm() + 1 != n_steps_)
                            break;
                        action_target_type = ActionTargetType::kConvert;
                        goto ok;
                    }
                    if (report.deleted) {
                        action_target_type = ActionTargetType::kNull;
                        goto ok;
                    }
                }
                state_i = state_i.Next({});
            }
            action_target_type = ActionTargetType::kNull;
        }
    ok:;
    }

    const auto& NShips() const {
        // TODO: これいい感じに量子化したほうが良い気がしてきた
        assert(action_target_type != ActionTargetType::kNull);
        return n_ships_;
    }
    const auto& RelativePosition() const {
        assert(action_target_type != ActionTargetType::kNull &&
               action_target_type != ActionTargetType::kSpawn);
        return relative_position_;
    }
    const auto& NSteps() const {
        // attack と convert もまあ一応大丈夫ではあるが…
        assert(action_target_type == ActionTargetType::kMove);
        return n_steps_; // [1, 21]
    }
    const auto& InitialMove() const {
        assert(action_target_type == ActionTargetType::kAttack ||
               action_target_type == ActionTargetType::kConvert);
        return initial_move_;
    }

    auto QuantizedNships() const {
        auto n = n_ships_;
        switch (action_target_type) {
        case ActionTargetType::kConvert:
            n -= 50;
            [[fallthrough]];
        case ActionTargetType::kSpawn:
        case ActionTargetType::kMove:
        case ActionTargetType::kAttack:
            return n < (int)kNShipsQuantizationTable.size()
                       ? kNShipsQuantizationTable[n]
                       : kNShipsQuantizationTable.back() + 1;
        default:
            assert(false);
        }
    }
};

struct Feature {
    static constexpr auto kFutureSteps = 21;
    static constexpr auto kNLocalFeatures = 13 * kFutureSteps;
    static constexpr auto kNGlobalFeatures = 8 * kFutureSteps + 1;
    nn::TensorBuffer<float, kNLocalFeatures, kSize, kSize> local_features;
    nn::TensorBuffer<float, kNGlobalFeatures> global_features;
    Feature(State state, const PlayerId player_id)
        : local_features(), global_features() {
        auto idx_local_features = 0;
        auto idx_global_features = 0;

        const auto step = state.step_;

        static auto reachable =
            nn::TensorBuffer<short, kFutureSteps, 2, kSize * 2, kSize * 2>();
        static auto reachable_all = nn::TensorBuffer<short, kFutureSteps, 2>();
        reachable.Fill_(0);
        reachable_all.Fill_(0);

        // 21 ステップ先まで見る
        for (auto i = 0; i < kFutureSteps; i++) {
            auto n_ships = array<short, 2>();
            auto sum_cargo = array<float, 2>();

            // local

            // board kore
            for (auto y = 0; y < kSize; y++) {
                for (auto x = 0; x < kSize; x++) {
                    local_features[idx_local_features][y][x] =
                        state.board_[{y, x}].kore_;
                }
            }
            idx_local_features++;

            // shipyard (自分, 相手) x (ships, max_spawn)
            for (const auto& [_, shipyard] : state.shipyards_) {
                const auto [y, x] = shipyard.position_;
                const auto away = shipyard.player_id_ != player_id;
                local_features[idx_local_features + away * 2][y][x] =
                    shipyard.ship_count_;
                local_features[idx_local_features + away * 2 + 1][y][x] =
                    shipyard.MaxSpawn();
                n_ships[away] += shipyard.ship_count_;
            }
            idx_local_features += 4;

            // fleet (自分, 相手)
            //        x (ships, cargo, flight_plan_length, 被撃墜可能性)
            for (const auto& [_, fleet] : state.fleets_) {
                const auto [y, x] = fleet.position_;
                const auto away = fleet.player_id_ != player_id;
                local_features[idx_local_features + away * 3][y][x] =
                    fleet.ship_count_;
                local_features[idx_local_features + away * 3 + 1][y][x] =
                    fleet.kore_;
                local_features[idx_local_features + away * 3 + 2][y][x] =
                    fleet.flight_plan_.size();

                // 被撃墜可能性、実装面倒だしいいか・・・
                // for (const auto& [_, shipyard] : state.shipyards_) {
                //     const auto shipyard_away = shipyard.player_id_ !=
                //     player_id;
                // }

                n_ships[away] += fleet.ship_count_;
                sum_cargo[away] += fleet.kore_;
            }
            idx_local_features += 6;

            // 到達可能艦数 (自分、相手)
            // 経路にたまたまいるのは除く
            for (const auto& [_, shipyard] : state.shipyards_) {
                const auto [sy, sx] = shipyard.position_;
                const auto delta = i == 0
                                       ? shipyard.ship_count_
                                       : (short)(shipyard.last_ship_increment_ -
                                                 shipyard.last_ship_decrement_);
                const auto away = shipyard.player_id_ != player_id;
                for (auto n = 0; i + n < kFutureSteps; n++) {
                    if (n < kSize / 2) {
                        const auto cx = sx - n - 1 >= 0 ? sx : sx + kSize;
                        const auto cy = sy - n >= 0 ? sy + 1 : sy + 1 + kSize;
                        reachable[i + n][away][cy - n - 1][cx] += delta;
                        reachable[i + n][away][cy - n][cx] += delta;
                        reachable[i + n][away][cy][cx - n - 1] -= delta;
                        reachable[i + n][away][cy][cx - n] -= delta;
                        reachable[i + n][away][cy][cx + n + 1] -= delta;
                        reachable[i + n][away][cy][cx + n] -= delta;
                        reachable[i + n][away][cy + n][cx] += delta;
                        reachable[i + n][away][cy + n + 1][cx] += delta;
                    } else if (n < kSize - 1) {
                        reachable_all[i + n][away] += delta;
                        const auto cx = sx + kSize / 2;
                        auto cy = sy + kSize / 2 + 1;
                        if (sy == kSize - 1)
                            cy -= kSize;
                        const auto r = kSize - 1 - n;
                        reachable[i + n][away][cy - r][cx] -= delta;
                        reachable[i + n][away][cy - r][cx + 1] -= delta;
                        reachable[i + n][away][cy][cx - r] += delta;
                        reachable[i + n][away][cy][cx + r + 1] += delta;
                        reachable[i + n][away][cy + 1][cx - r] += delta;
                        reachable[i + n][away][cy + 1][cx + r + 1] += delta;
                        reachable[i + n][away][cy + r + 1][cx] -= delta;
                        reachable[i + n][away][cy + r + 1][cx + 1] -= delta;
                    } else {
                        reachable_all[i + n][away] += delta;
                    }
                }
            }

            for (auto p = 0; p < 2; p++) {
                // 累積 (左下へ)
                for (auto y = 0; y < kSize * 2 - 1; y++)
                    for (auto x = 1; x < kSize * 2; x++)
                        reachable[i][p][y + 1][x - 1] += reachable[i][p][y][x];
                // reachable_all を reachable にマージ
                for (auto x = kSize; x < kSize * 2; x++)
                    reachable[i][p][kSize][x] += reachable_all[i][p];
                for (auto y = kSize + 1; y < kSize * 2; y++)
                    reachable[i][p][y][kSize] += reachable_all[i][p];
                // 累積 (右下へ)
                for (auto y = 0; y < kSize * 2 - 1; y++)
                    for (auto x = 0; x < kSize * 2 - 1; x++)
                        reachable[i][p][y + 1][x + 1] += reachable[i][p][y][x];
                // 4 つに分かれてるのを 1 箇所に集める
                for (auto y = 0; y < kSize; y++)
                    for (auto x = 0; x < kSize; x++)
                        local_features[idx_local_features][y][x] =
                            reachable[i][p][y][x] +
                            reachable[i][p][y][x + kSize] +
                            reachable[i][p][y + kSize][x] +
                            reachable[i][p][y + kSize][x + kSize];
                idx_local_features++;
            }

            // global
            global_features[idx_global_features++] =
                state.players_[player_id].kore_;
            global_features[idx_global_features++] =
                state.players_[1 - player_id].kore_;
            global_features[idx_global_features++] = sum_cargo[0];
            global_features[idx_global_features++] = sum_cargo[1];
            global_features[idx_global_features++] = n_ships[0];
            global_features[idx_global_features++] = n_ships[1];
            global_features[idx_global_features++] =
                state.players_[player_id].shipyard_ids_.size();
            global_features[idx_global_features++] =
                state.players_[1 - player_id].shipyard_ids_.size();

            // 次のターンへ
            auto action = SpawnAgent().ComputeNextMove(state, 0);
            auto action1 = SpawnAgent().ComputeNextMove(state, 1);
            action.Merge(action1);
            assert(action1.actions.size() == 0);
            state = state.Next(action);
        }

        global_features[idx_global_features++] = step;

        assert(kNLocalFeatures == idx_local_features);
        assert(kNGlobalFeatures == idx_global_features);
    }
};

// =========================================

// 棋譜検証
#ifdef TEST_KORE_FLEETS
int main() {
    auto game = Game();
    game.ValidateKif(cin);
    cout << "ok" << endl;
}
#endif

// 特徴一覧の出力
#ifdef TEST_PRINT_NNUE_FEATURE_NAMES
int main() {
    NNUEFeature::SetFeatureNames();
    for (const auto& name : NNUEFeature::feature_names) {
        cout << name << endl;
    }
}
// clang-format off
// clang++ kore_fleets.cpp -std=c++17 -Wall -Wextra -O3 -DTEST_PRINT_NNUE_FEATURE_NAMES
// clang-format on
#endif

// TODO: root node 以外はもっと子の数を制限したほうがいいかも -> done
// TODO: 艦隊が合体する時、fleetの数に制限をつける
// - そもそも合体考えなくていいかも
// TODO: 艦隊が U ターンする時、拾う数にペナルティ？
// TODO: どうしても spawn が選ばれやすくなるのをなんとかしたい
