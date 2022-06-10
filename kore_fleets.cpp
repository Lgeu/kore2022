#pragma once
#include "../marathon/library.hpp"
#include <cassert>
#include <cstddef>
#include <istream>
#include <map>
#include <set>
#include <string>

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

using Point = Vec2<signed char>;
using FleetId = short;
using ShipyardId = short;
using PlayerId = short;

template <> struct std::hash<Point> {
    size_t operator()(const Point& key) const { return key.y * 256 + key.x; }
};

enum struct Direction { N, E, S, W };
enum struct ShipyardActionType { kSpawn, kLaunch };

static auto IsClose(const double a, const double b) {
    return abs(a - b) <= max(abs(a), abs(b)) * 1e-9;
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

static Point GetColRow(const int pos) {
    return {(Point::value_type)(pos / kSize), (Point::value_type)(pos % kSize)};
}

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
                                : Point{v.x, 0};
    case Direction::W:
        return v.x != 0 ? Point{v.y, (Point::value_type)(v.x - 1)}
                        : Point{v.x, (Point::value_type)(kSize - 1)};
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
        direction_ = (Direction)direction_raw; // TODO: 確認
        if (strcmp(flight_plan_.c_str(), "null"))
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
        case ShipyardActionType::kLaunch:
            assert(num_ships > 0);
            assert(flight_plan_.size() > 0);
            assert(string("NESW").find(flight_plan[0]) != string::npos);
            for (const auto c : flight_plan)
                assert(string("NESWC0123456789").find(c) != string::npos);
            assert((int)flight_plan.size() <
                   Fleet::MaxFlightPlanLenForShipCount(num_ships));
            break;
        case ShipyardActionType::kSpawn:
            break;
        }
    }
    ShipyardAction(string raw) {
        if (raw.size() == 0) {
            assert(false);
        } else if (raw[0] == 'S') {
            // SPAWN
            ShipyardAction(ShipyardActionType::kSpawn, stoi(raw.substr(6)));
        } else {
            // LAUNCH
            const auto idx_underscore = raw.find('_', 8);
            ShipyardAction(ShipyardActionType::kLaunch,
                           stoi(raw.substr(7, idx_underscore - 7)),
                           raw.substr(idx_underscore + 1));
        }
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
    int ship_count_;
    Point position_;
    PlayerId player_id_;
    int turns_controlled_;

    Shipyard(const ShipyardId id, const int ship_count, const Point position,
             const PlayerId player_id, const int turns_controlled)
        : id_(id), ship_count_(ship_count), position_(position),
          player_id_(player_id), turns_controlled_(turns_controlled) {}

    Shipyard(const ShipyardId id, const PlayerId player_id, istream& is)
        : id_(id), player_id_(player_id) {
        Read(is);
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
        is >> ship_count_ >> position_raw >> turns_controlled_;
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
        return external_to_internal_.at(external_id);
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
};

struct State {
    int step_;
    array<Player, 2> players_;
    map<FleetId, Fleet> fleets_;
    map<ShipyardId, Shipyard> shipyards_;
    Board<Cell, kSize, kSize> board_;

    IdMapper<ShipyardId> shipyard_id_mapper_;
    IdMapper<FleetId> fleet_id_mapper_;
    FleetId next_fleet_id_;
    ShipyardId next_shipyard_id_;

    // State(const Observation& observation) {
    //     // TODO
    //     for (auto y = 0; y < kSize; y++) {
    //         for (auto x = 0; x < kSize; x++) {
    //             const auto kore = observation.kore[position.to_index(size)];
    //             // board_[{y, x}] = Cell(kore, -1, -1);
    //         }
    //     }
    //     for (PlayerId player_id = 0; player_id < 2; player_id++) {
    //         const auto& player_observation = observation.players[player_id];
    //         // players_[player_id] = {player_id,
    //         player_observation.player_kore}; for (const auto& fleet :
    //         player_observation.player_fleets) {
    //             AddFleet(fleet);
    //         }
    //         for (const auto& shipyard : player_observation.player_shipyards)
    //         {
    //             AddShipyard(shipyard);
    //         }
    //     }
    // }

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

    auto Same(const State& rhs) const {
        if (step_ != rhs.step_)
            return false;

        for (auto y = 0; y < kSize; y++) {
            for (auto x = 0; x < kSize; x++) {
                const auto& cell = board_[{y, x}];
                const auto& rhs_cell = rhs.board_[{y, x}];
                if (!IsClose(cell.kore_, rhs_cell.kore_))
                    return false;
                if ((cell.fleet_id_ == -1) != (rhs_cell.fleet_id_ == -1))
                    return false;
                if (cell.fleet_id_ != -1 &&
                    !fleets_.at(cell.fleet_id_)
                         .Same(rhs.fleets_.at(rhs_cell.fleet_id_)))
                    return false;

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

    void DeleteFleet(const Fleet& fleet) {
        players_[fleet.player_id_].fleet_ids_.erase(fleet.id_);
        if (board_[fleet.position_].fleet_id_ == fleet.id_) {
            board_[fleet.position_].fleet_id_ = -1;
        } else {
            cerr << "DeleteFleet: そんなことある？" << endl;
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
        // auto n_fleets = (short)0; // あーこれだめじゃん TODO
        // auto n_shipyards = (short)0;
        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            auto& player = state.players_[player_id];
            for (const auto& shipyard_id : player.shipyard_ids_) {
                auto& shipyard = state.shipyards_.at(shipyard_id);
                const auto& shipyard_action = action.actions.at(shipyard_id);
                if (shipyard_action.num_ships_ == 0) {
                    continue;
                } else if (shipyard_action.type_ ==
                           ShipyardActionType::kSpawn) {
                    if (player.kore_ >=
                            kSpawnCost * shipyard_action.num_ships_ &&
                        shipyard_action.num_ships_ <= shipyard.MaxSpawn()) {
                        player.kore_ -= kSpawnCost * shipyard_action.num_ships_;
                        shipyard.ship_count_ += shipyard_action.num_ships_;
                    } else {
                        cerr << "Next: その Spawn は無理" << endl;
                    }
                } else if (shipyard_action.type_ ==
                           ShipyardActionType::kLaunch) {
                    if (shipyard.ship_count_ >= shipyard_action.num_ships_) {
                        const auto& flight_plan = shipyard_action.flight_plan_;
                        shipyard.ship_count_ -= shipyard_action.num_ships_;
                        const auto& direction = CharToDirection(flight_plan[0]);
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
        // for (auto&& [_, fleet] : state.fleets_) {
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
                state.DeleteFleet(fleet);
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
                state.DeleteFleet(f1);
                return fid2;
            } else {
                f1.kore_ += f2.kore_;
                f1.ship_count_ += f2.ship_count_;
                state.DeleteFleet(f2);
                return fid1;
            }
        };

        // プレイヤーごと場所ごとに fleet を集める
        for (PlayerId player_id = 0; player_id < 2; player_id++) {
            const auto& player = state.players_[player_id];
            auto fleets_by_loc = unordered_map<Point, vector<FleetId>>();
            for (const auto& fleet_id : player.fleet_ids_)
                fleets_by_loc.at(state.fleets_.at(fleet_id).position_)
                    .push_back(fleet_id);
            for (auto&& [_, value] : fleets_by_loc) {
                // これソートいらないかと思いきや 3 つ同時とかだと必要になる
                sort(value.begin(), value.end(),
                     [&state](const FleetId& l, const FleetId& r) {
                         return state.fleets_.at(r).LessThanOtherAlliedFleet(
                             state.fleets_.at(l));
                     });
                const auto fid = value[0];
                for (auto i = 0; i < (int)value.size(); i++) {
                    const auto res_fid = combine_fleets(fid, value[i]);
                    assert(fid == res_fid);
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
            }
            for (const auto& fleet_id : deleted) {
                const auto fleet = state.fleets_.at(fleet_id);
                state.DeleteFleet(fleet);
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
                    state.DeleteFleet(fleet);
                } else {
                    shipyard.ship_count_ -= fleet.ship_count_;
                    state.players_[shipyard.player_id_].kore_ += fleet.kore_;
                    state.DeleteFleet(fleet);
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
                state.DeleteFleet(fleet);
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
                state.DeleteFleet(fleet);
            } else {
                fleet.ship_count_ -= damage;
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
                    cell.kore_ =
                        round(cell.kore_ * (1.0 + kRegenRate) * 1e3) * 1e-3;
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
    Action ComputeNextMove(const State& state) const {
        // TODO
        return Action();
    }
};

// 対戦するときのインタフェース、チェックするときの、次の一手するときの
struct Game {
    State state;
    Agent agent;

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
            const auto input_state = State().Read(is);
            assert(state.Same(input_state));
        }
    }

    // 対戦する
    void Match() {
        while (true) {
            state.Read(cin);
            auto action = agent.ComputeNextMove(state);
            action.Write(state.shipyard_id_mapper_, cout);
        }
    }
};

// =========================================

// auto PopulateBoard() {
//     auto seed = 3253151351u;

//     constexpr auto half = (kSize + 1) / 2;
//     auto grid = Board<double, kSize, kSize>();
//     grid.Fill(0.0);
//     static auto rng = Random(seed);
//     for (auto i = 0; i < half; i++) {
//         grid[{rng.randint(0, half), rng.randint(0, half)}] = i * i;
//         grid[{rng.randint(half / 2, half), rng.randint(half / 2, half)}] =
//             i * i;
//     }
//     auto radius_grid = grid;
//     for (auto r = 0; r < half; r++) {
//         for (auto c = 0; c < half; c++) {
//             const auto value = grid[{r, c}];
//             if (value == 0)
//                 continue;
//             const auto radius = min((int)round(sqrt(value / half)), 1);
//             if (radius < 1)
//                 continue;
//             radius_grid[{r, c}] = grid[{r, c}];
//         }
//     }
//     // TODO
// }

// 棋譜検証
int main() {
    auto game = Game();
    game.ValidateKif(cin);
    cout << "ok" << endl;
}
