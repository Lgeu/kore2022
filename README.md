## ルール和訳

### 行動

#### Spawn Ship / 船の建造
- SPAWN_X
- コスト: 10 kore / ship
- 造船所に X 隻の船を追加
- 造船所の支配期間に応じて造船できる数が増加

-----
|支配期間|0|2|7|17|34|60|97|147|212|294|
|-|-|-|-|-|-|-|-|-|-|-|
|最大造船数|1|2|3|4|5|6|7|8|9|10|
-----

#### Launch / 出撃
- LAUNCH_X_<FLIGHT_PLAN>
- コスト: 0 kore
- 飛行計画に従って X 隻の艦隊を出撃させる

#### 飛行計画
- 出撃する艦隊の進路を決定するもの
- 最大長は floor(2 * ln(num_ships)) + 1
- 長さは文字列の長さ len(flight_plan) で測られる
- 一度出撃したら、すべての敵から見えるようになる
- 1 ターンに 1 度の更新以外で変更できない

-----
|艦数|1|2|3|5|8|13|21|34|55|89|...|
|-|-|-|-|-|-|-|-|-|-|-|-|
|最大計画長|1|2|3|4|5|6|7|8|9|10|...|
-----

#### 

ln(num_ships_in_fleet) / 20 % の kore を拾う


## 棋譜フォーマット

全体
```
<フォーマットバージョン:string>
<棋譜ID:string>
<エージェント1の情報:string>
<エージェント2の情報:string>
===
<step 0>
---
<step 1>
...
---
<最終step>
===
<エージェント1のreward> <エージェント2のreward>
```
各stepのフォーマット
```
<エージェント1の前ステップの造船所の数:int>
<エージェント1の造船所1のID> <行動:string>
...
<エージェント1の造船所nのID> <行動:string>
<エージェント2の...>
<step:int>
<盤面のkore:float[21,21]>
<エージェント1の残り時間:float>
<エージェント1のkore:float>
<エージェント1の造船所の数:int>
<エージェント1の造船所1のID:string> <座標:int> <保有艦数:int> <支配ターン数:int>
...
<エージェント1の造船所1のID:string> <座標:int> <保有艦数:int> <支配ターン数:int>
<エージェント1の艦隊の数:int>
<エージェント1の艦隊1のID:string> <座標:int> <kore:float> <艦数:float> <方向:int> <残りの命令:string>
...
<エージェント1の艦隊nのID:string> <座標:int> <kore:float> <艦数:float> <方向:int> <残りの命令:string>
<エージェント2の...>
```

step 0 には行動は無く、最終 step には行動がある。

## boost のビルドとコンパイル

```
./bootstrap.sh --with-toolset=clang
./b2 install -j4 --with-python -d0 cxxflags=-fPIC cflags=-fPIC
clang++ -std=c++17 -Wall -Wextra -O3 --shared -fPIC kore_extension.cpp -o kore_extension.so -I/home/user/anaconda3/include/python3.8 /usr/local/lib/libboost_numpy38.a /usr/local/lib/libboost_python38.a -lpython3.8 -L/home/user/anaconda3/lib -march=broadwell -l:libblas.so.3
```
