import re
from pathlib import Path

with open(Path(__file__).parent / "kore_features.txt") as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
dict_name_to_index = {name: i for i, name in enumerate(lines)}

re_pattern = re.compile(r"(.*)y(-?\d\d?)_x(-?\d\d?)(.*)")

def f1(y, x):
    return x, y

def f2(y, x):
    return -y, -x

def get_shipyard_feature_converter(f):
    shipyard_feature_converter = []
    for i, name in enumerate(lines):
        if name == "":
            shipyard_feature_converter.append(i)
            continue
        l, y, x, r = re.fullmatch(re_pattern, name).groups()
        y, x = f(int(y), int(x))
        inverted_name = f"{l}y{y}_x{x}{r}"
        inverted_name_index = dict_name_to_index[inverted_name]
        shipyard_feature_converter.append(inverted_name_index)
    shipyard_feature_converter.append(len(shipyard_feature_converter))
    for i in range(len(lines)):
        assert i == shipyard_feature_converter[shipyard_feature_converter[i]]
    return shipyard_feature_converter

shipyard_feature_converter_f1 = get_shipyard_feature_converter(f1)
shipyard_feature_converter_f2 = get_shipyard_feature_converter(f2)
# print(shipyard_feature_converter_f1[-30:])
# print(shipyard_feature_converter_f2[-30:])
assert shipyard_feature_converter_f1 != shipyard_feature_converter_f2

def get_direction_converter(f):
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    converter = []
    for d in directions:
        converter.append(directions.index(f(*d)))
    for i in range(4):
        assert i == converter[converter[i]]
    return converter

direction_converter_f1 = get_direction_converter(f1)
direction_converter_f2 = get_direction_converter(f2)
#print(direction_converter_f1, direction_converter_f2)
assert direction_converter_f1 != direction_converter_f2

def get_relative_position_converter(f):
    converter = []
    for y in range(21):
        for x in range(21):
            fy, fx = f(y if y <= 10 else y - 21, x if x <= 10 else x - 21)
            assert abs(fy) <= 10
            assert abs(fx) <= 10
            fy %= 21
            fx %= 21
            converter.append(fy * 21 + fx)
    for i in range(21 * 21):
        assert(i == converter[converter[i]])
    return converter

relative_position_converter_f1 = get_relative_position_converter(f1)
relative_position_converter_f2 = get_relative_position_converter(f2)
# print(relative_position_converter_f1)
# print(relative_position_converter_f2)
assert relative_position_converter_f1 != relative_position_converter_f2
