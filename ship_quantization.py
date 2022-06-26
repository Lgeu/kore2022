import math
from tkinter import N

def max_flight_plan_length(n_ships):
    return int(2 * math.log(n_ships)) + 1

efficient_n_ships = set(n_ships for n_ships in range(2, 500) if max_flight_plan_length(n_ships) > max_flight_plan_length(n_ships - 1))

MAX_QUANTIZED = 30

quantization_table = []

for n_ships in range(500):
    if n_ships < 10:
        quantized = n_ships
    else:
        quantized = max(0, int(4 * math.log(n_ships)) - 2) * 3 // 2
        if n_ships in efficient_n_ships:
            quantized -= 1
    if quantized > MAX_QUANTIZED:
        break
    quantization_table.append(quantized)

table_length = len(quantization_table)

dequantization_table = [None for i in range(MAX_QUANTIZED + 2)]
dequantization_table[-1] = table_length
for n, q in enumerate(quantization_table):
    if dequantization_table[q] is None:
        dequantization_table[q] = n

with open("n_ships_quantization_table.cpp", "w") as f:
    print("//                                                                " + " ".join(f"{i:3d}" for i in range(table_length)), file=f)
    print(f"static constexpr array<signed char, {table_length}> kNShipsQuantizationTable{{" + ",".join(f"{q:3d}" for q in quantization_table) + "};", file=f)
    print(file=f)
    print("//                                                    " + "".join(f"{i:4d}" for i in range(MAX_QUANTIZED + 2)), file=f)
    print(f"static constexpr array<short, {MAX_QUANTIZED + 3}> kDequantizationTable{{" + ",".join(f"{d:3d}" for d in dequantization_table + [999]) + "};", file=f)
