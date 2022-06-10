with open("round.txt", "w") as f:
    print("{", end="", file=f)
    for a in range(800):
        for b in range(10):
            d = f"{a}.{b}25"
            d = float(d)
            d *= 1 + 0.02
            d = round(d, 4)
            print(f"{{{d},{round(d, 3)}}},", end="", file=f)

            d = f"{a}.{b}75"
            d = float(d)
            d *= 1 + 0.02
            d = round(d, 4)
            print(f"{{{d},{round(d, 3)}}},", end="", file=f)
    print("}", file=f)
