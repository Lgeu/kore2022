# 戻ってくるパターンの数は？
# W が 1 回だけだったり最後だったりするからややこしい
# 造船所が増えるとそれを行き来するのも考える必要がある・・・

max_length = 8
cnt = 0

def dfs(command="", command_sep=[], x=0, y=0):
    global cnt
    if len(command) >= max_length:
        return
    if command:
        new_command = None
        if x == 0:
            if y > 0:
                new_command = command + "N"
            else:
                new_command = command + "S"
        elif y == 0:
            if x > 0:
                new_command = command + "W"
            else:
                new_command = command + "E"
        if new_command is not None:
            print(new_command)
            cnt += 1
    last_direction = command_sep[-1][0] if command else None
    for c, dx, dy in zip("NESW", [0, 1, 0, -1], [-1, 0, 1, 0]):
        if c == last_direction:
            continue
        for i in range(10):
            ux = x + dx * (i + 1)
            uy = y + dy * (i + 1)
            if command:
                if x == 0:
                    if y * uy <= 0:
                        break
                elif y == 0:
                    if x * ux <= 0:
                        break
            new_command = command + c + ("" if i == 0 else str(i))
            new_command_sep = command_sep + [[c, i]]
            dfs(new_command, new_command_sep, ux, uy)
        
dfs()
print(f"{cnt=}")
