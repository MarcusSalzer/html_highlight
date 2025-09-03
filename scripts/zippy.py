a = [1, 2, 3]
b = [4, 5, 6]

z = list(zip(a, b))
print(f"{z = }")

u, v = zip(*z)
print(f"{u = }")
print(f"{v = }")
