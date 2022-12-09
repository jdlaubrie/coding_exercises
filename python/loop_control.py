
# python3

array0 = ["Federica","Marzio","Laura","Arianna","Viktor","Joan"]
for name in array0:
    print(name)

for name in array0:
    if name == "Joan": print("voila! that's me")

idx = 0
while array0[idx] != "Laura":
    idx += 1

print("That's Doctor {}".format(array0[idx]))
