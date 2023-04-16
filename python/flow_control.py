# python3

array0 = ["Federica","Marzio","Laura","Arianna","Viktor","Joan"]
for name in array0:
    print(name)

print("=======================================")
for name in array0:
    print(name)
    if name == "Joan": print("voila! that's me")

print("=========================================")
idx = 0
while array0[idx] != "Laura":
    print(array0[idx])
    idx += 1

print("=======================================")
print("that's Dr.{}, index {} ".format(array0[idx],idx))
