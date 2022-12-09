# python3

#first interaction with python
print("hello world!")

#playing with arrays
print("=============== ARRAYS ==============")
array0 = ["Elisa", "Patrizia", "Giulio", "Marzio"]
print(array0)
print(array0[2])
array0.append("Enrica")
print(array0)
array0.remove("Marzio")
print(array0)
n_array0 = len(array0)
print(n_array0)

#playing with dictionaries
print("=============== DICTIONARIES ==============")
dict0 = {"Name":"Joan", "Position":"PostDoc", "Office":207}
print(dict0)
print(dict0["Office"])
dict0 = {"Name":"Joan", "Position":"PostDoc", "Office":207, "Office":206}
print(dict0)
print(type(dict0))
print(type(dict0["Office"]))
dict1 = dict(name="Elisa",position="PhD Student",office=201)
print(dict1)
print(dict1.keys())
print(dict1.values())
dict2 = {}
print(dict2)
dict2["name"] = "Giulio"
dict2["position"] = "Master Student"
print(dict2.items())
dict2["Office"] = 00
print(dict2.items())
dict0.pop("Office")
dict1.pop("office")
dict2.pop("Office")
print(dict0)
print(dict1)
print(dict2)

#playing with tuples
tuple0 = (dict0,dict1,dict2)
print(tuple0)

