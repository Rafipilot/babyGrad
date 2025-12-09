import main

a = main.Tensor(3)

b = main.Tensor(7)

c = a*b

print(a)
print(b)
print(c)

c.backward()

print(a)
print(b)
print(c)
