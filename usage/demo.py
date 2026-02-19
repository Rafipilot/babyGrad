import babyGrad.grad as grad

a = grad.Tensor(2.0)

b = grad.Tensor(3.0)

c = a * b

print(c)

c.backward()

print(a.grad)  # should print 3.0
print(b.grad)  # should print 2.0
