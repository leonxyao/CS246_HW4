import numpy as np

x = np.zeros((3,5))
for index in range(len(x)):
	 x[index] = np.random.randn(5)

y = np.random.randn((3))
print "X ORIGINAL: "
print x
print "Y ORIGINAL: "
print y

shuffled_x = np.empty(x.shape, dtype=x.dtype)
shuffled_y = np.empty(y.shape, dtype=y.dtype)
permutation = np.random.permutation(len(x))

for old_index, new_index in enumerate(permutation):
    shuffled_x[new_index] = x[old_index]
    shuffled_y[new_index] = y[old_index]

x = shuffled_x
y = shuffled_y

print "X SHUFFLED: "
print x
print "Y SHUFFLED: "
print y