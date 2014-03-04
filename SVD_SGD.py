import collections
import numpy as np

C=100
k=0
step = 0.0001
epsilon = 0.001
d=122
n=6414
w = np.zeros((122))
b=0



x = np.zeros((n,d))
file = open('features.txt','r')
for index,line in enumerate(file):
	for i,elem in enumerate(line.split(',')):
		x[index][i] = float(elem)

y = np.zeros(n)

file = open('target.txt','r')
for index,line in enumerate(file):
	y[index]=float(line)

shuffled_x = np.empty(x.shape, dtype=x.dtype)
shuffled_y = np.empty(y.shape, dtype=y.dtype)
permutation = np.random.permutation(len(x))

for old_index, new_index in enumerate(permutation):
    shuffled_x[new_index] = x[old_index]
    shuffled_y[new_index] = y[old_index]

x = shuffled_x
y = shuffled_y

curr_error = 10000000000
w_init = 0
prev_fk = 0

for j in range(d):
	w_init+=w[j]**2

prev_int = 0
for i in range(n):
	inner_sum=0
	for j in range(d):
		inner_sum+=w[j]*x[i][j]
	prev_int+=max(0,1-y[i]*(inner_sum+b))
prev_fk = 0.5*w_init + C*prev_int
print prev_fk

i=0
while curr_error > epsilon:
	for j in range(d):
		gradient = 0
		confidence = y[i]*(np.dot(x[i],w)+b)
		if confidence >= 1:
			gradient = 0
		else:
			gradient = -1*y[i]+x[i][j]
		w[j] = w[j] - step*(w[j]+C*gradient)

	gradient_b = -1*y[i]*C
	b = b-step*gradient_b
	k=k+1
	i=i%n+1
	w_squared = 0
	errors = 0
	for j in range(d):
		w_squared+=w[j]**2

	for i in range(n):
		inner_sum=0
		for j in range(d):
			inner_sum+=w[j]*x[i][j]
		errors+=max(0,1-y[i]*(inner_sum+b))


	f_k = 0.5*w_squared + C*errors
	# print "ITERATION: ", k
	# print f_k
	curr_error = abs(((prev_fk)-f_k)/prev_fk*100)
	prev_fk = f_k
	print k, curr_error

print w
print k



