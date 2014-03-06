import collections
import numpy as np

C=100
k=0
step = 0.00001
epsilon = 0.01
d=122
n=6414
w = np.zeros((122))
b=0
batch_size = 20


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

curr_error = 5
prev_error = 5
init_errors = 0
for i in range(n):
	confidence = 1-y[i]*(np.dot(w,x[i])+b)
	init_errors+=max(0,confidence)


prev_fk = 0.5*sum(w**2) + C*init_errors 
print prev_fk

l=0
while curr_error > epsilon:
	w_prev = copy.copy(w)
	for j in range(d):
		gradient = 0
		for i in range(l*batch_size+1,min(n,(l+1)*batch_size)):
			confidence = y[l]*(np.dot(x[l],w_prev)+b)
			if confidence < 1:
				gradient += -1*y[l]*x[l][j]
		w[j] = w[j] - step*(w[j]+C*gradient)

	confidence = y[l]*(np.dot(x[l],w)+b)
	gradient_b = 0
	if confidence < 1:
		gradient_b = -1*y[l]*C
	else: 
		gradient_b = 0
	b = b-step*gradient_b
	k=k+1
	l = (l+1)%((n+batch_size-1)/batch_size)

	errors = 0
	for ii in range(n):
		confidence = 1-y[ii]*(np.dot(w,x[ii])+b)
		errors+=max(0,confidence)


	f_k = 0.5*sum(w**2) + C*errors 
	# print "ITERATION: ", k
	# print f_k
	curr_error = 0.5*abs(((prev_fk)-f_k)/prev_fk*100) + 0.5*prev_error
	prev_error = curr_error
	prev_fk = f_k
	print k, curr_error, f_k




