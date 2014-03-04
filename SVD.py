import collections
import numpy as np
import random
import copy

C=100
k=0
step = 0.0000003
epsilon = 0.25
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


while curr_error > epsilon:
	w_prev = copy.copy(w)
	for j in range(d):
		gradient = 0
		for i in range(n):
			confidence = y[i]*(np.dot(x[i],w_prev)+b)
			if confidence >= 1:
				gradient += 0
			else:
				gradient += -1*y[i]+x[i][j]
		w[j] = w[j] - step*(w[j]+C*gradient)
	gradient_b = 0
	for i in range(n):
		gradient_b += y[i]
	gradient_b*=-1*d*C
	b = b-step*gradient_b
	k=k+1
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










