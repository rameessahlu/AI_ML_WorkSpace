def activation(yin):
	if yin > 0:
		return 1
	elif yin == 0:
		return 0
	else: #yin < 0
		return -1

def training(inputs_and_targets):
	#The initial weights and threshold are set to zero
	w1 = w2 = b = theta = 0
	#The learning rate is set equal to 1
	a = 1
	
	number_of_epochs = 0
	training = True
	while training:
		calculated_outputs = []
		targets = []
		number_of_epochs = number_of_epochs + 1
		
		for i_n_t in inputs_and_targets:
			#calculate the net input
			x1 = i_n_t[0]
			x2 = i_n_t[1]
			t  = i_n_t[2]
			yin = b + (x1*w1) + (x2*w2)
			
			#output y is calculated by applying activations over the net input calculated(yin)
			y = activation(yin)
			
			calculated_outputs.append(y)
			targets.append(t)
			
			if y != t:
				#wi(new) = wi(old) + atxi
				w1 = w1 + (a*t*x1)
				w2 = w2 + (a*t*x2)
				#b(new) = b(old) * at 
				b = b + (a*t)
		
		#Training will be stopped if all the targets become equal to the calculated outputs
		if targets == calculated_outputs:
			training = False
	return w1, w2, b, number_of_epochs


#truth table for AND function with bipolar inputs and targets
inputs_and_targets = [
			[1, 1, 1],
			[1, -1, -1],
			[-1, 1, -1],
			[-1, -1, -1]
]
#The final weights and bias after final epoch
w1, w2, b, number_of_epochs = training(inputs_and_targets)
print("w1 = {}, w2 = {}, b = {}, number of epochs = {}".format(w1, w2, b, number_of_epochs))
