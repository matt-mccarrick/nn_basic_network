from numpy import exp, array, random, dot

class nn():
	def __init__(self):
		#Seed rng for debug
		random.seed(1)
		
		#model 3 inputs and 1 output
		#assign random weight to 3x1 matrix. Values in range from -1 to 1. Mean 0
		self.synaptic_weights = 2 * random.random((3,1)) - 1
	
	#Sigmoid function definition. S curve.
	#Takes weighted sum of inputs, it normalizes them between 0 and 1.
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))
	
	#Sigmoid derivative. Gets gradient of sigmoid curve. 
	#Marker of confidence of weight
	#The more confidence (x), the less slope
	def __sigmoid_derivative(self, x):
		return x * (1 - x)
	
	#Train network. Adjust synaptic weight on each training session.
	def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
		for iteration in xrange(number_of_iterations):
			#pass training set through network (we only have one node)
			output = self.think(training_set_inputs)
			
			#calculate error (diff between expected output and predicted output)
			error = training_set_outputs - output
			
			#dot product of the inputs with the product of the error and sig derivative
			#less confident weights adjusted more
			#zero inputs don't change the weight
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			
			#adjust weight
			self.synaptic_weights += adjustment
	
	def think(self, inputs):
		#pass dot product of the inputs with the current weights
		#applies the weighting to the choice we're making on the inputs
		return self.__sigmoid(dot(inputs, self.synaptic_weights))
	
if __name__ == "__main__":

	#init network
	neural_network = nn()
	
	#starting weights
	print "Random starting weights: "
	print neural_network.synaptic_weights
	
	#Input training set/output set
	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T
	
	#Train network using training set. Do it 12k times, adjusting weights as we go
	neural_network.train(training_set_inputs, training_set_outputs, 12000)
	
	print "New weights: "
	print neural_network.synaptic_weights
	
	#Test with a new input
	print "New problem [1, 0, 0]: "
	print neural_network.think(array([1,0,0]))
	
