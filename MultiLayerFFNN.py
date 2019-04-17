from numpy import exp, array, random, dot, mean, abs

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same number every time the program runs
        random.seed(1)

        # This model has 3 layers; 1. Input layer 2. hidden layer 3. output layer
        # We assign random weights with values in the range of -1 to 1 and mean 0
        self.syn0 = 2 * random.random ((3,4)) - 1
        self.syn1 = 2 * random.random((4,4)) - 1
        self.syn2 = 2 * random.random((4,1)) - 1

    #Activation Function (Sigmoid), which describes an s shaped curve
    # we pass the weighted sum of the inputs through This function to normalise them between 0 and 1
    def __sigmoid(self, x):
        return 1 /(1 + exp(-x))

    #gradient of the sigmoid curve - determine direction (positive or negative)
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    #Training function
    def train(self, X, Y, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #pass the training set through our neural net to create predictions
            l0 = X
            l1 = self.predict(l0, self.syn0)
            l2 = self.predict(l1, self.syn1)
            l3 = self.predict(l2, self.syn2)

            # For each layer, determine error (i.e. how much did we miss the target value?)
            # then multiply the error by the input and again by the gradient of the Sigmoid curve
            l3_error = Y - l3
            l3_delta = l3_error * self.__sigmoid_derivative(l3)

            l2_error = l3_delta.dot(self.syn2.T)
            l2_delta = l2_error * self.__sigmoid_derivative(l2)

            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * self.__sigmoid_derivative(l1)

            #admust the synaptic_weights
            self.syn2 += l2.T.dot(l3_delta)
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)

        #print out the final error for each layer
        print("\nl3 Error: " + str(mean(abs(l3_error))))
        print("l2 Error: " + str(mean(abs(l2_error))))
        print("l1 Error: " + str(mean(abs(l1_error))))

        #return the final prediction after training
        return l3

    #Prediction Function
    def predict(self, inputs, weight):
        #pass inputs through our neural network
        return self.__sigmoid(dot(inputs, weight))

    # Use trained neural network to make a prediction.
    def think(self, inputs):
        # Pass inputs through our neural network
        pred1 = self.__sigmoid(dot(inputs, neural_network.syn0))
        pred2 = self.__sigmoid(dot(pred1, neural_network.syn1))
        pred3 = self.__sigmoid(dot(pred2, neural_network.syn2))
        return pred3

if __name__ == "__main__":

    #initialize  NeuralNetwork
    neural_network = NeuralNetwork()

    #The training set. We have 4 examples, each consisting of 3 input values and i output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    #train the neural network using a training set.
    #Do it 60,000 times and make small adjustments each times
    prediction = neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print('\nOutput After Training:')
    print(prediction)

    #Test the neural network
    print("\nConsidering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0])))
    print("\n")
