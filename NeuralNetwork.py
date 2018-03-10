import numpy
import scipy.misc
import glob
from scipy.special import expit as sigmoid
from scipy.ndimage import rotate
import os


class deepFeedForward(object):

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, activation_function):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: activation_function(x)

    def train(self, inputs_list, targets_list):
        """This function queries the Neural Net and refines the weights according to the proporationate error"""

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        """This function queries the Neural Net and calculates the output from the input parameter according to the current weights"""

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # get weights from the neural network
    def get_weights(self):
        """This function returns the weights from the Neural Net to be exported"""
        return self.wih, self.who

    # set weights for neural network
    def set_weights(self, wih, who):
        """This function allows you to set the Neural Net's weights manually"""
        self.wih = wih
        self.who = who


if __name__ == "__main__":
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate
    learning_rate = 0.1

    # how often to train -> epochs
    epochs = 10

    # training targets
    t = [0.5, 0.5, 0.5]

    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # create instance of neural network
    neuralNetwork = deepFeedForward(input_nodes, hidden_nodes, output_nodes, learning_rate, sigmoid)

    # check if pre-trained weights are available
    mypath = os.path.dirname(os.path.realpath(__file__)) + '\\data'
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    fname_wih = 'data/wih.txt'
    fname_who = 'data/who.txt'
    if os.path.isfile(fname_wih) and os.path.isfile(fname_who):
        pre_trained = True
    else:
        pre_trained = False

    # train neural network and refine weights if trained weights not available
    if not pre_trained:
        # train the neural networkand refine weights
        for e in range(epochs):
            print('training')
            # go through all records in the training data set
            for record in training_data_list:
                # split the record by the ',' commas
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                inputs_plus10 = rotate(inputs.reshape(28,28), 10.0, cval=0.01, order=1, reshape=False).reshape(784)
                inputs_minus10 = rotate(inputs.reshape(28,28), -10.0, cval=0.01, order=1, reshape=False).reshape(784)
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                neuralNetwork.train(inputs, targets)
                neuralNetwork.train(inputs_plus10, targets)
                neuralNetwork.train(inputs_minus10, targets)
            print('training done for epoch ', e)

        wih, who = neuralNetwork.get_weights()
        numpy.savetxt(fname_wih, wih)
        numpy.savetxt(fname_who, who)
    # pre-trained weights available
    else:
        neuralNetwork.set_weights(numpy.loadtxt(fname_wih), numpy.loadtxt(fname_who))

    # our own image test data set
    our_own_dataset = []

    # how many records to test (incremented automatically)
    our_own_records = 0

    # load the png image data as test data set
    for image_file_name in glob.glob('my_own_images/2828_my_own_?.png'):
        our_own_records += 1
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])
        print(label, image_file_name)
        # print('label = ',label)

        # load image data from png files into an array
        print("loading ... ", image_file_name)
        img_array = scipy.misc.imread(image_file_name, flatten=True)

        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)

        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        # print(numpy.min(img_data))
        # print(numpy.max(img_data))

        # append label and image data  to test data set
        record = numpy.append(label, img_data)
        our_own_dataset.append(record)

    # test the neural network with our own images
    # records to test
    items = range(our_own_records)

    # plot image
    # matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28, 28), cmap='Greys', interpolation='None')

    # correct answer is first value
    for item in items:
        correct_label = our_own_dataset[item][0]
        # data is remaining values
        inputs = our_own_dataset[item][1:]

        # query the network
        outputs = neuralNetwork.query(inputs)
        # print(outputs)

        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print("network says ", label)
        # append correct or incorrect to list
        if (label == correct_label):
            print("match!")
        else:
            print("no match!, correct label is", correct_label)