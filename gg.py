import mnist_loader
training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
import network
net = network.NeuralNetwork([784, 30,15, 10])
net.SGD(training_data, 30, 20, 0.25, test_data=test_data)
 