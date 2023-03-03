from matplotlib import pyplot as plt
import numpy as np
import pickle

def get_data(inputs_file_path):
    """
    Takes in an inputs file path and labels file path, loads the data into a dict, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy 
    array of labels). 
    :param inputs_file_path: file path for ONE input batch, something like 
    'cifar-10-batches-py/data_batch_1'
    :return: NumPy array of inputs as float32 and labels as int8
    """
    #TODO: Load inputs and labels
    with open(inputs_file_path, 'rb') as fo:
        #dict = pickle.load(fo, encoding='bytes')
        dict = pickle.load(fo, encoding='latin1')
    #TODO: Normalize inputs
    #get labels
    label1 = []
    label = np.array(dict['labels'], dtype=np.int8)
   
    #get data
    data = dict['data']
    
    #normalize data to range 0 - 1
    data_normalize = (data - 0.0) / (255.0 - 0.0)
    data_normalize = data_normalize.astype(np.float32)
    return data_normalize, label

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 3072 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 30  #batch size value from hyperparameter tuning
        self.learning_rate = 0.0065  #learning rate value from hyperparameter tuning

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.num_classes, self.input_size))  #initialize weight to 0
        self.b = np.zeros((self.num_classes, 1))  #initialize bias to 0

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        x = inputs
        # print('w shape: ', self.W.shape)
        # print('x shape: ', x.shape)
        # print('b shape: ', self.b.shape)
        dot_wx = np.dot(x, self.W.transpose())
        q = dot_wx.transpose() + self.b

        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        exp_sum = np.sum(np.exp(q), axis = 0) #sum of exponential, size 1 * classes 10
        softmax_q = np.exp(q) / exp_sum  #softmax function result 10 * batch_size
        softmax_q = softmax_q.astype(np.str_)  #type is string to keep number accuracy
        #print('forward done once!!!!!!!!!!!!!!!!!!!')
        softmax_qt = softmax_q.transpose()  #batch_size x 10
        return softmax_qt
        #pass
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step). 
        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        cross_entropy = 0.0
        i = 0
        for label in labels:
            y = np.zeros((self.num_classes,1))
            y[label] = 1  #set one hot vector
            cross_entropy = cross_entropy - np.dot(np.log(probabilities[i].astype(np.float32)), y)
            i = i + 1

        return (cross_entropy / self.batch_size)
        #pass
    
    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        #gradient = pi - yi
        pi = probabilities.astype(np.float32)  #batch_size x 10
        for i in range(inputs.shape[0]):
            pi[i][labels[i]] = pi[i][labels[i]] - 1 #since y is one hot vector
    
        gradientW = np.dot(inputs.transpose(), pi) / self.batch_size #average gradient of weight. 3072*10
        gradientB = np.sum(pi, axis=0) / self.batch_size  #average gradient of bias
        gradientB = gradientB.reshape((gradientB.shape[0],1))  #average gradient of bias 10 * 1
        return gradientW, gradientB
        #pass
    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        accuracy = 0.0
        #get argmax for model.forward() output
        predict_ind = np.argmax(probabilities, axis=1)
        i = 0
        for label in labels:
            if label == predict_ind[i]:
                accuracy = accuracy + 1
            i = i + 1
       
        return (accuracy / probabilities.shape[0])
        #pass

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W = self.W.transpose() - self.learning_rate * gradW
        self.W = self.W.transpose()
        self.b = self.b - self.learning_rate * gradB
        #pass
    
def train(model, train_inputs, train_labels):
#def train(model, train_inputs, train_labels, test_inputs, test_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_labels: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.
    start_num = 0
    sample_num = model.batch_size
    train_size = train_inputs.shape[0]
    iteration = np.ceil(train_size / model.batch_size)
    loss = []
    
    for i in range(int(iteration)):
        #get train data and label for current batch
        current_train = train_inputs[range(start_num, sample_num)]
        current_label = train_labels[range(start_num, sample_num)]
        start_num = start_num + model.batch_size
        sample_num = sample_num + model.batch_size
        #size for the last batch
        if sample_num > train_size:
            sample_num = train_size

        #train forward and get softmax probability
        softmax_prob = model.forward(current_train)
        gradient_weight, gradient_bias = model.compute_gradients(current_train, softmax_prob, current_label)
        model.gradient_descent(gradient_weight, gradient_bias)
        #append current loss per batch
        current_loss = model.loss(softmax_prob, current_label)
        #print('current loss per batch:?????????????????? ', current_loss)
        loss.append(current_loss)
        
    #visualize loss    
    visualize_loss(loss)
    #pass

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. 
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    accuracy = 0.0
    prob = model.forward(test_inputs)
    accuracy = model.accuracy(prob, test_labels)
    return accuracy
    #pass

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. You can call this in train() to observe.
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """

    plt.ion()
    plt.show()

    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses, color='r')
    plt.draw()
    plt.pause(0.001)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.forward()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    plt.ioff()

    images = np.reshape(image_inputs, (-1, 3, 32, 32))
    images = np.moveaxis(images, 1, -1)
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in CIFAR10 data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. 
    :return: None
    '''

    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    data, label = get_data('cifar-10-batches-py/data_batch_1')
    data2, label2 = get_data('cifar-10-batches-py/data_batch_2')
    data3, label3 = get_data('cifar-10-batches-py/data_batch_3')
    data4, label4 = get_data('cifar-10-batches-py/data_batch_4')
    data5, label5 = get_data('cifar-10-batches-py/data_batch_5')

    data = np.append(data, data2, axis=0)
    data = np.append(data, data3, axis=0)
    data = np.append(data, data4, axis=0)
    data = np.append(data, data5, axis=0)

    label = np.append(label, label2, axis=0)
    label = np.append(label, label3, axis=0)
    label = np.append(label, label4, axis=0)
    label = np.append(label, label5, axis=0)
    
    test_data, test_label = get_data('cifar-10-batches-py/test_batch')

    # #separate hyperparameter tuning function
    # perceptron2 = Model()
    # #take 40000 as training data and 10000 as validation data
    # train_data = data[range(10000, 50000)]
    # train_label = label[range(10000, 50000)]
    # valid_data = data[range(0, 10000)]
    # valid_label = label[range(0, 10000)]
    # valid_accuracy = 0.0
    # batch_tune = 0
    # learn_rate_tune = 0.0
    # accuracy_record = []
    # good_record = []

    # for batch in range(30, 150, 10):
    #     for learn_rate in np.arange(0.0005, 0.0205, 0.0005):
    #         print('batch & learn rate: ', batch, learn_rate)
    #         perceptron2.batch_size = batch
    #         perceptron2.learning_rate = learn_rate
    #         #train with train data
    #         train(perceptron2, train_data, train_label)
    #         #test with validation data
    #         current_valid_accuracy = test(perceptron2, valid_data, valid_label)
    #         print('valid_accuracy!!!!!!!!!!!!!!!!!!!!!: ', current_valid_accuracy)
    #         accuracy_record.append((batch, learn_rate, current_valid_accuracy))
    #         if current_valid_accuracy >= 0.39:
    #             good_record.append((batch, learn_rate, current_valid_accuracy))
    #         if current_valid_accuracy > valid_accuracy:
    #             valid_accuracy = current_valid_accuracy
    #             batch_tune = batch
    #             learn_rate_tune = learn_rate
    # print('final batch: !!!!!!!!!!!!!!!!', batch_tune)
    # print('final learn rate: !!!!!!!!!!!!!!!', learn_rate_tune)


    # TODO: Create Model
    perceptron = Model()

    #perceptron.batch_size = 30
    #perceptron.learning_rate = 0.0065

    # TODO: Train model by calling train() ONCE on all data
    train(perceptron, data, label)

    # TODO: Test the accuracy by calling test() after running train()
    test_accuracy = test(perceptron, test_data, test_label)
    print('test_accuracy: ', f"{test_accuracy:.4f}")

    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    sample_data = test_data[range(10, 20)]
    sample_label = test_label[range(10, 20)]
    prob = perceptron.forward(sample_data)
    visualize_results(sample_data, prob, sample_label)
    #pass
    
if __name__ == '__main__':
    main()
