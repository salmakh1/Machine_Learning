import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x,self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        prod=self.run(x)
        if nn.as_scalar(prod)>=0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size=1
        c = True
        while c == True:
            c = False
            for x, y in dataset.iterate_once(batch_size):
                ypred= self.get_prediction(x)
                if ypred != nn.as_scalar(y):
                    self.get_weights().update(x, nn.as_scalar(y))
                    c = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1=nn.Parameter(1,20)
        self.b1=nn.Parameter(1,20)
        self.w2 = nn.Parameter(20, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w1)
        predicted_y1 =nn.ReLU(nn.AddBias(xw1, self.b1))
        xw2=nn.Linear(predicted_y1,self.w2)
        predicted_y2=nn.AddBias(xw2,self.b2)
        return predicted_y2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred=self.run(x)
        loss = nn.SquareLoss(y_pred, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size=20
        l = float("inf")
        while l > 0.0100:
            for x, y in dataset.iterate_once(batch_size):
                l=self.get_loss(x,y)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(l, [self.w1, self.b1,self.w2, self.b2])
                self.w1.update(grad_wrt_w1,-0.01)
                self.b1.update(grad_wrt_b1, -0.01)
                self.w2.update(grad_wrt_w2, -0.01)
                self.b2.update(grad_wrt_b2, -0.01)
                l = nn.as_scalar(self.get_loss(x, y))

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        y_pred1=nn.ReLU(nn.AddBias((nn.Linear(x,self.w1)),self.b1))
        y_pred2=nn.ReLU(nn.AddBias((nn.Linear(y_pred1,self.w2)),self.b2))
        y_pred3=nn.AddBias((nn.Linear(y_pred2,self.w3)),self.b3)
        return y_pred3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        y_pred = self.run(x)
        loss = nn.SoftmaxLoss(y_pred, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc= -float("inf")
        while acc < 0.975:
            for x, y in dataset.iterate_once(50):
                l=self.get_loss(x,y)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 ,grad_wrt_w3, grad_wrt_b3 = nn.gradients(l, [self.w1, self.b1,self.w2, self.b2,self.w3, self.b3])
                self.w1.update(grad_wrt_w1,-0.1)
                self.b1.update(grad_wrt_b1, -0.1)
                self.w2.update(grad_wrt_w2, -0.1)
                self.b2.update(grad_wrt_b2, -0.1)
                self.w3.update(grad_wrt_w3, -0.1)
                self.b3.update(grad_wrt_b3, -0.1)
            acc=dataset.get_validation_accuracy()




class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        self.w1= nn.Parameter(state_dim, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, action_dim)
        self.b3 = nn.Parameter(1, action_dim)
        self.parameters=[self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        self.numTrainingGames = 7000
        self.batch_size = 20

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_predicted3=self.run(states)
        loss = nn.SquareLoss(Q_predicted3, Q_target)
        return loss

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

        Q_predicted1 = nn.ReLU(nn.AddBias((nn.Linear(states, self.w1)), self.b1))
        Q_predicted2 = nn.ReLU(nn.AddBias((nn.Linear(Q_predicted1, self.w2)), self.b2))
        Q_predicted3 = nn.AddBias((nn.Linear(Q_predicted2, self.w3)), self.b3)
        return Q_predicted3

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        l = self.get_loss(states, Q_target)
        grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3 = nn.gradients(l,
                                                                                                    [self.w1, self.b1,
                                                                                                     self.w2, self.b2,
                                                                                                     self.w3, self.b3])
        self.w1.update(grad_wrt_w1, -0.1)
        self.b1.update(grad_wrt_b1, -0.1)
        self.w2.update(grad_wrt_w2, -0.1)
        self.b2.update(grad_wrt_b2, -0.1)
        self.w3.update(grad_wrt_w3, -0.1)
        self.b3.update(grad_wrt_b3, -0.1)

        "*** YOUR CODE HERE ***"