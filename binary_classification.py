import numpy as np
import scipy
class l_layer_neural_network:

    def __init__(self,layers):
        self.layers = layers
        self.L = len(self.layers)
        self.parameters = {}
        self.caches = []

    def sigmoid(self,Z):
        cache = Z
        A = 1/(1+np.exp(-Z))
        return A,cache

    def relu(self,Z):
        cache = Z
        A = np.maximum(0,Z)
        return A,cache

    def sigmoid_backward(self,dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA*s*(1-s)
        return dZ

    def relu_backward(self,dA, cache):
        Z = cache
        dZ = np.array(dA,copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def initialize_parameters(self):
        for l in range(1,self.L):
            self.parameters['W'+str(l)] = np.random.randn(self.layers[l],self.layers[l-1])*0.01
            self.parameters['b'+str(l)] = np.zeros((self.layers[l],1))

            assert(self.parameters['W'+str(l)].shape == (self.layers[l],self.layers[l-1])),"Parameters W dimension do not match"
            assert(self.parameters['b'+str(l)].shape == (self.layers[l],1)),"Parameters b dimension do not match"

    def linear_forward(self,A,W,b):
        Z = np.dot(W,A)+b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A,W,b)
        return Z,cache

    def linear_activation_forward(self,A_prev,W,b,activation):
        if activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = self.relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def total_forward(self,X):
        self.caches = []
        A = X
        L = self.L
        m = X.shape[1]
        for l in range(1,L-1):
            A_prev = A
            W = self.parameters['W'+str(l)]
            b = self.parameters['b'+str(l)]
            A, cache = self.linear_activation_forward(A_prev,W,b,'relu')
            self.caches.append(cache)
            # print(np.array(self.caches).shape)
        W = self.parameters['W'+str(L-1)]
        b = self.parameters['b'+str(L-1)]
        AL, cache = self.linear_activation_forward(A,W,b,'sigmoid')
        self.caches.append(cache)
        # print(np.array(self.caches).shape)
        assert(AL.shape == (1,m))
        return AL

    def compute_cost(self,AL):
        cost = -1*(np.sum(np.multiply(self.Y,np.log(AL)) + np.multiply(1-self.Y,np.log(1-AL))))/self.m
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost

    def linear_backward(self,dZ,cache):
        A_prev, W, b = cache
        dW = (np.dot(dZ,A_prev.T))/self.m
        db = np.sum(dZ,axis=1,keepdims=True)/self.m
        dA_prev = np.dot(W.T,dZ)
        assert(dA_prev.shape == A_prev.shape)
        assert(db.shape == b.shape)
        assert(dW.shape == W.shape)
        return dA_prev, dW, db

    def linear_activation_backward(self,dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == 'relu':
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        if activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def total_backward(self,AL):
        self.grads = {}
        dAL = -(np.divide(self.Y,AL) - np.divide(1-self.Y,1-AL))
        current_cache = self.caches[self.L-2]
        self.grads["dA"+str(self.L-1)], self.grads["dW"+str(self.L-1)], self.grads["db"+str(self.L-1)] = self.linear_activation_backward(dAL,current_cache,"sigmoid")
        for l in reversed(range(self.L-2)):
            current_cache = self.caches[l]
            current_dA = self.grads['dA'+str(l+2)]
            # print(len(current_cache))
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(current_dA,current_cache,"relu")
            self.grads['dA'+str(l+1)] = dA_prev_temp
            self.grads['dW'+str(l+1)] = dW_temp
            self.grads['db'+str(l+1)] = db_temp

    def update_parameters(self):
        for l in range(1,self.L):
            self.parameters["W"+str(l)] = self.parameters["W"+str(l)] - self.learning_rate*self.grads["dW"+str(l)]
            self.parameters["b"+str(l)] = self.parameters["b"+str(l)] - self.learning_rate*self.grads["db"+str(l)]

    def train(self,X,Y,learning_rate=0.075,num_iterations=3000,print_cost=False):
        assert(X.shape[1] == self.layers[0]),"The training feature size does not match size of first layer"
        self.m = X.shape[0]
        self.X = X.T
        self.Y = Y
        self.learning_rate = learning_rate
        self.initialize_parameters()
        self.costs = []
        for i in range(num_iterations):
            AL = self.total_forward(self.X)
            cost = self.compute_cost(AL)
            self.total_backward(AL)
            self.update_parameters()
            if print_cost and i%100 == 0:
                print("Cost after iteration "+str(i)+": "+str(cost))
                self.costs.append(cost)
                print(self.parameters)
        predictions = self.predict(self.X.T)
        print(predictions)
        accuracy = (predictions == self.Y).sum()*100/self.m
        return accuracy

    def predict(self,X):
        X = np.array(X).T
        predictions = self.total_forward(X)
        predictions[predictions < 0.5] = 0
        predictions[predictions >= 0.5] = 1
        return predictions
