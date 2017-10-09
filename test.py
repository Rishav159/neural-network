import numpy as np
from binary_classification import l_layer_neural_network
nn = l_layer_neural_network([3,4,4,4,3,2,1])

train_features_file = open('dataset/train.txt','r')
train_target_file = open('dataset/train_Target.txt','r')
test_features_file = open('dataset/test.txt','r')
test_target_file = open('dataset/test_target.txt','r')

train_features = train_features_file.read()
train_target = train_target_file.read()
test_features = test_features_file.read()
test_target = test_target_file.read()

train_features_file.close()
train_target_file.close()
test_features_file.close()
test_target_file.close()

train_features = train_features.strip().split('\n')
train_features = np.array([list(map(int,x.strip().split(' '))) for x in train_features])
test_features = test_features.strip().split('\n')
test_features = np.array([list(map(int,x.strip().split(' '))) for x in test_features])
train_target = np.array(list(map(int,train_target.strip().split('\n'))))
test_target = np.array(list(map(int,test_target.strip().split('\n'))))

train_accuracy = nn.train(train_features,train_target,0.0045,6000,True)
predictions = nn.predict(test_features)
test_accuracy = (predictions == test_target).sum()*100/predictions.shape[1]
print(train_accuracy)
print(test_accuracy)
