import random
def get_y(x1,x2,x3):
    a = x1*x1*3 + x2*5 + x3 + 9
    b = 30*30*3 + 150*5 + 554 + 9
    if a < b:
        return 1
    else:
        return 0
train = 10000
test = 5000
train_features = open('dataset/train.txt','w')
train_target = open('dataset/train_Target.txt','w')
test_features = open('dataset/test.txt','w')
test_target = open('dataset/test_target.txt','w')

for i in range(train):
    x1 = random.randrange(1,150,1)
    x2 = random.randrange(1,150,1)
    x3 = random.randrange(1,150,1)
    y = get_y(x1,x2,x3)
    train_features.write(str(x1) + " " +str(x2)+ " "+str(x3)+"\n")
    train_target.write(str(y)+"\n")
for i in range(test):
    x1 = random.randrange(1,150,1)
    x2 = random.randrange(1,150,1)
    x3 = random.randrange(1,150,1)
    y = get_y(x1,x2,x3)
    test_features.write(str(x1) + " " +str(x2)+ " "+str(x3)+"\n")
    test_target.write(str(y)+"\n")

train_features.close()
train_target.close()
test_features.close()
test_target.close()
