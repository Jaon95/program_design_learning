#%%
import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

print(paddle.__version__)

# %%
# 加载数据
batch_size = 32
transform = ToTensor()
cifar10_train = paddle.vision.datasets.Cifar10(mode='train',
                                               transform=transform)
cifar10_test = paddle.vision.datasets.Cifar10(mode='test',
                                              transform=transform)
train_loader = paddle.io.DataLoader(cifar10_train,
                                        shuffle=True,
                                        batch_size=batch_size)
# %%
sample = next(train_loader())
print(np.array(sample[0][0]).shape)
# %%
# print(sample[0])
a = np.array(sample[0][0])
# # im = im.transpose((2, 0, 1))
# # print(im.shape)
# a.transpose(1,2,0)
# plt.imshow(a.transpose(1,2,0))
# %%
def convolutional_neural_network2(img):
    #下面分别定义卷积层和池化层,第一个
    con1 = paddle.nn.Conv2D(input=img,filter_size=5,num_filters=20,act="relu",name='i am con01')
    pool1 = paddle.nn.MaxPool2D(input=con1,pool_size=2,pool_stride=2)
    pool1 = paddle.nn.BatchNorm2D(pool1)
    print(pool1.shape)

    #
    con2 = paddle.nn.Conv2D(input=pool1,filter_size=5,num_filters=50,act="relu")
    pool2 = paddle.nn.MaxPool2D(input=con2,pool_size=2,pool_stride=2)
    pool2  = paddle.nn.BatchNorm2D(pool2)
    print(pool2.shape)

    #
    con3= paddle.nn.Conv2D(input = pool2,filter_size=5,num_filters=50,act='relu')
    #pool3 = fluid.layers.nn.pool2d(input=con3,pool_size=2,pool_stride=2)
    pool3 = con3
    print(pool3.shape)

    return fluid.layers.fc(input=pool3,size=10,act='softmax')
# %%
