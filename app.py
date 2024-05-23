import streamlit as st
import pandas as pd
import numpy as np
st.title("# 基于改进UNet算法的农业图像分割应用研究")

st.text("")

#文本语句
st.write("""

""")

st.code("""

""")

#图片语句
#st.image("tt.png")

st.write("""
### 导入组件包
""")

st.code("""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  
from keras.layers import *  
from tensorflow.keras.utils import img_to_array
from keras.callbacks import ModelCheckpoint  
from sklearn.preprocessing import LabelEncoder  
from keras.models import Model
from keras.models import load_model
from keras.layers import concatenate
import cv2
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
""")
st.write("""### 下载数据""")
st.code("""
https://tianchi.aliyun.com/competition/entrance/231717/
""")


st.write("""### 解压数据集""")
st.code("""
#解压数据集
!unzip -qq -o ./unet2_dataset.zip -d ./
!rm -rf ./unet2_dataset.zip
""")
st.write("""
### 数据可视化
标注作物类型图片是一张unint8单通道图像。每个像素点值表示原始图片中对应位
""")
st.image("output.png")

st.write("""
### 设置数据和标签
这里采用labelEncoder方法进行标签的处理，标准化标签，将标签值统一转换成range范围内。
""")
st.code("""
width = 512  
height = 512  
classes = [0. ,  1.,  2.,   3.  , 4.]  
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 
""")
st.image("code1.png")

st.write("""
### 图片加载函数
对指定路径的图片进行加载，如果是灰度图，直接读取，，如果是RGB图像，则对图像进行归一化。
""")

st.code("""
def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img
""")

st.write("""
## 模型搭建训练和改进

### 定义训练参数和数据
""")
st.code("""
seed = 7  
np.random.seed(seed)  
""")

st.write("""
### 设置验证集获取规则，将训练集的1/4用作验证集。
""")

st.code("""
def get_train_val(val_rate = 0.25):
    train_url = []    
    train_set = []
    val_set  = []
    for pic in os.listdir('./UNetP_data/train_data/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set
""")

st.write("""
### 设置训练集生成函数
""")

st.code("""
def generateData(batch_size,data=[]):  
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img('./UNetP_data/train_data/' + url)
            img = img_to_array(img)  
            train_data.append(img)  
            label = load_img('./UNetP_data/train_label/' + url, grayscale=True) 
            label = img_to_array(label)
            train_label.append(label)  
            if batch % batch_size==0: 
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
""")


st.write("""
### 设置验证集生成函数
""")

st.code("""
def generateValidData(batch_size,data=[]):  
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img('./UNetP_data/train_data/'+ url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img('./UNetP_data/train_label/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label)  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  
""")

st.write("""
### UNet网络

Unet网络结构如其名呈现一个U字形，即由卷积和池化单元构成，左半边为编码器即如传统的分类网络是“下采样阶段”，右半边为解码器是“上采样阶段”，中间的灰色箭头为跳跃连接，将浅层的特征与深层的特征拼接，因为浅层通常可以抓取图像的一些简单的特征，比如边界，颜色，而深层经过的卷积操作多抓取到图像的一些抽象特征，将浅深同时利用起来会起到比较好的效果，Unet的效果图如下：
""")

st.code("""
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
img_org = cv2.imread('./unet.jpeg')
plt.imshow(img_org)
""")
st.image("code2.png")

st.text("")



st.code("""
img_org = cv2.imread('./unet1.jpeg')
plt.imshow(img_org)
""")
st.image("code3.png")

st.write("""
## 训练函数的定义
""")

st.code("""
#rom keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import Adam

def train(mselect, learning_rate=0.001): 
    EPOCHS = 5
    BS = 16  
    if mselect == 'UNet':
        input_shape = (512, 512, 3)
        num_classes = 2
        model = unet(input_shape, num_classes)
        optimizer = Adam(learning_rate=learning_rate)  # 设置学习率
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        modelcheck = ModelCheckpoint('model_unet.h5', monitor='val_accuracy', save_best_only=True, mode='max')  
    else:
        input_shape = (512, 512, 3)
        model = unet_new(input_shape)
        optimizer = Adam(learning_rate=learning_rate)  # 设置学习率
        # model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        modelcheck = ModelCheckpoint('model_unet_new.h5', monitor='val_accuracy', save_best_only=True, mode='max') 

    callbacks_list = [modelcheck]  

    train_set, val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print("the number of train data is", train_numb)  
    print("the number of val data is", valid_numb)

    train_data_generator = generateData(BS, train_set)
    val_data_generator = generateValidData(BS, val_set)
    H = model.fit_generator(generator=train_data_generator,
                            steps_per_epoch=train_numb // BS,
                            epochs=EPOCHS,
                            verbose=1,  
                            validation_data=val_data_generator,
                            validation_steps=valid_numb // BS,
                            callbacks=callbacks_list,
                            max_queue_size=1)  

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
    if mselect == 'UNet':
        plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    else:
        plt.title("Training Loss and Accuracy on U-Net_new Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')

# 调用 train 函数时，传入学习率参数
# train('UNet_new', learning_rate=0.001)

""")

st.write("""
## unet网络的定义

池化层采用最大池化，激活函数采用relu，上采样的线性插值方式采取keras的UpSampling2D方法
""")

st.code("""
import tensorflow as tf

def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', padding='same'):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
    return conv

def decoder_block(inputs, skip, filters, strides=(2, 2), padding='same'):
    upsample = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=strides, padding=padding)(inputs)
    upsample = tf.keras.layers.concatenate([upsample, skip], axis=3)
    return upsample

def unet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 编码器部分
    conv1 = conv_block(inputs, 64)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(pool1, 128)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(pool2, 256)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(pool3, 512)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # 中间部分
    conv5 = conv_block(pool4, 1024)
    
    # 解码器部分
    up6 = decoder_block(conv5, conv4, 512)
    conv6 = conv_block(up6, 512)
    
    up7 = decoder_block(conv6, conv3, 256)
    conv7 = conv_block(up7, 256)
    
    up8 = decoder_block(conv7, conv2, 128)
    conv8 = conv_block(up8, 128)

    up9 = decoder_block(conv8, conv1, 64)
    conv9 = conv_block(up9, 64)
    
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
""")


st.write("""
## 训练unet网络

为了匹配通道维度的顺序，需要调整维度顺序为channel_last，然后进行训练，可以看到原始unet损失函数收敛较慢，准确率也一直较低，后面我们会对unet模型进行改进
""")

st.code("""
from keras import backend as K

K.set_image_data_format('channels_last')
train('UNet')  
""")
st.image("code4.png")
st.image("code5.png")

st.write("""
## 模型改进

我们对原来的unet网络增加了下采样层，并且设定了权重的随机初始化，池化方式更改为平均池化，并且在网络中加入了批规范化BatchNormalization的处理，基于梯度的训练过程可以更加有效的进行，即加快收敛速度，减轻梯度消失或爆炸导致的无法训练的问题，并且增加了Dropout的机制，有效的防止过拟合。
""")

st.code("""
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation
from keras.models import Model  

def unet_new(input_shape):
    inputs = Input(input_shape)
    
    # 编码器部分
    conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  
    
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  
    
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  
    
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)   
    
    # 解码器部分
    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    
    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    
    up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    
    up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    
    # 输出层
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      
    return model
""")

st.write("""
## 训练改进后的模型

可以看到，经过改进后，训练函数的收敛迅速，并且准确率也得到了明显的提高
""")

st.code("""
train('UNet_new', 0.001) 
""")
st.image("code6.png")
st.image("code7.png")

#文本语句
st.write("""

""")

st.code("""

""")

#图片语句
#st.image("07709.png")

st.write("""
## 模型使用
""")
st.write("""
## 定义预测函数，加载预测模型
""")

st.code("""
def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('model_unet_new.h5')
    stride = 512
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        #load the image
        image = cv2.imread(path)
        
        h,w,_ = image.shape
        
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        #padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        
        print ('src:',padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[i*stride:i*stride+image_size,j*stride:j*stride+image_size,:3]#trick
               
                #_,ch,cw = crop.shape
#                 if ch != 512 or cw != 512:
#                     print ('invalid size!')
#                     continue
                
                crop = np.expand_dims(crop, axis=0) 
                
                crop.transpose(0,3,2,1)
                
                # crop.reshape(1,3,512,512)
                #print(crop.shape)
                pred = model.predict(crop,verbose=2)
                # pred = model.predict(crop.transpose(0,3,1,2),verbose=2)
                #print (np.unique(pred))  
                pred = pred.reshape((512,512)).astype(np.uint8)
                #print 'pred:',pred.shape
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

        
        cv2.imwrite('pre'+str(n+1)+'.png',mask_whole[0:h,0:w])

""")

st.write("""
## 加载原始图像
""")
st.image("test.png")



st.write("""

## 图像分割结果可视化

""")

st.code("""
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
# 加载原始图像
img_org = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(img_org)
""")
st.image("code8.png")


#文本语句
st.write("""
## 预测结果可视化
""")

st.code("""
# 加载模型预测结果
img_label = cv2.imread('./pre1-original.png', cv2.IMREAD_GRAYSCALE)
# img_label = cv2.imread('./pre1.png', cv2.IMREAD_GRAYSCALE)
# 像素放大
img_label*=30
plt.imshow(img_label)
""")

st.image("code9.png")

st.write("""
## 合并模型结果与原始图像，可以看到模型比较成功地将植被与背景环境分割开来，基本达到了预期效果。
""")

st.code("""
img_show = cv2.addWeighted(img_org, 0.1, img_label, 0.8, 0,dtype = cv2.CV_32F) 

plt.imshow(img_show)
""")
st.image("code10.png")
























