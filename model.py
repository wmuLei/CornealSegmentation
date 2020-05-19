
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.layers import merge, BatchNormalization, Subtract
from keras.models import Model


def CONV2D(x, filter_num, kernel_size, activation='relu', **kwargs):
    x = Conv2D(filter_num, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if activation=='relu': 
        x = Activation('relu', **kwargs)(x)
    elif activation=='sigmoid': 
        x = Activation('sigmoid', **kwargs)(x)
    else:
        x = Activation('softmax', **kwargs)(x)
    return x


def OurNet(shape, classes=1):
    inputs = Input(shape)
    conv0 = BatchNormalization()(inputs)

    conv0 = CONV2D(conv0, 32, (3, 3))
    conv1 = CONV2D(conv0, 32, (3, 3)); edge1 = Subtract()([conv0, conv1]); conv1 = CONV2D(merge([conv1, edge1], mode='concat', concat_axis=3), 32, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

    conv0 = CONV2D(pool1, 64, (3, 3))
    conv2 = CONV2D(conv0, 64, (3, 3)); edge2 = Subtract()([conv0, conv2]); conv2 = CONV2D(merge([conv2, edge2], mode='concat', concat_axis=3), 64, (3, 3));
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

    conv0 = CONV2D(pool2, 128, (3, 3))
    conv3 = CONV2D(conv0, 128, (3, 3)); edge3 = Subtract()([conv0, conv3]); conv3 = CONV2D(merge([conv3, edge3], mode='concat', concat_axis=3), 128, (3, 3));
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 

    conv0 = CONV2D(pool3, 256, (3, 3))
    conv4 = CONV2D(conv0, 256, (3, 3)); edge4 = Subtract()([conv0, conv4]); conv4 = CONV2D(merge([conv4, edge4], mode='concat', concat_axis=3), 256, (3, 3));
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) 

    conv0 = CONV2D(pool4, 512, (3, 3))
    conv5 = CONV2D(conv0, 512, (3, 3)); edge5 = Subtract()([conv0, conv5]); conv5 = CONV2D(merge([conv5, edge5], mode='concat', concat_axis=3), 512, (3, 3));
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5) 

    #=============================================
    conv0 = CONV2D(pool5, 512, (3, 3))
    conv6 = CONV2D(conv0, 512, (3, 3)); edge6 = Subtract()([conv0, conv6]); conv6 = CONV2D(merge([conv6, edge6], mode='concat', concat_axis=3), 512, (3, 3));

    #=============================================
    # for boundary extraction
    #=============================================

    up1 = UpSampling2D(size=(2, 2))(edge6)
    merg1 = merge([up1, edge5], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 256, (3, 3))
    conv7 = CONV2D(conv0, 256, (3, 3)); edge7 = Subtract()([conv0, conv7]);

    up1 = UpSampling2D(size=(2, 2))(edge7)
    merg1 = merge([up1, edge4], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 128, (3, 3))
    conv8 = CONV2D(conv0, 128, (3, 3)); edge8 = Subtract()([conv0, conv8]);
    
    up1 = UpSampling2D(size=(2, 2))(edge8)
    merg1 = merge([up1, edge3], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 64, (3, 3))
    conv9 = CONV2D(conv0, 64, (3, 3)); edge9 = Subtract()([conv0, conv9]);

    up1 = UpSampling2D(size=(2, 2))(edge9)
    merg1 = merge([up1, edge2], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 32, (3, 3))
    conv10 = CONV2D(conv0, 32, (3, 3)); edge10 = Subtract()([conv0, conv10]);
    
    up1 = UpSampling2D(size=(2, 2))(edge10)
    merg1 = merge([up1, edge1], mode='concat', concat_axis=3) 

    conv0 = CONV2D(merg1, 32, (3, 3))
    conv11 = CONV2D(conv0, 32, (3, 3)); edge11 = Subtract()([conv0, conv11]);
    Boundary = CONV2D(edge11, classes, (1, 1), activation='sigmoid')
    
    #=============================================
    # for object extraction
    #=============================================
    up1 = UpSampling2D(size=(2, 2))(conv6)
    merg1 = merge([up1, conv5, edge5, edge7], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 256, (3, 3))
    conv7 = CONV2D(conv0, 256, (3, 3)); edge7 = Subtract()([conv0, conv7]); conv7 = CONV2D(merge([conv7, edge7], mode='concat', concat_axis=3), 256, (3, 3));

    up1 = UpSampling2D(size=(2, 2))(conv7)
    merg1 = merge([up1, conv4, edge4, edge8], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 128, (3, 3))
    conv8 = CONV2D(conv0, 128, (3, 3)); edge8 = Subtract()([conv0, conv8]); conv8 = CONV2D(merge([conv8, edge8], mode='concat', concat_axis=3), 128, (3, 3));
    
    up1 = UpSampling2D(size=(2, 2))(conv8)
    merg1 = merge([up1, conv3, edge3, edge9], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 64, (3, 3))
    conv9 = CONV2D(conv0, 64, (3, 3)); edge9 = Subtract()([conv0, conv9]); conv9 = CONV2D(merge([conv9, edge9], mode='concat', concat_axis=3), 64, (3, 3));
    
    up1 = UpSampling2D(size=(2, 2))(conv9)
    merg1 = merge([up1, conv2, edge2, edge10], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 32, (3, 3))
    conv10 = CONV2D(conv0, 32, (3, 3)); edge10 = Subtract()([conv0, conv10]); conv10 = CONV2D(merge([conv10, edge10], mode='concat', concat_axis=3), 32, (3, 3));
    
    up1 = UpSampling2D(size=(2, 2))(conv10)
    merg1 = merge([up1, conv1, edge1, edge11], mode='concat', concat_axis=3) 
    conv0 = CONV2D(merg1, 32, (3, 3))
    conv11 = CONV2D(conv0, 32, (3, 3)); edge11 = Subtract()([conv0, conv11]); conv11 = CONV2D(merge([conv11, edge11], mode='concat', concat_axis=3), 32, (3, 3));
    
    Object = CONV2D(conv11, classes, (1, 1), activation='sigmoid')

    model = Model(input=inputs, output=[Object, Boundary])
    model.summary() 
    return model



