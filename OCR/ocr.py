import os
import string
import cv2
import h5py
import fnmatch
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Lambda, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ocr(path):
    output_path = os.path.dirname(os.path.abspath(__file__)) + '/downloads/output.txt'
    output_file = open("templates/downloads/output.txt",'w+')
    ofile = open(output_path,'w+')
    output_file.write(crnn(path))
    ofile.write(crnn(path))
    ofile.close()
    output_file.close()

def crnn(path):

    char_list = string.ascii_letters + string.digits
    max_label_len = 26

    # input with shape of height=32 and width=128 
    inputs = Input(shape=(32,128,1))
    
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    # poolig layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)
    
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    
    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
    
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
    
    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)
    
    act_model = Model(inputs, outputs)

    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = 32
    height = 128

    # dsize
    dsize = (height, width)
    img = cv2.resize(img, dsize)
    img = np.expand_dims(img , axis = 2)
    img = img/255.

    # load the saved best model weights
    act_model.load_weights('best_model.hdf5')

    # predict outputs on validation images
    prediction = act_model.predict(img[np.newaxis,:,:])
    
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
    
    # see the results
    for x in out:
        #print(valid_orig_txt[i])
        r = []
        for p in x:  
            if int(p) != -1:
                #print(char_list[int(p)], end = '')   
                r.append(char_list[int(p)])    
        #print('\n')
        return(''.join(r) )

