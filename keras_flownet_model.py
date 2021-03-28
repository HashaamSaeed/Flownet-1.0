import keras
from keras.models import Model
from keras import backend as K
from keras.layers import UpSampling2D, ZeroPadding2D, concatenate
from keras.layers import Conv2D, Input, LeakyReLU, Conv2DTranspose
import numpy as np

from data_pipeline import get_train_data



########################################################################################################################### Loss definitions 


## defining our own coustom end point error loss as mentioned in the paper 
def EPE(y_true, y_pred):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    UNKNOWN_FLOW_THRESH = 1e7
    LARGEFLOW = 1e8
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    tu = y_true[:,:,:,0]      ## (384, 512, 2)  readflow 
    tv = y_true[:,:,:,1]
    u = y_pred[:,:,:,0]
    v = y_pred[:,:,:,1]


    stu = tu
    stv = tv
    su = u
    sv = v
    """

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (
        abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    """

    """

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    """

    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    # epe = K.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = K.sqrt( K.square( stu - su ) + K.square( stv - sv ) )
    #epe = epe[ind2]
    mepe = K.mean(epe)

    return mepe



########################################################################################################################### Data piepline

datagen = get_train_data()


########################################################################################################################### optimiser


optim = keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)



########################################################################################################################### Model

def FLowNet_S(shape):


    x = Input(shape=shape)
    conv0 = Conv2D(64,(3,3),padding='same',name='conv0', kernel_initializer='he_normal')(x)
    conv0 = LeakyReLU(0.1)(conv0)
    padding = ZeroPadding2D()(conv0)

    conv1 = Conv2D(64,(3,3),strides=(2,2),padding='valid',name='conv1' ,kernel_initializer='he_normal')(padding)
    conv1 = LeakyReLU(0.1)(conv1)
    
    conv1_1 = Conv2D(128,(3,3),padding='same',name='conv1_1',kernel_initializer='he_normal')(conv1)
    conv1_1 = LeakyReLU(0.1)(conv1_1)
    padding = ZeroPadding2D()(conv1_1)
    
    conv2 = Conv2D(128,(3,3),strides=(2,2),padding='valid',name='conv2',kernel_initializer='he_normal')(padding)
    conv2 = LeakyReLU(0.1)(conv2)
    
    conv2_1 = Conv2D(128,(3,3),padding='same',name='conv2_1',kernel_initializer='he_normal')(conv2)
    conv2_1 = LeakyReLU(0.1)(conv2_1)
    padding = ZeroPadding2D()(conv2_1)

    conv3 = Conv2D(256,(3,3),strides=(2,2),padding='valid',name='conv3',kernel_initializer='he_normal')(padding)
    conv3 = LeakyReLU(0.1)(conv3)

    conv3_1 = Conv2D(256,(3,3),padding='same',name='conv3_1',kernel_initializer='he_normal')(conv3)
    conv3_1 = LeakyReLU(0.1)(conv3_1)
    padding = ZeroPadding2D()(conv3_1)

    conv4 = Conv2D(512,(3,3),strides=(2,2),padding='valid',name='conv4',kernel_initializer='he_normal')(padding)
    conv4 = LeakyReLU(0.1)(conv4)
    
    conv4_1 = Conv2D(512,(3,3),padding='same',name='conv4_1',kernel_initializer='he_normal')(conv4)
    conv4_1 = LeakyReLU(0.1)(conv4_1)
    padding = ZeroPadding2D()(conv4_1)
    
    conv5 = Conv2D(512,(3,3),strides=(2,2),padding='valid',name='conv5',kernel_initializer='he_normal')(padding)
    conv5 = LeakyReLU(0.1)(conv5)
    
    conv5_1 = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv5_1',kernel_initializer='he_normal')(conv5)
    conv5_1 = LeakyReLU(0.1)(conv5_1)
    padding = ZeroPadding2D()(conv5_1)
    
    conv6 = Conv2D(1024,(3,3),strides=(2,2),padding='valid',name='conv6',kernel_initializer='he_normal')(padding)
    conv6 = LeakyReLU(0.1)(conv6)
    
    conv6_1 = Conv2D(1024,(3,3),padding='same',name='conv6_1',kernel_initializer='he_normal')(conv6)
    conv6_1 = LeakyReLU(0.1)(conv6_1)

    flow6 = Conv2D(2,(3,3),padding='same',name='predict_flow6',kernel_initializer='he_normal')(conv6_1)
    flow6_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow6_to_5',padding='same',kernel_initializer='he_normal')(flow6)
    deconv5 = Conv2DTranspose(512,(4,4),strides=(2,2),padding='same',name='deconv5',kernel_initializer='he_normal')(conv6_1)
    deconv5 = LeakyReLU(0.1)(deconv5)

    # print(deconv5.get_shape())
    concat5 = concatenate([conv5_1,deconv5,flow6_up],axis=3)  # 16
    inter_conv5 = Conv2D(512,(3,3),padding='same',name='inter_conv5',kernel_initializer='he_normal')(concat5)
    flow5 = Conv2D(2,(3,3),padding='same',name='predict_flow5',kernel_initializer='he_normal')(inter_conv5)

    flow5_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow5_to4',padding='same',kernel_initializer='he_normal')(flow5) #32
    deconv4 =  Conv2DTranspose(256,(4,4),strides=(2,2),name='deconv4',padding='same',kernel_initializer='he_normal')(concat5)
    deconv4 = LeakyReLU(0.1)(deconv4)

    concat4 = concatenate([conv4_1,deconv4,flow5_up],axis=3)
    inter_conv4 = Conv2D(256,(3,3),padding='same',name='inter_conv4',kernel_initializer='he_normal')(concat4)
    flow4 = Conv2D(2,(3,3),padding='same',name='predict_flow4',kernel_initializer='he_normal')(inter_conv4)  # (1, 2, 32, 32)

    flow4_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow4_to3',padding='same',kernel_initializer='he_normal')(flow4)  #64
    deconv3 = Conv2DTranspose(128,(4,4),strides=(2,2),name='deconv3',padding='same',kernel_initializer='he_normal')(concat4)
    deconv3 = LeakyReLU(0.1)(deconv3)

    concat3 = concatenate([conv3_1,deconv3,flow4_up],axis=3)  # 64
    inter_conv3 = Conv2D(128,(3,3),padding='same',name='inter_conv3',kernel_initializer='he_normal')(concat3)
    flow3 = Conv2D(2,(3,3),padding='same',name='predict_flow3',kernel_initializer='he_normal')(inter_conv3)
    flow3_up = Conv2DTranspose(2,(4,4),strides=(2,2),name='upsampled_flow3_to2',padding='same',kernel_initializer='he_normal')(flow3)  #128
    deconv2 = Conv2DTranspose(64,(4,4),strides=(2,2),name='deconv2',padding='same',kernel_initializer='he_normal')(concat3)
    deconv2 = LeakyReLU(0.1)(deconv2)

    concat2 = concatenate([conv2_1,deconv2,flow3_up],axis=3)
    inter_conv2 = Conv2D(64,(3,3),padding='same',name='inter_conv2',kernel_initializer='he_normal')(concat2)
    flow2 = Conv2D(2,(3,3),padding='same',name='predict_flow2',kernel_initializer='he_normal')(inter_conv2)
    result = UpSampling2D(size=(4,4),interpolation='bilinear')(flow2)   # 4*128


    model = Model(x,result)

    model.compile(optimizer=optim, loss=EPE)  # ,metrics=[dice_coef]


    return model


########################################################################################################################### Model definition 

model = FLowNet_S([384,512,6])
model.summary()


callbacks = [
    keras.callbacks.ModelCheckpoint('FLowNet_S.h5',save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=1, patience=2, mode='min'), ## new_lr = lr * factor # 5
    keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=4, mode='min', restore_best_weights=True),    
    keras.callbacks.CSVLogger('training.csv'),
    keras.callbacks.TensorBoard(log_dir='logs',write_graph=True),
    keras.callbacks.TerminateOnNaN()
]




Model_fit = model.fit(datagen[0], datagen[1], validation_split=0.1, batch_size=8, epochs=30, callbacks=callbacks)



model.save('FLowNet_S_1')





