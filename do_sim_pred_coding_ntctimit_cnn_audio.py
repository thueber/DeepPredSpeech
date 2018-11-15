#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
DeepPredSpeech: Computational models of predictive speech coding based on deep learning 
Main script for building models based on log-spectrogram (see associated article for more info)
T. Hueber - CNRS/GIPSA-lab - 2018 
thomas.hueber@gipsa-lab.fr
"""
from __future__ import print_function
import numpy as np
import scipy
import scipy.io as sio
from scipy import fftpack
import matplotlib
#matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import os
import librosa
from librosa import display
from os import listdir, mkdir, system
from os.path import join, isdir, basename, splitext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import ssi_utils
from ssi_utils import *
import glob
from progress.bar import Bar
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Merge, Flatten, Dropout, BatchNormalization, Add, Concatenate, Multiply, Average, Maximum,  Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.callbacks import EarlyStopping, Callback
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.models import load_model, clone_model
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.layers.advanced_activations import LeakyReLU
import soundfile as sf
from PIL import Image
import datetime

import pdb # debugger

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

# CONFIG (edit)
###############
# Output directory
output_dir_root = '/localdata/huebert/res_erc_ntcdtimit_cnn_multispeaker_cnn_audio/' #on gpu (tmux session 5)
#output_dir_root = '/localdata/huebert/res_erc_ntcdtimit_cnn_multispeaker_cnn_audio_taup_3/' #on gpu (tmux session 5)

# Audio data and video data
audio_root_dir = '/localdata/huebert/data/NTCD-timit/audio_clean/'
video_root_dir = '/localdata/huebert/data/NTCD-timit/lips_roi/'

# Corpus parameters
nb_sentences_per_speaker = 98

# Audio analysis (STFT)
stft_dir = '/localdata/huebert/data/NTCD-timit/fft_16k/'
audio_fs = 16000
n_fft = 512 # resulting in (n_fft/2)+1 fft bins
window = 'hamming'
win_length = 400 #i.e. 0.025 ms at 16kHz
hop_length = 400 #i.e. 0.025 ms at 16kHz

# TensorFlow wizardry
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # Don't pre-allocate memory; allocate as-needed
config.gpu_options.per_process_gpu_memory_fraction = 1.0 # Only allow a total of half the GPU memory to be allocated
k.tensorflow_backend.set_session(tf.Session(config=config)) # Create a session with the above options


# Audio model (CNN)
kernel_size_1D = 20
nb_filters_audio = [64] #[nb_filter_conv_layer_1,nb_filter_conv_layer_2, etc.]      
pooling_factor_1D = 2
cnn_arch_fc_audio = [256] #[nb_neurons_layer_1, nb_neurons_layer_2, etc.] (number of layers/neurons after flattening)

# Video model (CNN)
target_im_size = (32,32) 
nb_filters_video = [16,32,64] #[nb_filter_conv_layer_1,nb_filter_conv_layer_2, etc.]    
kernel_size_2D = 3   
pooling_factor_2D = 2
cnn_arch_fc_video = [256] #[nb_neurons_layer_1, nb_neurons_layer_2, etc.] (number of layers/neurons after flattening)

# Audio-video model (number of layers/neurons after fusion) 
cnn_arch_fc_audio_video = [256] #[nb_neurons_layer_1, nb_neurons_layer_2, etc.] 

# Training parameters 
max_nb_frames = 500000 # Memory pre-allocation for data loading
activation_function = 'tanh'#'relu'
nb_epoch = 1000
batch_size = 256
opt = 'adam'
initializer = 'random_normal' #he_normal'
dropout_ratio = 0.25
verbose = 1
early_stopping_patience = 10
my_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
validation_split_frac = 0.2 # Fraction of the training set using for validation/early_stopping 
nb_folds = 3 # K-fold cross-validation (depreciated)
train_test_ratio = 0.33 # Fraction of the database used for training+validation on one hand, and test for the other hand 

# Past context to use
all_tau_p = [0,1,2,3]

# Future context to predict
all_tau_f = [0,1,2,3,4,6,10]

# Silence removal
mlf_filename = '/localdata/huebert/data/NTCD-timit/volunteer_labelfiles.mlf'
offset = 6 # in frames (i.e. 150 ms)

# Steps
step_extract_stft = 0
step_remove_silence = 1
step_display = 0

step_export_mse_per_frame = 0 # used to build the database of prediction errors (slow and need large storage capacity)

# RUN (do not edit)
###################
# fix random seed for reproducibility of the results
seed1 = 7
seed2 = 42
np.random.seed(seed1)

def main_multispeaker(): 
    # Create output directory if necessary
    if isdir(output_dir_root) is False:
        mkdir(output_dir_root)

    if isdir(output_dir_root + '/multispeaker/') is False:
        mkdir(output_dir_root + '/multispeaker/')
        
    print('Scanning audio features directory %s ...' % stft_dir)
    all_stft_speakers_fullpath = sorted(listdir_fullpath(stft_dir), key=numericalSort)
    all_stft_filenames = [];
    for s in range(shape(all_stft_speakers_fullpath)[0]):
        current_speaker_audio_filenames = sorted(glob.glob(all_stft_speakers_fullpath[s] + '/*.npy'), key=numericalSort)
        all_stft_filenames = np.append(all_stft_filenames,current_speaker_audio_filenames,axis=0)
    print('%i files found\n' % shape(all_stft_filenames)[0])

    print('Scanning video directory %s' % video_root_dir)
    all_im_speakers = sorted(listdir_fullpath(video_root_dir), key=numericalSort)
    all_im_dir = []
    for s in range(shape(all_im_speakers)[0]):
        current_speaker_im_dir = sorted(listdir_fullpath(join(video_root_dir,all_im_speakers[s])), key=numericalSort)
        all_im_dir = np.append(all_im_dir,current_speaker_im_dir,axis=0)

    print('%i files found\n' % shape(all_im_dir)[0])

    # Main K-fold loop
    #pdb.set_trace()
    train_ind, test_ind = train_test_split(range(shape(all_stft_filenames)[0]),test_size=train_test_ratio, random_state=seed2)
    #kf = KFold(n_splits=nb_folds)
    current_fold_index = 1
    if 1:
    #for train_ind, test_ind in kf.split(range(shape(all_stft_filenames)[0])):
        # create output directory and sub-directories for the current cross-validation fold
        output_dir = (output_dir_root + '/multispeaker/%i/' % current_fold_index)

        if isdir(output_dir) is False:
            mkdir(output_dir)
        
        all_rmse_audio = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
        all_rmse_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
        all_rmse_audio_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
        all_evr_audio = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
        all_evr_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
        all_evr_audio_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
            
        sim_iter = 1
        for p in range(np.shape(all_tau_p)[0]):
            tau_p = all_tau_p[p]
            for f in range(np.shape(all_tau_f)[0]):
                tau_f = all_tau_f[f]
                print('\n\nSimulation %i/%i (folds %i / tau_p = %i / tau_f = %i)' % (sim_iter,np.shape(all_tau_p)[0]*np.shape(all_tau_f)[0]*nb_folds,current_fold_index,tau_p,tau_f))

                # TRAIN
                model_audio, model_video, model_audio_video, audio_min_max_scaler_in, audio_min_max_scaler_out  = do_train(output_dir, all_im_dir[[train_ind]], all_stft_filenames[[train_ind]], tau_p, tau_f)
                    
                # Save models (for multispeaker experiments only)
                model_audio.save(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                model_video.save(output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                model_audio_video.save(output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                joblib.dump(audio_min_max_scaler_in, output_dir + '/audio_min_max_scaler_in_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                joblib.dump(audio_min_max_scaler_out, output_dir + '/audio_min_max_scaler_out_' + str(tau_p) + '_' + str(tau_f) +'.dat')
                
                # Plot models
                #plot_model(model_audio, to_file=output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                #plot_model(model_video, to_file=output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                #plot_model(model_audio_video, to_file=output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                
                #model_audio = load_model(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                #model_video = load_model(output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                #model_audio_video = load_model(output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                #audio_min_max_scaler_in = joblib.load(output_dir + '/audio_min_max_scaler_in_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                #audio_min_max_scaler_out = joblib.load(output_dir + '/audio_min_max_scaler_out_' + str(tau_p) + '_' + str(tau_f) + '.dat')
            
                # TEST
                all_rmse_audio_current_exp,all_rmse_video_current_exp, all_rmse_audio_video_current_exp, all_evr_audio_current_exp, all_evr_video_current_exp, all_evr_audio_video_current_exp = do_test(output_dir, all_im_dir[[test_ind]],all_stft_filenames[[test_ind]], tau_p, tau_f, audio_min_max_scaler_in,audio_min_max_scaler_out, model_audio, model_video, model_audio_video)

                all_rmse_audio[p,f,:] = all_rmse_audio_current_exp
                all_rmse_video[p,f,:] = all_rmse_video_current_exp 
                all_rmse_audio_video[p,f,:] = all_rmse_audio_video_current_exp 
                all_evr_audio[p,f,:] = all_evr_audio_current_exp
                all_evr_video[p,f,:] = all_evr_video_current_exp 
                all_evr_audio_video[p,f,:] = all_evr_audio_video_current_exp 

                print('RMSE audio = %f / RMSE video = %f / RMSE audio_video = %f ' % (np.mean(all_rmse_audio_current_exp),np.mean(all_rmse_video_current_exp),np.mean(all_rmse_audio_video_current_exp)))
                print('EVR audio = %f / EVR video = %f / EVR audio_video = %f ' % (np.mean(all_evr_audio_current_exp),np.mean(all_evr_video_current_exp), np.mean(all_evr_audio_video_current_exp)))
                    
                sim_iter = sim_iter + 1
                
                # Save results (numpy format)
                np.save(output_dir + '/all_rmse_audio_' + str(current_fold_index) + '.mat',all_rmse_audio)
                np.save(output_dir + '/all_rmse_video_' + str(current_fold_index) + '.mat',all_rmse_video)
                np.save(output_dir + '/all_rmse_audio_video_' + str(current_fold_index) + '.mat',all_rmse_audio_video)
                np.save(output_dir + '/all_evr_audio_' + str(current_fold_index) + '.mat',all_evr_audio)
                np.save(output_dir + '/all_evr_video_' + str(current_fold_index) + '.mat',all_evr_video)
                np.save(output_dir + '/all_evr_audio_video_' + str(current_fold_index) + '.mat',all_evr_audio_video)

        current_fold_index = current_fold_index + 1
############################################
############################################

# Sub-functions
###############
def extract_stft_ntcdtimit_librosa(input_dir, target_dir):
    
    if isdir(target_dir) is False:
        mkdir(target_dir)

    all_audio_speakers = sorted(listdir(input_dir), key=numericalSort)
    all_audio_speakers_fullpath = sorted(listdir_fullpath(input_dir), key=numericalSort)

    for s in range(shape(all_audio_speakers)[0]):
        if isdir(target_dir + '/' + all_audio_speakers[s]) is False:
            mkdir(target_dir + '/' + all_audio_speakers[s])
        current_speaker_audio_filenames = glob.glob(all_audio_speakers_fullpath[s] + '/straightcam/*.wav')

        bar = Bar('Processing speaker %s' % all_audio_speakers[s], max=shape(current_speaker_audio_filenames)[0])
        for f in range(shape(current_speaker_audio_filenames)[0]):
            # Load audio file
            y, sr = librosa.load(current_speaker_audio_filenames[f],sr=None)

            # Comput STFT
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y,n_fft=n_fft, win_length=win_length, hop_length=hop_length)), ref=np.max)

            # Save
            audio_filename_no_root = basename(current_speaker_audio_filenames[f])
            audio_basename = splitext(audio_filename_no_root)[0]

            np.save(target_dir + '/' + all_audio_speakers[s] + '/' + audio_basename + '.npy',D.transpose())
            bar.next()
        bar.finish()
 
######################################################

def listdir_fullpath(d):
    return [join(d, f) for f in listdir(d)]
############################################

def build_inout(indata,tp,tf):
    indata_context = []
    for r in range(tp,(np.shape(indata)[0]-tf)):
        tmp = indata[r,:]
        for p in range(tp):
            tmp = np.append(tmp,indata[r-(p+1),:],axis=0)
            #print(tmp)
        if r==tp:
            indata_context = tmp;
        else:
            indata_context = np.vstack([indata_context,tmp])
            
    outdata = indata[range(tf+tp,np.shape(indata)[0]),:]
          
    return indata_context, outdata
############################################

def load_data_single_sentence(video_dir, audio_feature_filename,tp,tf):
    audio_features = np.load(audio_feature_filename)

    all_im_current_sentence = glob.glob(join(video_dir, '*.png'))

    all_im_current_sentence = sorted(all_im_current_sentence, key=numericalSort)
    nb_im_in_current_sentence = size(all_im_current_sentence)
    img_current_sentence = []
    for f in range(nb_im_in_current_sentence):
        img = cv2.imread(all_im_current_sentence[f], 0)
        img_resized = cv2.resize(img,target_im_size)
        img_resized_np = np.array(img_resized, dtype=np.float)/255
        img_current_sentence.append(img_resized_np.ravel())
    video_features = np.asarray(img_current_sentence)# convert list of images to a numpy array 

    # interpolate video images (vectorized) to fit the audio analysis rate
    video_features = scipy.interpolate.griddata(range(shape(video_features)[0]), video_features, np.linspace(0, shape(video_features)[0]-1, shape(audio_features)[0]), method='linear', fill_value=nan, rescale=False)

    # Remove silence
    if step_remove_silence:
        (froot,sname) = os.path.split(audio_feature_filename)
        (froot,speaker_name) = os.path.split(froot)
        audio_basename = speaker_name  + '/' + sname
        
        current_sentence_start, current_sentence_stop, current_sentence_labels = extract_labels_from_mlf_ntcdtimit(audio_basename, mlf_filename)
        first_frame_to_keep = max(0,int(round((float(current_sentence_stop[0]*10e-8)*float(audio_fs)/float(hop_length)))) - offset)
        last_frame_to_keep = min(shape(video_features)[0],int(round((float(current_sentence_start[-1]*10e-8)*float(audio_fs)/float(hop_length)))) + offset)
        
        audio_features = audio_features[range(first_frame_to_keep,last_frame_to_keep),:]
        video_features = video_features[range(first_frame_to_keep,last_frame_to_keep),:]
        
    
    audio_features_with_past_context, audio_features_target = build_inout(audio_features,tp,tf)
    video_features_with_past_context, video_features_target = build_inout(video_features,tp,tf)

    return audio_features_with_past_context, audio_features_target, video_features_with_past_context, video_features_target
############################################

def load_data(all_video_dir,all_audio_feature_filenames, tp, tf):

    # Pre-allocate memory
    audio_data_in = np.zeros((((n_fft/2)+1)*(tp+1),max_nb_frames))
    video_data_in = np.zeros((target_im_size[0]*target_im_size[1]*(tp+1),max_nb_frames))
    audio_data_out = np.zeros((((n_fft/2)+1),max_nb_frames))
    video_data_out = np.zeros((target_im_size[0]*target_im_size[1],max_nb_frames))

    bar = Bar('Processing ', max=shape(all_audio_feature_filenames)[0])
    iter = 0
    for f in range(shape(all_audio_feature_filenames)[0]):
	audio_feature_filename = all_audio_feature_filenames[f]
        video_dir = all_video_dir[f]

        audio_features_with_past_context, audio_features_target, video_features_with_past_context, video_features_target = load_data_single_sentence(video_dir, audio_feature_filename,tp,tf)

        audio_data_in[:,iter:iter + shape(audio_features_with_past_context)[0]] = audio_features_with_past_context.transpose()
        video_data_in[:,iter:iter + shape(video_features_with_past_context)[0]] = video_features_with_past_context.transpose()
        audio_data_out[:,iter:iter + shape(audio_features_target)[0]] = audio_features_target.transpose()
        video_data_out[:,iter:iter + shape(video_features_target)[0]] = video_features_target.transpose()

        iter = iter + shape(audio_features_target)[0]
        
        bar.next()
    bar.finish()
    print('%i frames loaded' % iter)
    return video_data_in[:,range(iter)].transpose(), audio_data_in[:,range(iter)].transpose(), video_data_out[:,range(iter)].transpose(), audio_data_out[:,range(iter)].transpose()
############################################

def build_dnn(arch,nb_input_features,nb_output_features,dropout_ratio,activation_function,verbosity):
    input_layer = Input(shape=(nb_input_features,))
    added = Dense(arch[0],kernel_initializer=initializer)(input_layer)
    added = BatchNormalization()(added)
    added = Activation(activation_function)(added)
    added = Dropout(dropout_ratio)(added)

    if shape(arch)[0]>1:
        for l in range(shape(arch)[0]-1):
            added = Dense(arch[l+1],kernel_initializer=initializer)(added)
            added = BatchNormalization()(added)
            added = Activation(activation_function)(added)
            added = Dropout(dropout_ratio)(added)
        
    output_layer = Dense(nb_output_features,kernel_initializer=initializer,activation='linear')(added)

    model = Model(inputs=input_layer, outputs=output_layer)
    if verbosity:
        model.summary()
        
    return model
############################################

def build_cnn_video(im_size, tp, nb_output_features, nb_filters, kernel_size_2D, pooling_factor,arch,dropout_ratio,activation_function,verbosity):

    model = Sequential()
    if tp==0:
        input_shape = (im_size[0],im_size[1],1)
        kernel_size = (kernel_size_2D,kernel_size_2D)
        pool_size = (pooling_factor,pooling_factor)

        for l in range(shape(nb_filters)[0]):
            model.add(Convolution2D(nb_filters[l], kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer))
            model.add(BatchNormalization())
	    model.add(LeakyReLU())	
	    model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Dropout(dropout_ratio))
        
    else:
        input_shape = (tp+1,im_size[0],im_size[1],1)
        kernel_size = (tp+1,kernel_size_2D,kernel_size_2D)
        pool_size = (1,pooling_factor,pooling_factor)

        for l in range(shape(nb_filters)[0]):
            model.add(Convolution3D(nb_filters[l],kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer))
            model.add(BatchNormalization())
	    model.add(LeakyReLU())	
	    model.add(MaxPooling3D(pool_size=pool_size))

        model.add(Dropout(dropout_ratio))
       
    model.add(Flatten())

    for l in range(shape(arch)[0]):
        model.add(Dense(arch[l],kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_ratio))

    # Output layer
    model.add(Dense(nb_output_features,kernel_initializer=initializer))
    model.add(Activation('linear'))

    if verbosity:
        model.summary()

    return model
#######################################################

def build_cnn_audio(im_size, tp, nb_output_features, nb_filters, kernel_size_1D, pooling_factor,arch,dropout_ratio,activation_function,verbosity):

    model = Sequential()
    input_shape = (im_size,tp+1,1)
    kernel_size = (kernel_size_1D,tp+1)
    pool_size = (pooling_factor,1)

    for l in range(shape(nb_filters)[0]):
        model.add(Convolution2D(nb_filters[l], kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer))
        model.add(BatchNormalization())
	model.add(LeakyReLU())	
	model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(dropout_ratio))
        
    model.add(Flatten())

    for l in range(shape(arch)[0]):
        model.add(Dense(arch[l],kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(activation_function))
        model.add(Dropout(dropout_ratio))

    # Output layer
    model.add(Dense(nb_output_features,kernel_initializer=initializer))
    model.add(Activation('linear'))

    if verbosity:
        model.summary()

    return model
#######################################################

def build_multistream_cnn_pretraining(model_cnn_video,model_cnn_audio,nb_filters_video,nb_filters_audio,arch,dropout_ratio,activation_function,verbosity):

    # Get all layer of the CNN video until Flatten
    input1 = model_cnn_video.input
    x1 = model_cnn_video.layers[0](input1) 
    for l in range(shape(nb_filters_video)[0]*4+1):
        #model_cnn_video.layers[l+1].trainable = False
        x1 = model_cnn_video.layers[l+1](x1)

    # Get all layer of the CNN audio until Flatten
    input2 = model_cnn_audio.input
    x2 = model_cnn_audio.layers[0](input2) 
    for l in range(shape(nb_filters_audio)[0]*4+1):
        #model_cnn_audio.layers[l+1].trainable = False
        x2 = model_cnn_audio.layers[l+1](x2)
    

    # Fusion
    added = Concatenate()([x1, x2])  

    for l in range(shape(arch)[0]):
        added = Dense(arch[l],kernel_initializer=initializer, activation=activation_function)(added) 
        added = Dropout(dropout_ratio)(added)

    out = Dense(model_cnn_audio.output_shape[1], activation='linear')(added)

    model = Model(inputs=[input1, input2], outputs=out)

    if verbosity:
        model.summary()

    return model
#######################################################

def do_train(output_dir, all_train_video_dir, all_train_audio_filenames, tp, tf):
    print('Preparing training data')
    video_train_data_in, audio_train_data_in, video_train_data_out, audio_train_data_out  = load_data(all_train_video_dir,all_train_audio_filenames, tp,tf)
    print('Training set: %i video frames loaded' % shape(video_train_data_in)[0])
    print('Training set: %i audio frames loaded' % shape(audio_train_data_in)[0])
    
    # Normalize training and test data (FIXME: do it indepndently for each speaker??)
    audio_min_max_scaler_in = MinMaxScaler()#StandardScaler()#MinMaxScaler()
    audio_train_data_in_norm = audio_min_max_scaler_in.fit_transform(audio_train_data_in)

    audio_min_max_scaler_out = MinMaxScaler()#StandardScaler() #MinMaxScaler()
    audio_train_data_out_norm = audio_min_max_scaler_out.fit_transform(audio_train_data_out) # FIXME: use the same normalizer for input and output audio data (bug with the dimension)
    
    # Building and training audio model
    print('Training audio model (CNN)')
    model_audio = build_cnn_audio((n_fft/2)+1,tp,audio_train_data_out_norm.shape[1],nb_filters_audio,kernel_size_1D,pooling_factor_1D,cnn_arch_fc_audio,dropout_ratio,activation_function,verbose)

    if tp==0:
        audio_train_data_in_cnn = reshape(audio_train_data_in_norm,(shape(audio_train_data_in)[0],(n_fft/2)+1,1,1))
    else:
        audio_train_data_in_cnn = reshape(audio_train_data_in_norm,(shape(audio_train_data_in)[0],(n_fft/2)+1,tp+1,1))

    model_audio.compile(loss='mse', optimizer=opt)
    history_audio = model_audio.fit(audio_train_data_in_cnn, audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])

    # Get a backup of model_audio
    model_audio_copy = clone_model(model_audio)
    model_audio_copy.set_weights(model_audio.get_weights())
    
    # Building and training video model
    print('Training video model (CNN)')
    model_video = build_cnn_video(target_im_size,tp,audio_train_data_out_norm.shape[1],nb_filters_video,kernel_size_2D,pooling_factor_2D,cnn_arch_fc_video,dropout_ratio,activation_function,verbose)

    if tp==0:
        video_train_data_in_cnn = reshape(video_train_data_in,(shape(video_train_data_in)[0],target_im_size[0],target_im_size[1],1))
    else:
        video_train_data_in_cnn = reshape(video_train_data_in,(shape(video_train_data_in)[0],tp+1,target_im_size[0],target_im_size[1],1))

    model_video.compile(loss='mse', optimizer=opt)
    history_video = model_video.fit(video_train_data_in_cnn, audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])

    model_video_copy = clone_model(model_video)
    model_video_copy.set_weights(model_video.get_weights())

    # Building and training audio_video model
    print('Training audio_video model (CNN-DNN) (with pretraining)')
    model_audio_video = build_multistream_cnn_pretraining(model_video,model_audio,nb_filters_video,nb_filters_audio,cnn_arch_fc_audio_video,dropout_ratio,activation_function,verbose)

    model_audio_video.compile(loss='mse', optimizer=opt)
    history_audio_video = model_audio_video.fit([video_train_data_in_cnn,audio_train_data_in_cnn], audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])
    
    return model_audio_copy, model_video_copy, model_audio_video, audio_min_max_scaler_in, audio_min_max_scaler_out

###################################
def do_test(output_dir, all_video_dir, all_audio_feature_filenames, tp, tf, audio_min_max_scaler_in,audio_min_max_scaler_out,model_audio,model_video,model_audio_video):

    all_rmse_audio = zeros((shape(all_audio_feature_filenames)[0]))
    all_rmse_video = zeros((shape(all_video_dir)[0]))
    all_rmse_audio_video = zeros((shape(all_audio_feature_filenames)[0]))
    all_evr_audio = zeros((shape(all_audio_feature_filenames)[0]))
    all_evr_video = zeros((shape(all_video_dir)[0]))
    all_evr_audio_video = zeros((shape(all_audio_feature_filenames)[0]))
    
    bar = Bar('Processing ', max=shape(all_audio_feature_filenames)[0])
    for f in range(shape(all_audio_feature_filenames)[0]):
	audio_feature_filename = all_audio_feature_filenames[f]
        video_dir = all_video_dir[f]

        audio_features_with_past_context, audio_features_target, video_features_with_past_context, video_features_target = load_data_single_sentence(video_dir, audio_feature_filename,tp,tf)
        
        audio_features_in_norm = audio_min_max_scaler_in.transform(audio_features_with_past_context) 

        if tp==0:
            video_test_data_in_cnn = reshape(video_features_with_past_context,(shape(video_features_with_past_context)[0],target_im_size[0],target_im_size[1],1))
            audio_test_data_in_cnn = reshape(audio_features_in_norm,(shape(audio_features_in_norm)[0],(n_fft/2)+1,1,1))
        else:
            video_test_data_in_cnn = reshape(video_features_with_past_context,(shape(video_features_with_past_context)[0],tp+1,target_im_size[0],target_im_size[1],1))
            audio_test_data_in_cnn = reshape(audio_features_in_norm,(shape(audio_features_in_norm)[0],(n_fft/2)+1,tp+1,1))

        # Predict
        audio_features_out_predict_from_audio = model_audio.predict(audio_test_data_in_cnn, batch_size=batch_size)
        audio_features_out_predict_from_video = model_video.predict(video_test_data_in_cnn, batch_size=batch_size)
        audio_features_out_predict_from_audio_video = model_audio_video.predict([video_test_data_in_cnn, audio_test_data_in_cnn], batch_size=batch_size)
        
        # Denorm predicted values
        audio_features_out_predict_from_audio_denorm = audio_min_max_scaler_out.inverse_transform(audio_features_out_predict_from_audio)
        audio_features_out_predict_from_video_denorm = audio_min_max_scaler_out.inverse_transform(audio_features_out_predict_from_video)
        audio_features_out_predict_from_audio_video_denorm = audio_min_max_scaler_out.inverse_transform(audio_features_out_predict_from_audio_video)
        
        # Calculate RMSE
        rmse_audio = mean_squared_error(audio_features_target, audio_features_out_predict_from_audio_denorm)
        rmse_video = mean_squared_error(audio_features_target, audio_features_out_predict_from_video_denorm)
        rmse_audio_video = mean_squared_error(audio_features_target, audio_features_out_predict_from_audio_video_denorm)
    
        # Calculate the Explained variance regression score (MSE/VAR)
        evr_audio = explained_variance_score(audio_features_target, audio_features_out_predict_from_audio_denorm,multioutput='variance_weighted')
        evr_video = explained_variance_score(audio_features_target, audio_features_out_predict_from_video_denorm,multioutput='variance_weighted')
        evr_audio_video = explained_variance_score(audio_features_target, audio_features_out_predict_from_audio_video_denorm,multioutput='variance_weighted')

        # Save predictions and errors for all files 
        if step_export_mse_per_frame:
            # Calculate MSE between audio and audiovisual prediction for each frame of a given sentence 
            current_sentence_mse_audio = np.zeros((shape(audio_features_out_predict_from_audio_denorm)[0]))
            current_sentence_mse_audio_video = np.zeros((shape(audio_features_out_predict_from_audio_denorm)[0]))
            for fr in range(shape(audio_features_out_predict_from_audio_denorm)[0]):
                current_sentence_mse_audio[fr]=mean_squared_error(audio_features_target[fr,:],audio_features_out_predict_from_audio_denorm[fr,:])
                current_sentence_mse_audio_video[fr]=mean_squared_error(audio_features_target[fr,:],audio_features_out_predict_from_audio_video_denorm[fr,:])

            # Load phonetic labels
            (froot,sname) = os.path.split(audio_feature_filename)
            (froot,speaker_name) = os.path.split(froot)
            audio_basename = speaker_name  + '/' + sname

            # Load phonetic labels (from MLF file)
            current_sentence_start, current_sentence_stop, current_sentence_labels = extract_labels_from_mlf_ntcdtimit(audio_basename, mlf_filename)
            first_frame_to_keep = max(0,int(round((float(current_sentence_stop[0]*10e-8)*float(audio_fs)/float(hop_length)))) - offset)
            last_frame_to_keep = int(round((float(current_sentence_start[-1]*10e-8)*float(audio_fs)/float(hop_length)))) + offset

            # give a label to each frame given frame index and sampling rate 
            label_vect_current_sentence = []
            for l in range(shape(current_sentence_labels)[0]):
                nb_frame_current_phoneme = round((current_sentence_stop[l]-current_sentence_start[l])* 10**(-8) * float(hop_length))
                label_vect_current_phoneme  = np.matlib.repeat(current_sentence_labels[l], int(nb_frame_current_phoneme))
                label_vect_current_sentence = np.append(label_vect_current_sentence,label_vect_current_phoneme)

            # Crop label vector (silence, tp, tf)
            label_vect_current_sentence = label_vect_current_sentence[range(first_frame_to_keep+tp,min(last_frame_to_keep-tp+tf,shape(label_vect_current_sentence)[0]))]

            nb_frames_current_sentence = shape(current_sentence_mse_audio)[0]
            nb_labels_current_sentence = shape(label_vect_current_sentence)[0]
            if nb_frames_current_sentence > nb_labels_current_sentence:
                current_sentence_mse_audio = current_sentence_mse_audio[range(nb_labels_current_sentence)]
                current_sentence_mse_audio_video = current_sentence_mse_audio[range(nb_labels_current_sentence)]

            if nb_frames_current_sentence < nb_labels_current_sentence:
                label_vect_current_sentence = label_vect_current_sentence[range(nb_frames_current_sentence)]
            
            # Save prediction results for each sentence in a seperate text file (for database release)
            (sname_no_ext,ext) = os.path.splitext(sname)
            data_to_write = np.zeros(shape(current_sentence_mse_audio)[0], dtype=[('var1', 'U6'), ('var2', float) , ('var3', float)])
            data_to_write['var1'] = label_vect_current_sentence
            data_to_write['var2'] = current_sentence_mse_audio
            data_to_write['var3'] = current_sentence_mse_audio_video

            if isdir(output_dir + '/pred_db/') is False:
                mkdir(output_dir + '/pred_db/')

            np.savetxt(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) + '.txt',data_to_write,fmt='%s\t%.3f\t%.3f')
            # save data in binary (numpy) format to limit dataset size
            np.save(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) + '_logspectro_orig.npy',audio_features_target)
            np.save(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) +'_logspectro_pred_audio.npy',audio_features_out_predict_from_audio_denorm)
            np.save(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) + '_logspectro_pred_audio_video.npy',audio_features_out_predict_from_audio_video_denorm)
            
            # Build Figure 5 
            if 0:#to build Figure 5:0--> f==29:
                # Load audio waveform
                y, sr = librosa.load(audio_root_dir + speaker_name + '/straightcam/' + sname[:-4] + '.wav',sr=None)
            
                # Display
                plt.figure()
                plt.subplot(7, 1, 1)
                plt.title('(a)',fontsize=10)
                plt.hold(True)
                valid_sample_after_offset = range(int((first_frame_to_keep+tp)*float(hop_length)),int((last_frame_to_keep-tp+tf)*float(hop_length)))
                y_crop = y[valid_sample_after_offset]
                t_data = np.arange(0,shape(y_crop)[0]/float(sr),1/float(sr))
                plt.plot(t_data, y_crop) 
                plt.xlim((0,t_data[-1]))
                plt.yticks(())
                # plot phonetic labels (and temporal segmentation)
                foo_bkp = 0
                for l in range(shape(current_sentence_labels)[0]):
                    foo = float(current_sentence_stop[l]*10e-8)-(first_frame_to_keep+tp)*float(hop_length)/float(audio_fs)
                    plt.plot([foo,foo],[np.min(y_crop), np.max(y_crop)],'r')
                    plt.text(foo-0.9*(foo-foo_bkp),0.75*np.max(y_crop),str(current_sentence_labels[l]))
                    foo_bkp = foo

                # display some lip images
                plt.subplot(7, 1, 2)
                plt.title('(b)',fontsize=10)
                import torch
                import torchvision.transforms as transforms
                from torchvision.utils import make_grid
                imglist=[]
                for fr in range(1,shape(video_features_with_past_context)[0],2):
                    imglist.append(transforms.ToTensor()(np.reshape(np.round(video_features_with_past_context[fr,0:1024]*255),(32,32,1))))

                show(make_grid(imglist, nrow=range(shape(video_features_target)[0]),padding=0))
                plt.xticks(())
                plt.yticks(())
            
                plt.subplot(7, 1, 3)
                y_max = 4000 # in Hz
                librosa.display.specshow(audio_features_target.transpose(), y_axis='linear',sr=audio_fs)
                plt.ylim((0, y_max)) 
                plt.title('(c)',fontsize=10)

                plt.subplot(7, 1, 4)
                librosa.display.specshow(audio_features_out_predict_from_audio_denorm.transpose(), y_axis='linear',sr=audio_fs)
                plt.ylim((0, y_max)) 
                plt.title('(d)',fontsize=10)

                plt.subplot(7, 1, 5)
                librosa.display.specshow(audio_features_out_predict_from_video_denorm.transpose(), y_axis='linear',sr=audio_fs)
                plt.ylim((0, y_max)) 
                plt.title('(e)',fontsize=10)

                plt.subplot(7, 1, 6)
                librosa.display.specshow(audio_features_out_predict_from_audio_video_denorm.transpose(), y_axis='linear',sr=audio_fs)
                plt.ylim((0, y_max)) 
                plt.title('(f)',fontsize=10)

                plt.subplot(7, 1, 7)
                plt.hold(True)
                plt.plot(current_sentence_mse_audio,'b');
                plt.plot(current_sentence_mse_audio_video,'r');
                plt.xlim((0, np.shape(audio_features_out_predict_from_audio_denorm)[0]))
                plt.title('(g)',fontsize=10)
                plt.legend(('audio','audio+video'))
                plt.xticks(())

                matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
            
                plt.show()
                plt.savefig(output_dir + 'ex_reconstructed_spectro_' + str(tp) + '_'  + str(tf) + '.eps',format='eps', dpi=600)


        #############END DEBUG
            
        
        all_rmse_audio[f] = rmse_audio
        all_rmse_video[f] = rmse_video 
        all_rmse_audio_video[f] = rmse_audio_video 
        all_evr_audio[f] = evr_audio
        all_evr_video[f] = evr_video 
        all_evr_audio_video[f] = evr_audio_video 

        bar.next()

    bar.finish()
    return all_rmse_audio,all_rmse_video, all_rmse_audio_video, all_evr_audio, all_evr_video, all_evr_audio_video
############
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
##################################
def write_config_log():

    if isdir(output_dir_root) is False:
        mkdir(output_dir_root)
        
    now = datetime.datetime.now()
    f = open(output_dir_root + 'log_config.txt','w')
    f.write('Experiment started at:')
    f.write(now.strftime("%Y-%m-%d %H:%M\n"))

    f.write('\nDirectories\n')
    f.write('-----------\n')
    f.write('output_dir_root=%s\n' % output_dir_root)
    f.write('audio_root_dir=%s\n' % audio_root_dir)
    f.write('video_root_dir=%s\n' % video_root_dir)
    f.write('stft_dir=%s\n' % stft_dir)
    f.write('\n')

    f.write('STFT analysis\n')
    f.write('-------------\n')
    f.write('audio_fs=%i\n' % audio_fs)
    f.write('n_fft=%i\n' % n_fft)
    f.write('window=%s\n' % window)
    f.write('win_length=%i\n' % win_length)
    f.write('hop_length=%i\n' % hop_length)
    f.write('\n')
    
    f.write('Audio model (CNN)\n')
    f.write('-----------------\n')
    f.write('target_im_size=(%i,%i)\n' % (target_im_size[0],target_im_size[1]))
    f.write('nb_filters_audio:')
    for k in range(shape(nb_filters_audio)[0]):
        f.write('%i ' % nb_filters_audio[k])

    f.write('\nkernel_size_2D=%i\n' % kernel_size_2D)
    f.write('pooling_factor_2D=%i\n' % pooling_factor_2D)
    f.write('cnn_arch_fc_audio: ')
    for k in range(shape(cnn_arch_fc_audio)[0]):
        f.write('%i ' % cnn_arch_fc_audio[k])
    f.write('\n\n')
    
    f.write('Video model (CNN)\n')
    f.write('-----------------\n')
    f.write('target_im_size=(%i,%i)\n' % (target_im_size[0],target_im_size[1]))
    f.write('nb_filters_video:')
    for k in range(shape(nb_filters_video)[0]):
        f.write('%i ' % nb_filters_video[k])

    f.write('\nkernel_size_2D=%i\n' % kernel_size_2D)
    f.write('pooling_factor_2D=%i\n' % pooling_factor_2D)
    f.write('cnn_arch_fc_video: ')
    for k in range(shape(cnn_arch_fc_video)[0]):
        f.write('%i ' % cnn_arch_fc_video[k])
    f.write('\n\n')

    f.write('Audio/video model architecture (DNN/CNN)\n')
    f.write('----------------------------------------\n')
    f.write('cnn_arch_fc_audio_video: ')
    for k in range(shape(cnn_arch_fc_audio_video)[0]):
        f.write('%i ' % cnn_arch_fc_audio_video[k])
    f.write('\n\n')
        
    f.write('Training parameters\n')
    f.write('-------------------\n')
    f.write('max_nb_frames=%i\n' % max_nb_frames)
    f.write('activation_function=%s\n' % activation_function)
    f.write('nb_epoch=%i\n' % nb_epoch)
    f.write('batch_size=%i\n' % batch_size)
    f.write('opt=%s\n' % opt)
    f.write('initializer=%s\n' % initializer)
    f.write('dropout_ratio=%f\n' % dropout_ratio)
    f.write('early_stopping_patience=%i\n' % early_stopping_patience)
    f.write('validation_split_frac=%f\n' % validation_split_frac)
    f.write('nb_folds=%i\n' % nb_folds)
    f.write('train_test_ratio=%f\n' % train_test_ratio)
    f.write('\n\n')

    f.write('Past/future context\n')
    f.write('---------------\n')
    f.write('all_tau_p: ')
    for k in range(shape(all_tau_p)[0]):
        f.write('%i ' % all_tau_p[k])

    f.write('\nall_tau_f: ')
    for k in range(shape(all_tau_f)[0]):
        f.write('%i ' % all_tau_f[k])
    f.write('\n\n')

    f.write('Silence removal\n')
    f.write('---------------\n')
    f.write('step_remove_silence=%i\n' % step_remove_silence)
    f.write('mlf_filename=%s\n' % mlf_filename)
    f.write('offset=%i\n' % offset)


    f.write('END OF LOG FILE\n\n')
    f.close()
    
    return


############################
if __name__ == '__main__':

    # Write log to store all configuration variables
    write_config_log()
    
    # Extract audio features)
    if step_extract_stft: 
        print('Extracting STFT')
        extract_stft_ntcdtimit_librosa(audio_root_dir, stft_dir)
    
    main_multispeaker()

