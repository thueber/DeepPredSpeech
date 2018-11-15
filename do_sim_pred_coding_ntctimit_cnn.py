#!/Usr/bin/env python2
# -*- coding: utf-8 -*-
"""
DeepPredSpeech: Computational models of predictive speech coding based on deep learning 
Main script for building models based on MFCC-spectrogram (see associated article for more info)
T. Hueber - CNRS/GIPSA-lab - 2018
thomas.hueber@gipsa-lab.fr
"""
from __future__ import print_function
import numpy as np
import scipy
import scipy.io as sio
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import os
import librosa
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
from keras.layers import Input, Dense, Activation, Merge, Flatten, Dropout, BatchNormalization, Concatenate, Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
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
output_dir_root = '/localdata/huebert/res_erc_ntcdtimit_cnn_multispeaker_16k/' 
#output_dir_root = '/localdata/huebert/res_erc_ntcdtimit_cnn_monospeaker_adapt_v2/' 

# Audio data and video data
audio_root_dir = '/localdata/huebert/data/NTCD-timit/audio_clean/'
video_root_dir = '/localdata/huebert/data/NTCD-timit/lips_roi/'

# Corpus parameters
nb_sentences_per_speaker = 98

# Audio analysis (MFCC)
mfcc_dir = '/localdata/huebert/data/NTCD-timit/mfcc_16k/'
audio_fs = 16000
n_mels = 40
n_fft = 512
window = 'hamming'
fmin = 20
fmax = 8000
n_mfcc = 13
win_length = 400 #i.e. 0.025 ms at 16kHz
hop_length = 400 #i.e. 0.025 ms at 16kHz

# TensorFlow wizardry
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # Don't pre-allocate memory; allocate as-needed
config.gpu_options.per_process_gpu_memory_fraction = 1.0 # Only allow a total of half the GPU memory to be allocated
k.tensorflow_backend.set_session(tf.Session(config=config)) # Create a session with the above options

# DNN architecture 
dnn_architecture = [256,256,256] # [nb_neurons_layer_1, nb_neurons_layer_2, etc.] 

# Video model (CNN)
target_im_size = (32,32) #in pixels
nb_filters = [16,32,64]     #8
kernel_size_2D = 3   
pooling_factor = 2
cnn_arch_fc = [256] #(number of layers/neurons after flattening) 

# Audio-video model (number of layers/neurons after fusion) 
cnn_dnn_arch = [256]#[128,128]

# Training parameters 
max_nb_frames = 500000 # Memory pre-allocation for data loading
activation_function = 'tanh'
nb_epoch = 1000
batch_size = 256
opt = 'adam'
initializer = 'random_normal'
dropout_ratio = 0.25
verbose = 1
early_stopping_patience = 10
my_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
validation_split_frac = 0.2 # Fraction of the training set using for validation/early_stopping 
nb_folds = 3 # K-fold cross-validation (depreciated)
train_test_ratio = 0.33 # Fraction of the database used for training+validation on one hand, and test for the other hand 
train_test_ratio_monospeaker_adapt = 0.25 # Fraction of the training set used for adaptation for the speaker-adapted training 

# Past context to use (in number of frames)
all_tau_p =[0,1,2,3]

# Future context to predict (in number of frames)
all_tau_f = [0,1,2,3,4,6,10]

# Silence removal
mlf_filename = '/localdata/huebert/data/NTCD-timit/volunteer_labelfiles.mlf'
offset = 6 # in frames (i.e. 150 ms)

# Steps
step_extract_mfcc = 0
step_remove_silence = 1
step_display = 0
training_mode = 'multispeaker' # or 'speaker_adapted'

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
        
    print('Scanning audio features directory %s ...' % mfcc_dir)
    all_mfcc_speakers_fullpath = sorted(listdir_fullpath(mfcc_dir), key=numericalSort)
    all_mfcc_filenames = [];
    for s in range(shape(all_mfcc_speakers_fullpath)[0]):
        current_speaker_audio_filenames = sorted(glob.glob(all_mfcc_speakers_fullpath[s] + '/*.npy'), key=numericalSort)
        all_mfcc_filenames = np.append(all_mfcc_filenames,current_speaker_audio_filenames,axis=0)
    print('%i files found\n' % shape(all_mfcc_filenames)[0])

    print('Scanning video directory %s' % video_root_dir)
    all_im_speakers = sorted(listdir_fullpath(video_root_dir), key=numericalSort)
    all_im_dir = []
    for s in range(shape(all_im_speakers)[0]):
        current_speaker_im_dir = sorted(listdir_fullpath(join(video_root_dir,all_im_speakers[s])), key=numericalSort)
        all_im_dir = np.append(all_im_dir,current_speaker_im_dir,axis=0)

    print('%i files found\n' % shape(all_im_dir)[0])

    # Main K-fold loop
    train_ind, test_ind = train_test_split(range(shape(all_mfcc_filenames)[0]),test_size=train_test_ratio, random_state=seed2)
    #kf = KFold(n_splits=nb_folds)
    current_fold_index = 1
    if 1:
    #for train_ind, test_ind in kf.split(range(shape(all_mfcc_filenames)[0])):
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
                model_audio, model_video, model_audio_video, audio_min_max_scaler_in, audio_min_max_scaler_out  = do_train(output_dir, all_im_dir[[train_ind]], all_mfcc_filenames[[train_ind]], tau_p, tau_f, dnn_architecture)
                    
                # Save models (for multispeaker experiments only)
                model_audio.save(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                model_video.save(output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                model_audio_video.save(output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                joblib.dump(audio_min_max_scaler_in, output_dir + '/audio_min_max_scaler_in_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                joblib.dump(audio_min_max_scaler_out, output_dir + '/audio_min_max_scaler_out_' + str(tau_p) + '_' + str(tau_f) +'.dat')

                # Plot models architecture using Keras tools
                #plot_model(model_audio, to_file=output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                #plot_model(model_video, to_file=output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                #plot_model(model_audio_video, to_file=output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                    
                # TEST
                model_audio = load_model(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                model_video = load_model(output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                model_audio_video = load_model(output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                audio_min_max_scaler_in = joblib.load(output_dir + '/audio_min_max_scaler_in_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                audio_min_max_scaler_out = joblib.load(output_dir + '/audio_min_max_scaler_out_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                
                all_rmse_audio_current_exp,all_rmse_video_current_exp, all_rmse_audio_video_current_exp, all_evr_audio_current_exp, all_evr_video_current_exp, all_evr_audio_video_current_exp = do_test(output_dir, all_im_dir[[test_ind]],all_mfcc_filenames[[test_ind]], tau_p, tau_f, audio_min_max_scaler_in,audio_min_max_scaler_out, model_audio, model_video, model_audio_video)

                all_rmse_audio[p,f,:] = all_rmse_audio_current_exp
                all_rmse_video[p,f,:] = all_rmse_video_current_exp 
                all_rmse_audio_video[p,f,:] = all_rmse_audio_video_current_exp 
                all_evr_audio[p,f,:] = all_evr_audio_current_exp
                all_evr_video[p,f,:] = all_evr_video_current_exp 
                all_evr_audio_video[p,f,:] = all_evr_audio_video_current_exp 

                print('RMSE audio = %f / RMSE video = %f / RMSE audio_video = %f ' % (np.mean(all_rmse_audio_current_exp),np.mean(all_rmse_video_current_exp),np.mean(all_rmse_audio_video_current_exp)))
                print('EVR audio = %f / EVR video = %f / EVR audio_video = %f ' % (np.mean(all_evr_audio_current_exp),np.mean(all_evr_video_current_exp), np.mean(all_evr_audio_video_current_exp)))
                    
                sim_iter = sim_iter + 1

                # Save results in numpy format
                np.save(output_dir + '/all_rmse_audio_' + str(current_fold_index) + '.mat',all_rmse_audio)
                np.save(output_dir + '/all_rmse_video_' + str(current_fold_index) + '.mat',all_rmse_video)
                np.save(output_dir + '/all_rmse_audio_video_' + str(current_fold_index) + '.mat',all_rmse_audio_video)
                np.save(output_dir + '/all_evr_audio_' + str(current_fold_index) + '.mat',all_evr_audio)
                np.save(output_dir + '/all_evr_video_' + str(current_fold_index) + '.mat',all_evr_video)
                np.save(output_dir + '/all_evr_audio_video_' + str(current_fold_index) + '.mat',all_evr_audio_video)

        current_fold_index = current_fold_index + 1
############################################
############################################

def main_monospeaker_adapt(): 
    # Create output directory if necessary
    if isdir(output_dir_root) is False:
        mkdir(output_dir_root)

    if isdir(output_dir_root + '/monospeaker_adapt/') is False:
        mkdir(output_dir_root + '/monospeaker_adapt/')

    all_mfcc_speakers_fullpath = sorted(listdir_fullpath(mfcc_dir), key=numericalSort)
    all_speakers_im_dir_fullpath = sorted(listdir_fullpath(video_root_dir), key=numericalSort)

    # Define the speakers used for training the multispeaker model and those used for testin
    train_speaker_ind, test_speaker_ind = train_test_split(range(shape(all_mfcc_speakers_fullpath)[0]-1),test_size=train_test_ratio_monospeaker_adapt, random_state=seed2)
    print('Nb speakers for training/adapting the multispeaker model:%i/%i' % (shape(train_speaker_ind)[0],shape(test_speaker_ind)[0]))

    train_ind, test_ind = train_test_split(range(nb_sentences_per_speaker),test_size=train_test_ratio, random_state=seed2)

    # Train the multispeaker model
    all_mfcc_filenames = []
    for s in range(shape(train_speaker_ind)[0]):
        current_speaker_audio_filenames = np.asarray(sorted(glob.glob(all_mfcc_speakers_fullpath[train_speaker_ind[s]] + '/*.npy'), key=numericalSort))
        all_mfcc_filenames = np.append(all_mfcc_filenames,current_speaker_audio_filenames[train_ind],axis=0)
    print('%i training audio files found\n' % shape(all_mfcc_filenames)[0])

    all_im_speakers = sorted(listdir_fullpath(video_root_dir), key=numericalSort)
    all_im_dir = []
    for s in range(shape(train_speaker_ind)[0]):
        current_speaker_im_dir = np.asarray(sorted(listdir_fullpath(join(video_root_dir,all_im_speakers[train_speaker_ind[s]])), key=numericalSort))
        all_im_dir = np.append(all_im_dir,current_speaker_im_dir[train_ind],axis=0)
    print('%i training image directories found\n' % shape(all_im_dir)[0])

    current_fold_index  = 1
    output_dir = (output_dir_root + '/monospeaker_adapt/')

    # define train and test sentences for monospeaker experiments (assuming the same partioning for all speakers)
    train_ind, test_ind = train_test_split(range(shape(current_speaker_audio_filenames)[0]),test_size=train_test_ratio, random_state=seed2)
    
    all_rmse_audio = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0],shape(test_speaker_ind)[0]))
    all_rmse_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0],shape(test_speaker_ind)[0]))
    all_rmse_audio_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0],shape(test_speaker_ind)[0]))
    all_evr_audio = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0],shape(test_speaker_ind)[0]))
    all_evr_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0],shape(test_speaker_ind)[0]))
    all_evr_audio_video = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0],shape(test_speaker_ind)[0]))
            
    sim_iter = 1
    for p in range(np.shape(all_tau_p)[0]):
        tau_p = all_tau_p[p]
        for f in range(np.shape(all_tau_f)[0]):
            tau_f = all_tau_f[f]
            print('\n\nSimulation %i/%i (tau_p = %i / tau_f = %i / speaker %i)' % (sim_iter,np.shape(all_tau_p)[0]*np.shape(all_tau_f)[0],s,tau_p,tau_f)) # FIXME 

            # TRAIN
            model_audio, model_video, model_audio_video, audio_min_max_scaler_in, audio_min_max_scaler_out  = do_train(output_dir, all_im_dir, all_mfcc_filenames, tau_p, tau_f, dnn_architecture)
                    
            # Save models (for multispeaker experiments only)
            model_audio.save(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
            model_video.save(output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
            model_audio_video.save(output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
            joblib.dump(audio_min_max_scaler_in, output_dir + '/audio_min_max_scaler_in.dat')
            joblib.dump(audio_min_max_scaler_out, output_dir + '/audio_min_max_scaler_out.dat')
            
            # Plot models
            #plot_model(model_audio, to_file=output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
            #plot_model(model_video, to_file=output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
            #plot_model(model_audio_video, to_file=output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')

            #model_audio = load_model(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
            #model_video = load_model(output_dir + '/model_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
            #model_audio_video = load_model(output_dir + '/model_audio_video_' + str(tau_p) + '_' + str(tau_f) + '.h5')
            #audio_min_max_scaler_in = joblib.load(output_dir + '/audio_min_max_scaler_in.dat')
            #audio_min_max_scaler_out = joblib.load(output_dir + '/audio_min_max_scaler_out.dat')

            # Speaker adaptation
            for s in range(shape(test_speaker_ind)[0]): 
                print('Fine tuning mutlispeaker model on speaker %i' % test_speaker_ind[s])

                current_test_speaker_audio_filenames = np.asarray(sorted(glob.glob(all_mfcc_speakers_fullpath[test_speaker_ind[s]] + '/*.npy'), key=numericalSort))
                current_test_speaker_im_dir = np.asarray(sorted(listdir_fullpath(all_speakers_im_dir_fullpath[test_speaker_ind[s]]), key=numericalSort))

                model_audio_adapted, model_video_adapted, model_audio_video_adapted = do_adapt(output_dir, current_test_speaker_im_dir[[train_ind]], current_test_speaker_audio_filenames[[train_ind]], tau_p, tau_f, audio_min_max_scaler_in, audio_min_max_scaler_out, model_audio, model_video, model_audio_video)
            
                # TEST
                all_rmse_audio_current_exp,all_rmse_video_current_exp, all_rmse_audio_video_current_exp, all_evr_audio_current_exp, all_evr_video_current_exp, all_evr_audio_video_current_exp = do_test(output_dir, current_test_speaker_im_dir[[test_ind]],current_test_speaker_audio_filenames[[test_ind]], tau_p, tau_f, audio_min_max_scaler_in,audio_min_max_scaler_out, model_audio_adapted, model_video_adapted, model_audio_video_adapted)

                all_rmse_audio[p,f,:,s] = all_rmse_audio_current_exp
                all_rmse_video[p,f,:,s] = all_rmse_video_current_exp 
                all_rmse_audio_video[p,f,:,s] = all_rmse_audio_video_current_exp 
                all_evr_audio[p,f,:,s] = all_evr_audio_current_exp
                all_evr_video[p,f,:,s] = all_evr_video_current_exp 
                all_evr_audio_video[p,f,:,s] = all_evr_audio_video_current_exp 
            
                print('RMSE audio = %f / RMSE video = %f / RMSE audio_video = %f ' % (np.mean(all_rmse_audio_current_exp),np.mean(all_rmse_video_current_exp),np.mean(all_rmse_audio_video_current_exp)))
                print('EVR audio = %f / EVR video = %f / EVR audio_video = %f ' % (np.mean(all_evr_audio_current_exp),np.mean(all_evr_video_current_exp), np.mean(all_evr_audio_video_current_exp)))

                sim_iter = sim_iter + 1
            
    # Save also in numpy format
    np.save(output_dir + '/all_rmse_audio_' + str(current_fold_index) + '.mat',all_rmse_audio)
    np.save(output_dir + '/all_rmse_video_' + str(current_fold_index) + '.mat',all_rmse_video)
    np.save(output_dir + '/all_rmse_audio_video_' + str(current_fold_index) + '.mat',all_rmse_audio_video)
    np.save(output_dir + '/all_evr_audio_' + str(current_fold_index) + '.mat',all_evr_audio)
    np.save(output_dir + '/all_evr_video_' + str(current_fold_index) + '.mat',all_evr_video)
    np.save(output_dir + '/all_evr_audio_video_' + str(current_fold_index) + '.mat',all_evr_audio_video)                

    current_fold_index = current_fold_index + 1
############################################

# Sub-functions
###############
def extract_mfcc_ntcdtimit_librosa(input_dir, target_dir):
    
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

            # Comput MFCC
            D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**2
            S = librosa.feature.melspectrogram(S=D, y=y, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)
            feats = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)

            # Save
            audio_filename_no_root = basename(current_speaker_audio_filenames[f])
            audio_basename = splitext(audio_filename_no_root)[0]
            np.save(target_dir + '/' + all_audio_speakers[s] + '/' + audio_basename + '.npy',feats.transpose())
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
    audio_data_in = np.zeros((n_mfcc*(tp+1),max_nb_frames))
    video_data_in = np.zeros((target_im_size[0]*target_im_size[1]*(tp+1),max_nb_frames))
    audio_data_out = np.zeros((n_mfcc,max_nb_frames))
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

def build_cnn(im_size, tp, nb_output_features, nb_filters, kernel_size_2D, pooling_factor,arch,dropout_ratio,activation_function,verbosity):
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

def build_multistream_cnn_dnn_pretraining(model_cnn,model_dnn,nb_filters,arch,dropout_ratio,activation_function,verbosity):
    # Get all layer of the CNN until Flatten
    input1 = model_cnn.input
    x1 = model_cnn.layers[0](input1) 
    for l in range(shape(nb_filters)[0]*4+1):
        #model_cnn.layers[l+1].trainable = False
        x1 = model_cnn.layers[l+1](x1)
    
    # Get all layers of DNN until Linear output
    input2 = Input(shape=(model_dnn.input_shape[1],))
    #model_dnn.layers[1].trainable = False
    x2 = model_dnn.layers[1](input2) 
    for l in range(2,shape(model_dnn.layers)[0]-1):
        #model_dnn.layers[l].trainable = False
        x2 = model_dnn.layers[l](x2) 

    # Fusion
    added = Concatenate()([x1, x2])  

    for l in range(shape(arch)[0]):
        added = Dense(arch[l],kernel_initializer=initializer, activation=activation_function)(added) 
        added = Dropout(dropout_ratio)(added)

    out = Dense(model_dnn.output_shape[1], activation='linear')(added)

    model = Model(inputs=[input1, input2], outputs=out)

    if verbosity:
        model.summary()

    return model

#######################################################
def build_multistream_cnn_dnn(arch_dnn,nb_input_features_dnn,nb_output_features, im_size, tp, nb_filters, kernel_size_2D, pooling_factor,arch_after_fusion,dropout_ratio,activation_function_dnn,verbosity):

    # Audio branch (DNN)
    input1 = Input(shape=(nb_input_features_dnn,))
    x1 = Dense(arch_dnn[0],kernel_initializer=initializer)(input1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation_function_dnn)(x1)
    x1 = Dropout(dropout_ratio)(x1)

    if shape(arch_dnn)[0]>1:
        for l in range(shape(arch_dnn)[0]-1):
            x1 = Dense(arch_dnn[l+1],kernel_initializer=initializer)(x1)
            x1 = BatchNormalization()(x1)
            x1 = Activation(activation_function_dnn)(x1)
            x1 = Dropout(dropout_ratio)(x1)
        
    # Video branch (CNN)
    if tp==0:
        input_shape = (im_size[0],im_size[1],1)
        kernel_size = (kernel_size_2D,kernel_size_2D)
        pool_size = (pooling_factor,pooling_factor)

        input2 = Input(shape=input_shape)
        
        for l in range(shape(nb_filters)[0]):
            if l == 0:
                x2 = Convolution2D(nb_filters[l], kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer)(input2)
            else:
                x2 = Convolution2D(nb_filters[l], kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer)(x2)
                
            x2 = BatchNormalization()(x2)
	    x2 = LeakyReLU()(x2)
	    x2 = MaxPooling2D(pool_size=pool_size)(x2)

        x2 = Dropout(dropout_ratio)(x2)
        
    else:
        input_shape = (tp+1,im_size[0],im_size[1],1)
        kernel_size = (tp+1,kernel_size_2D,kernel_size_2D)
        pool_size = (1,pooling_factor,pooling_factor)

        input2 = Input(shape=input_shape)
                
        for l in range(shape(nb_filters)[0]):
            if l == 0:
                x2 = Convolution3D(nb_filters[l],kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer)(input2)
            else:
                x2 = Convolution3D(nb_filters[l],kernel_size,border_mode='same',input_shape=input_shape,kernel_initializer=initializer)(x2)

            x2 = BatchNormalization()(x2)
	    x2 = LeakyReLU()(x2)	
	    x2 = MaxPooling3D(pool_size=pool_size)(x2)

        x2 = Dropout(dropout_ratio)(x2)
       
    x2 = Flatten()(x2)
    
    # Audio-Video Fusion
    added = Concatenate()([x2, x1])  

    for l in range(shape(arch_after_fusion)[0]):
        added = Dense(arch_after_fusion[l],kernel_initializer=initializer, activation=activation_function)(added) 
        added = Dropout(dropout_ratio)(added)

    out = Dense(nb_output_features, activation='linear')(added)

    model = Model(inputs=[input2, input1], outputs=out)

    if verbosity:
        model.summary()

    return model
#######################################################

def do_train(output_dir, all_train_video_dir, all_train_audio_filenames, tp, tf, dnn_arch):
    print('Preparing training data')
    video_train_data_in, audio_train_data_in, video_train_data_out, audio_train_data_out  = load_data(all_train_video_dir,all_train_audio_filenames, tp,tf)
    print('Training set: %i video frames loaded' % shape(video_train_data_in)[0])
    print('Training set: %i audio frames loaded' % shape(audio_train_data_in)[0])
    
    # Normalize training and test data (FIXME: do it indepndently for each speaker??)
    audio_min_max_scaler_in = MinMaxScaler()
    audio_train_data_in_norm = audio_min_max_scaler_in.fit_transform(audio_train_data_in)

    audio_min_max_scaler_out = MinMaxScaler()
    audio_train_data_out_norm = audio_min_max_scaler_out.fit_transform(audio_train_data_out) 

    # Building and training audio model
    print('Training audio model (DNN)')
    model_audio = build_dnn(dnn_arch, shape(audio_train_data_in_norm)[1],shape(audio_train_data_out_norm)[1],dropout_ratio,activation_function,verbose)
    model_audio.compile(loss='mse', optimizer=opt)
    history_audio = model_audio.fit(audio_train_data_in_norm, audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])

    # Get a backup of model_audio
    model_audio_copy = clone_model(model_audio)
    model_audio_copy.set_weights(model_audio.get_weights())
    
    # Building and training video model
    print('Training video model (CNN)')
    model_video = build_cnn(target_im_size,tp,audio_train_data_out_norm.shape[1],nb_filters,kernel_size_2D,pooling_factor,cnn_arch_fc,dropout_ratio,activation_function,verbose)

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
    model_audio_video = build_multistream_cnn_dnn_pretraining(model_video,model_audio,nb_filters,cnn_dnn_arch,dropout_ratio,activation_function,verbose)

    #print('Training audio_video model (CNN-DNN)')
    #model_audio_video = build_multistream_cnn_dnn(dnn_arch,shape(audio_train_data_in_norm)[1],shape(audio_train_data_out_norm)[1],target_im_size,tp,nb_filters,kernel_size_2D,pooling_factor,cnn_arch_fc,dropout_ratio,activation_function,verbose)

    model_audio_video.compile(loss='mse', optimizer=opt)
    history_audio_video = model_audio_video.fit([video_train_data_in_cnn,audio_train_data_in_norm], audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])
    
    return model_audio_copy, model_video_copy, model_audio_video, audio_min_max_scaler_in, audio_min_max_scaler_out
############################################

def do_adapt(output_dir, all_train_video_dir, all_train_audio_filenames, tp,tf, audio_min_max_scaler_in, audio_min_max_scaler_out, model_audio,model_video, model_audio_video):

    video_train_data_in, audio_train_data_in, video_train_data_out, audio_train_data_out  = load_data(all_train_video_dir,all_train_audio_filenames, tp,tf)
    print('Adaptation set: %i video frames loaded' % shape(video_train_data_in)[0])
    print('Adaptation set: %i audio frames loaded' % shape(audio_train_data_in)[0])
    
    # Normalize training and test data 
    audio_train_data_in_norm = audio_min_max_scaler_in.transform(audio_train_data_in)
    audio_train_data_out_norm = audio_min_max_scaler_out.transform(audio_train_data_out)

    print('Adapting audio model (DNN)')
    model_audio.compile(loss='mse', optimizer=opt)
    history_audio = model_audio.fit(audio_train_data_in_norm, audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])

    # Get a backup of model_audio
    model_audio_copy = clone_model(model_audio)
    model_audio_copy.set_weights(model_audio.get_weights())
    
    # Building and training video model
    print('Adapting video model (CNN)')
    if tp==0:
        video_train_data_in_cnn = reshape(video_train_data_in,(shape(video_train_data_in)[0],target_im_size[0],target_im_size[1],1))
    else:
        video_train_data_in_cnn = reshape(video_train_data_in,(shape(video_train_data_in)[0],tp+1,target_im_size[0],target_im_size[1],1))

    model_video.compile(loss='mse', optimizer=opt)
    history_video = model_video.fit(video_train_data_in_cnn, audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])

    model_video_copy = clone_model(model_video)
    model_video_copy.set_weights(model_video.get_weights())

    # Building and training audio_video model
    print('Adapting audio_video model (CNN-DNN)')
    model_audio_video.compile(loss='mse', optimizer=opt)
    history_audio_video = model_audio_video.fit([video_train_data_in_cnn,audio_train_data_in_norm], audio_train_data_out_norm, batch_size=batch_size, verbose=verbose, epochs=nb_epoch, validation_split=validation_split_frac, callbacks=[my_callback])
    
    return model_audio_copy, model_video_copy, model_audio_video 
###################################

def do_test(output_dir, all_video_dir, all_audio_feature_filenames, tp, tf, audio_min_max_scaler_in,audio_min_max_scaler_out,model_audio,model_video,model_audio_video):

    all_rmse_audio = zeros((shape(all_audio_feature_filenames)[0]))
    all_rmse_video = zeros((shape(all_video_dir)[0]))
    all_rmse_audio_video = zeros((shape(all_audio_feature_filenames)[0]))
    all_evr_audio = zeros((shape(all_audio_feature_filenames)[0]))
    all_evr_video = zeros((shape(all_video_dir)[0]))
    all_evr_audio_video = zeros((shape(all_audio_feature_filenames)[0]))

    ### only for post-analysis 
    all_frames_mse_audio = []
    all_frames_mse_audio_video = []
    all_frames_labels = []
    #######
    
    bar = Bar('Processing ', max=shape(all_audio_feature_filenames)[0])
    for f in range(shape(all_audio_feature_filenames)[0]):
	audio_feature_filename = all_audio_feature_filenames[f]
        video_dir = all_video_dir[f]

        audio_features_with_past_context, audio_features_target, video_features_with_past_context, video_features_target = load_data_single_sentence(video_dir, audio_feature_filename,tp,tf)
        
        audio_features_in_norm = audio_min_max_scaler_in.transform(audio_features_with_past_context) 

        if tp==0:
            video_test_data_in_cnn = reshape(video_features_with_past_context,(shape(video_features_with_past_context)[0],target_im_size[0],target_im_size[1],1))
        else:
            video_test_data_in_cnn = reshape(video_features_with_past_context,(shape(video_features_with_past_context)[0],tp+1,target_im_size[0],target_im_size[1],1))
        
        # Predict
        audio_features_out_predict_from_audio = model_audio.predict(audio_features_in_norm, batch_size=batch_size)
        audio_features_out_predict_from_video = model_video.predict(video_test_data_in_cnn, batch_size=batch_size)
        audio_features_out_predict_from_audio_video = model_audio_video.predict([video_test_data_in_cnn, audio_features_in_norm], batch_size=batch_size)
        
            
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


        ############## 
        if step_export_mse_per_frame:
            # Calculate MSE between audio and audiovisual prediction for each frame of a given sentence 
            current_sentence_mse_audio = np.zeros((shape(audio_features_out_predict_from_audio_denorm)[0]))
            current_sentence_mse_audio_video = np.zeros((shape(audio_features_out_predict_from_audio_denorm)[0]))
            for fr in range(shape(audio_features_out_predict_from_audio_denorm)[0]):
                current_sentence_mse_audio[fr]=mean_squared_error(audio_features_target[fr,:], audio_features_out_predict_from_audio_denorm[fr,:],)
                current_sentence_mse_audio_video[fr]=mean_squared_error(audio_features_target[fr,:], audio_features_out_predict_from_audio_video_denorm[fr,:])
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
            np.savetxt(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) + '_mfcc_orig.txt',audio_features_target)
            np.savetxt(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) +'_mfcc_pred_audio.txt',audio_features_out_predict_from_audio_denorm)
            np.savetxt(output_dir + '/pred_db/' + speaker_name + '_' + sname_no_ext + '_' + str(tp) + '_' + str(tf) + '_mfcc_pred_audio_video.txt',audio_features_out_predict_from_audio_video_denorm)

            # Concatenate all errors for each phoneme (to build fig_error_per_phoneme figure)
            if f==0: 
                all_frames_mse_audio = current_sentence_mse_audio
                all_frames_mse_audio_video = current_sentence_mse_audio_video
                all_frames_labels = label_vect_current_sentence
            else:
                all_frames_mse_audio = np.concatenate((all_frames_mse_audio, current_sentence_mse_audio))
                all_frames_mse_audio_video = np.concatenate((all_frames_mse_audio_video, current_sentence_mse_audio_video))
                all_frames_labels = np.concatenate((all_frames_labels, label_vect_current_sentence), axis = 0)
        ############## END step_export_mse_per_frame
        
        all_rmse_audio[f] = rmse_audio
        all_rmse_video[f] = rmse_video 
        all_rmse_audio_video[f] = rmse_audio_video 
        all_evr_audio[f] = evr_audio
        all_evr_video[f] = evr_video 
        all_evr_audio_video[f] = evr_audio_video 

        bar.next()

    bar.finish()

    # Save
    if step_export_mse_per_frame:
        np.save(output_dir + '/all_frames_mse_audio_' + str(tp) + '_' + str(tf) + '.npy',all_frames_mse_audio)
        np.save(output_dir + '/all_frames_mse_audio_video_' + str(tp) + '_' + str(tf) + '.npy',all_frames_mse_audio_video)
        np.save(output_dir + '/all_frames_labels_' + str(tp) + '_' + str(tf) + '.npy',all_frames_labels)
    
    return all_rmse_audio,all_rmse_video, all_rmse_audio_video, all_evr_audio, all_evr_video, all_evr_audio_video
############

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
    f.write('mfcc_dir=%s\n' % mfcc_dir)
    f.write('\n')

    f.write('MFCC analysis\n')
    f.write('-------------\n')
    f.write('audio_fs=%i\n' % audio_fs)
    f.write('n_mels=%i\n' % n_mels)
    f.write('n_fft=%i\n' % n_fft)
    f.write('window=%s\n' % window)
    f.write('fmin=%i\n' % fmin)
    f.write('fmax=%i\n' % fmax)
    f.write('n_mfcc=%i\n' % n_mfcc)
    f.write('win_length=%i\n' % win_length)
    f.write('hop_length=%i\n' % hop_length)
    f.write('\n')
    
    f.write('Audio model architecture\n')
    f.write('------------------------\n')
    f.write('dnn_architecture:')
    for k in range(shape(dnn_architecture)[0]):
        f.write('%i ' % dnn_architecture[k])
    f.write('\n\n')
    
    f.write('Video model (CNN)\n')
    f.write('-----------------\n')
    f.write('target_im_size=(%i,%i)\n' % (target_im_size[0],target_im_size[1]))
    f.write('nb_filters:')
    for k in range(shape(nb_filters)[0]):
        f.write('%i ' % nb_filters[k])

    f.write('\nkernel_size_2D=%i\n' % kernel_size_2D)
    f.write('pooling_factor=%i\n' % pooling_factor)
    f.write('cnn_arch_fc: ')
    for k in range(shape(cnn_arch_fc)[0]):
        f.write('%i ' % cnn_arch_fc[k])
    f.write('\n\n')

    f.write('Audio/video model architecture (DNN/CNN)\n')
    f.write('----------------------------------------\n')
    f.write('cnn_dnn_arch: ')
    for k in range(shape(cnn_dnn_arch)[0]):
        f.write('%i ' % cnn_dnn_arch[k])
    f.write('\n\n')
        
    f.write('Training parameters\n')
    f.write('-------------------\n')
    f.write('training_mode=%s\n' % training_mode)
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
    f.write('train_test_ratio_monospeaker_adapt=%f\n' % train_test_ratio_monospeaker_adapt)
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
    
    # Extract MFCC (audio features)
    if step_extract_mfcc: 
        print('Extracting MFCC')
        extract_mfcc_ntcdtimit_librosa(audio_root_dir, mfcc_dir)

    if training_mode == 'multispeaker':
        print('Start experiments using multispeaker training configuration')
        main_multispeaker()

    if training_mode == 'speaker_adapted':
        print('Start experiments using speaker-adapted training configuration')
        main_monospeaker_adapt()
