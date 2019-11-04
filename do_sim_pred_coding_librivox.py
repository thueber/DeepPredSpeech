#!/Usr/bin/env python2
# -*- coding: utf-8 -*-
"""
DeepPredSpeech: Computational models of predictive speech coding based on deep learning 
Main script for building models based on MFCC-spectrogram (see associated article for more info) on Librispeech dataset
T. Hueber - CNRS/GIPSA-lab - 2018
thomas.hueber@gipsa-lab.fr
"""
from __future__ import print_function
import numpy as np
import scipy
import scipy.io as sio
from scipy import fftpack
import matplotlib.pyplot as plt
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
from keras.layers import Input, Dense, Activation, Merge, Flatten, Dropout, BatchNormalization, Concatenate
from keras.callbacks import EarlyStopping, Callback
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.models import load_model, clone_model
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.applications.imagenet_utils import preprocess_input
from keras.layers.advanced_activations import LeakyReLU
import soundfile as sf
import datetime
import pdb # debugger
## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

# CONFIG (edit)
###############
# Output directory
output_dir_root = '/localdata/huebert/res_erc_librivox_v3/' 

# Audio data
audio_root_dir = '/localdata/huebert/data/LibriSpeech/train-clean-100/'

# Audio analysis (MFCC)
mfcc_dir = '/localdata/huebert/data/LibriSpeech/train-clean-100-mfcc/'
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # Don't pre-allocate memory; allocate as-needed
config.gpu_options.per_process_gpu_memory_fraction = 1.0 # Only allow a total of half the GPU memory to be allocated
k.tensorflow_backend.set_session(tf.Session(config=config)) # Create a session with the above options

# DNN architecture 
dnn_architecture = [256,256,256] # [nb_neurons_layer_1, nb_neurons_layer_2, etc.] 

# Training parameters 
max_nb_frames = 10000000 # Memory pre-allocation for data loading
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

# Past context to use (in number of frames)
all_tau_p =[0,1,2,3,4]

# Future context to predict (in number of frames)
all_tau_f = [0,1,2,3,4,6,8,10,14,18]

# Steps
step_extract_mfcc = 1

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
        all_evr_audio = zeros((np.shape(all_tau_p)[0],np.shape(all_tau_f)[0],shape(test_ind)[0]))
        sim_iter = 1
        for p in range(np.shape(all_tau_p)[0]):
            tau_p = all_tau_p[p]
            for f in range(np.shape(all_tau_f)[0]):
                tau_f = all_tau_f[f]
                print('\n\nSimulation %i/%i (folds %i / tau_p = %i / tau_f = %i)' % (sim_iter,np.shape(all_tau_p)[0]*np.shape(all_tau_f)[0]*nb_folds,current_fold_index,tau_p,tau_f))

                # TRAIN
                model_audio, audio_min_max_scaler_in, audio_min_max_scaler_out  = do_train(output_dir, all_mfcc_filenames[[train_ind]], tau_p, tau_f, dnn_architecture)
                    
                # Save models (for multispeaker experiments only)
                model_audio.save(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                joblib.dump(audio_min_max_scaler_in, output_dir + '/audio_min_max_scaler_in_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                joblib.dump(audio_min_max_scaler_out, output_dir + '/audio_min_max_scaler_out_' + str(tau_p) + '_' + str(tau_f) +'.dat')

                # Plot models architecture using Keras tools
                #plot_model(model_audio, to_file=output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.png', show_shapes=True, show_layer_names=True, rankdir='TB')
                    
                # TEST
                model_audio = load_model(output_dir + '/model_audio_' + str(tau_p) + '_' + str(tau_f) + '.h5')
                audio_min_max_scaler_in = joblib.load(output_dir + '/audio_min_max_scaler_in_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                audio_min_max_scaler_out = joblib.load(output_dir + '/audio_min_max_scaler_out_' + str(tau_p) + '_' + str(tau_f) + '.dat')
                
                all_rmse_audio_current_exp,all_evr_audio_current_exp = do_test(output_dir, all_mfcc_filenames[[test_ind]], tau_p, tau_f, audio_min_max_scaler_in,audio_min_max_scaler_out, model_audio)

                all_rmse_audio[p,f,:] = all_rmse_audio_current_exp
                all_evr_audio[p,f,:] = all_evr_audio_current_exp

                print('RMSE audio = %f ' % (np.mean(all_rmse_audio_current_exp)))
                print('EVR audio = %f ' % (np.mean(all_evr_audio_current_exp)))
                    
                sim_iter = sim_iter + 1

                # Save results in numpy format
                np.save(output_dir + '/all_rmse_audio_' + str(current_fold_index) + '.mat',all_rmse_audio)
                np.save(output_dir + '/all_evr_audio_' + str(current_fold_index) + '.mat',all_evr_audio)

        current_fold_index = current_fold_index + 1
############################################
############################################

# Sub-functions
###############
def extract_mfcc_librivox_librosa(input_dir, target_dir):
    
    if isdir(target_dir) is False:
        mkdir(target_dir)

    all_audio_speakers = sorted(listdir(input_dir), key=numericalSort)
    all_audio_speakers_fullpath = sorted(listdir_fullpath(input_dir), key=numericalSort)
    
    for s in range(shape(all_audio_speakers)[0]):
        if isdir(target_dir + '/' + all_audio_speakers[s]) is False:
            mkdir(target_dir + '/' + all_audio_speakers[s])

        current_audio_speaker_fullpath =  all_audio_speakers_fullpath[s]
        current_audio_speaker_all_chapters_fullpath = sorted(listdir_fullpath(current_audio_speaker_fullpath), key=numericalSort)

        print('Processing speaker %s/%i' % (all_audio_speakers[s],shape(all_audio_speakers)[0]))     

        for c in range(shape(current_audio_speaker_all_chapters_fullpath)[0]):
            current_speaker_current_chapter_audio_filenames = glob.glob(current_audio_speaker_all_chapters_fullpath[c] + '/*.flac')

            for f in range(shape(current_speaker_current_chapter_audio_filenames)[0]):
                # Load audio file
                y, sr = librosa.load(current_speaker_current_chapter_audio_filenames[f],sr=None)

                # Comput MFCC
                D = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**2
                S = librosa.feature.melspectrogram(S=D, y=y, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)
                feats = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)

                # Save
                audio_filename_no_root = basename(current_speaker_current_chapter_audio_filenames[f])
                audio_basename = splitext(audio_filename_no_root)[0]
                np.save(target_dir + '/' + all_audio_speakers[s] + '/' + audio_basename + '.npy',feats.transpose())

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

def load_data_single_sentence(audio_feature_filename,tp,tf):
    audio_features = np.load(audio_feature_filename)
    audio_features_with_past_context, audio_features_target = build_inout(audio_features,tp,tf)

    return audio_features_with_past_context, audio_features_target
############################################

def load_data(all_audio_feature_filenames, tp, tf):
    # Pre-allocate memory
    audio_data_in = np.zeros((n_mfcc*(tp+1),max_nb_frames))
    audio_data_out = np.zeros((n_mfcc,max_nb_frames))

    bar = Bar('Processing ', max=shape(all_audio_feature_filenames)[0])
    iter = 0
    for f in range(shape(all_audio_feature_filenames)[0]):
	audio_feature_filename = all_audio_feature_filenames[f]
        #print("Loading data from %s (f=%i)\n" % (audio_feature_filename,f))
        
        audio_features_with_past_context, audio_features_target = load_data_single_sentence(audio_feature_filename,tp,tf)

        if ((iter + shape(audio_features_with_past_context)[0])<max_nb_frames):
            audio_data_in[:,iter:iter + shape(audio_features_with_past_context)[0]] = audio_features_with_past_context.transpose()
            audio_data_out[:,iter:iter + shape(audio_features_target)[0]] = audio_features_target.transpose()

            iter = iter + shape(audio_features_target)[0]
        else:
            print('*** Error: not enough memory allocated (please increase max_nb_frames) ***\n');
            return audio_data_in[:,range(iter)].transpose(), audio_data_out[:,range(iter)].transpose()
            
        bar.next()
    bar.finish()
    print('%i frames loaded' % iter)
    return audio_data_in[:,range(iter)].transpose(), audio_data_out[:,range(iter)].transpose()
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


def do_train(output_dir, all_train_audio_filenames, tp, tf, dnn_arch):
    print('Preparing training data')
    audio_train_data_in, audio_train_data_out  = load_data(all_train_audio_filenames, tp,tf)
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

    return model_audio, audio_min_max_scaler_in, audio_min_max_scaler_out
############################################


def do_test(output_dir, all_audio_feature_filenames, tp, tf, audio_min_max_scaler_in,audio_min_max_scaler_out,model_audio):

    all_rmse_audio = zeros((shape(all_audio_feature_filenames)[0]))
    all_evr_audio = zeros((shape(all_audio_feature_filenames)[0]))

    ### only for post-analysis 
    all_frames_mse_audio = []
    all_frames_labels = []
    #######
    
    bar = Bar('Processing ', max=shape(all_audio_feature_filenames)[0])
    for f in range(shape(all_audio_feature_filenames)[0]):
	audio_feature_filename = all_audio_feature_filenames[f]

        audio_features_with_past_context, audio_features_target = load_data_single_sentence(audio_feature_filename,tp,tf)
        
        audio_features_in_norm = audio_min_max_scaler_in.transform(audio_features_with_past_context) 

        # Predict
        audio_features_out_predict_from_audio = model_audio.predict(audio_features_in_norm, batch_size=batch_size)
            
        # Denorm predicted values
        audio_features_out_predict_from_audio_denorm = audio_min_max_scaler_out.inverse_transform(audio_features_out_predict_from_audio)

        # Calculate RMSE
        rmse_audio = mean_squared_error(audio_features_target, audio_features_out_predict_from_audio_denorm)
    
        # Calculate the Explained variance regression score (MSE/VAR)
        evr_audio = explained_variance_score(audio_features_target, audio_features_out_predict_from_audio_denorm,multioutput='variance_weighted')

        all_rmse_audio[f] = rmse_audio
        all_evr_audio[f] = evr_audio

        bar.next()

    bar.finish()

    return all_rmse_audio, all_evr_audio
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
        extract_mfcc_librivox_librosa(audio_root_dir, mfcc_dir)

    print('Start experiments using multispeaker training configuration')
    main_multispeaker()

