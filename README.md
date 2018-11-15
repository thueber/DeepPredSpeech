# DeepPredSpeech
Computational models of predictive speech coding based on deep learning

This repository contains the source code used to design the computational models of predictive speech coding described in Hueber. T., Tatulli, E., Girin, L., Schwartz, J-L, "How predictive can be predictions in the neurocognitive processing of auditory and audiovisual speech? A deep learning study.", and to train/evaluate them on the NTCD-TIMIT multimodal speech database. All training/test data (including audio features and raw lip images), pre-trained models, and experimental results have been made publicly available on Zenodo (DOI 10.5281/zenodo.1487974). 

This repository contains two main scripts (Python 2.7): "do_sim_pred_coding_ntcdtimit_cnn.py" and "do_sim_pred_coding_ntcdtimit_cnn_audio.py" which are respectively related to the processing of MFCC-spectrogram and log-magnitude spectrogram (both eventually combined with visual input, i.e. lip movements). See our paper for more details about these audio representations and the corresponding model architectures (e.g. feed-forwared deep neural network or convolutional neural network or a combination of both). 

Many options related to the feature extraction, the architecture of audio, visual and audio-visual predictive models, their training, etc. can be modified by editing directly the "CONFIG" section at the beginning of each script (see comments for a brief explanation of each options). 

Don't hesitate to contact me for more details. 

Thomas Hueber, Ph. D. 
CNRS researcher, GIPSA-lab, Grenoble, France
thomas.hueber@gipsa-lab.fr
