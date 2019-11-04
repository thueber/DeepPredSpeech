# DeepPredSpeech
Computational models of predictive speech coding based on deep learning

This repository contains the source code used to design the computational models of predictive speech coding described in Hueber. T., Tatulli, E., Girin, L., Schwartz, J-L, "How predictive can be predictions in the neurocognitive processing of auditory and audiovisual speech? A deep learning study.". The initial submission is available as a preprint at https://doi.org/10.1101/471581 (the paper is currently under revision in a peer-reviewed journal). 

This repository contains the following scripts (Python 2.7): 
* "do_sim_pred_coding_ntcdtimit_cnn.py" and "do_sim_pred_coding_ntcdtimit_cnn_audio.py" which are related to the experiments conducted on the NTCD-TIMIT audiovisual speech database. The first one is related to the experiments based on MFCC-spectrogram, the second one on log-magnitude spectrogram (both eventually combined with visual input, i.e. lip movements). See the preprint for more details about these audio representations and the corresponding model architectures (e.g. feed-forwared deep neural network or convolutional neural network or a combination of both). All training/test data (including audio features and raw lip images), pre-trained models, and experimental results have been made publicly available on Zenodo (DOI 10.5281/zenodo.1487974). 
* do_sim_pred_coding_librivox.py  which are related to the experiments conducted on the Librispeech (audio-only database), based on the MFCC spectrogram (not presented in the preprint).

Many options related to the feature extraction, the architecture of audio, visual and audio-visual predictive models, their training, etc. can be modified by editing directly the "CONFIG" section at the beginning of each script (see comments for a brief explanation of each options). 

This repository contains also a jupyter notebook DeepPredSpeechXplore allowing to simulate predictive coding of speech with our models pre-trained on Librispeech from any audio file. Pretrained models and audio sound examples are available in the DeepPredSpeechXplore_res.zip archive file (which has to me unzip and placed in the same directory as the notebook).  

Don't hesitate to contact me for more details. 

Thomas Hueber, Ph. D. 
CNRS researcher, GIPSA-lab, Grenoble, France
thomas.hueber@gipsa-lab.fr
