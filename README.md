## Overview
This project contains 3 files Train.py, Predict.py, and Utils.py. Train.py and Predict.py can be run by following the
Instruction below. Utils.py is only there to bundle code that is needed in more than one place and running it will do nothing.

This project requires no additional packages besides pytorch.

In the models directory there are already trained models that can be used by Predict.py.

## Introductions on how to use Train.py and Predict.py

### Train.py:
Train.py takes 2 arguments:
 - --source : path to the directory where the data for training is located. Make sure that the training data has the same structure
 as the data in the dataset directory. Also note that all JSON files from the provided directory will be read so make sure that
 only the files meant for training are in the specified directory.
 - --destination : path to the directory where the models will be stored. Train.py will produce 3 models: location_model.pt, 
 type_model.pt, and token_model.pt. All 3 models will be stored in the specified directory. Too make it 100% explicit: since
 Train.py produces 3 models providing destination does NOT specify the file name of those models, but only the directory where the
 models will be stored. Also note that the provided directory needs to exist already.

 > Example Usage: python Train.py --source ../dataset --destination ./models

 When running Train.py all models are trained successively. The current state of the training is printed to the console the first
 number indicates the latest epoch and the second number indicates the the cummulative loss for this epoch. The total number of
 epochs is 20 for each model. The training takes some time! When training on GPU the training takes roughly over 1h per model.
 Also make sure that the GPU is not busy doing other things as this will slow down training tremendously!


### Predict.py:

 Predict.py: takes 3 arguments:
 - --model : path to the directory where the models are located. make sure that the models have the same name as produced by
 Train.py.
 - --source : path to the directory where the data for prediction is located. The restrictions are the same as the ones for
 Train.py, except the fields that are meant to be predicted, obviously those are not required to exist (e.g. the fix_location or 
 the correct_code fields are not mandatory).
 - --destination : path to which the output JSON file should be written. Note that this time the output will be a single file and
 the filename + extension should also be part of the path (see example below)

 > Example Usage: python Predict.py --model ./models --source ../eval --destination ./output.json

  the time it takes to run Predict.py depends on the number of samples provided as input in the directory specified by --source.

  ## Data Format
  The format of the data the tool expects can be seen in [report.pdf](https://github.com/lars447/ASDL-Project/blob/master/report.pdf)