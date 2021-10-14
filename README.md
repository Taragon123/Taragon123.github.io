The code is based off of the google code lab: TensorFlow.js — Handwritten digit recognition with CNNs and has been heavily modified to fit the needs of the project.

The finished project is intended to be integrated into the [CS Field Guide](https://www.csfieldguide.org.nz/) to provide pre-tirtiary teachers an additional resource to assit in teaching about AI.

***Config.json***
This file contains most of the configuration one might want to change for the application, but any major changes will likely require rework of relevant code

*groupKeys*:
Each entry in this represents a toggelable set of inputs for the neural network's parameters. This is in the format of "<gropName> : [<list of html input ids>]"

*options*:
This is split into 3 sections;

preset:
This contains all presets for the system that users can select from.
Each preset contains:
The list of groupKeys to hide
The non-default configuration paramaters, defined using the input id
The name to display the configuration as to users
When making presets, they must be named by as "preset<preset number>* where number must be the next posative integer. If you wish to change the naming scheme, the "checkUrlParams" function must be updated in script.js (the specific line is near the bottom of the function. Look for teh regex test :) )

default:
This contains the list of groupKeys to be hidden by default, the default values are defined later

advanced:
This is the list of terms that can be used as a query paramater value for the advanced key. Each value included will show that input set (labels are not shown by default, so always remember to include them!). Additionally, the value of "all" can be used to show all inputs (excluding the presets selector).
    example: to see label selectors, filter configuration and epoch count, use the url "<baseURL>?advanced=labels,filters,epochs"

*defaults*:
This section contains all default values for the application, split into the file it is used in or the section dedicated to the dataset values

script:
Contains all default values used in script.js

language    : Sets the language profile to be used for user facing strings (see strings section of config) the value must match a key in the strings config section

courseFineMapPath   :   The relative path to the course_to_fine_mapping.json file from the location of the script 
fineLabelsPath      :   The relative path to the fine_label_names.txt file from the location of the script 
coarseLabelsPath    :   The relative path to the coarse_label_names.txt file from the location of the script 

numEpochs               :   The default number of epochs the training will run
maxWrong"               :   The default maximum number of examples the system will give of wrong predictions (for each label)
layerCount"             :   The default number of layers used in the convolutional network

kernelSize"             :   The default size of the kernel used in the convolutional netowrk
*Note: Filter fourmual  :   multiplier * (base ^ layer number)*
filtersMultiplier"      :   The default multiplier for the filters in the convolutional netowrk
filterPowerBase"        :   The default power base for the filters in the convolutional netowrk
filterStrides"          :   The default strides for filters in the convolutional netowrk

*Note: Pool settings reflect a square or x size ([x, x] config)*
poolSize"               :   The default size of the pools in the convolutional netowrk
*See section on demsionality reduction below for important note*
poolStride"             :   The default stride of the pools in the convolutional netowrk

trainScale"             :   The default percentage of the training set that will be used for training the model

exampleCount"           :   The default number of examples shown to the user before the training starts
clearHistoryOutput"     :   A boolean flag indicating if the history of wrong examples should be cleared
clearHistoryInput"      :    A boolean flag indicating if the history input examples should be cleared

*Note: As a demension reduction technique is used on the output matrix of the predictions flattern layer, when the number of parameters in this grow too large, the demension reduction process will take exponentially longer. This will likely occur if the pool strides is set to 1* 
demensionReduction:
    sizeLimit           :   The maximum output size of flatten layer for demension reduction to occur
    ignoreLimit"        :   Boolean flag to disable sizeLimit
}


data:
Contains all default values used in data.js

defaultBatchSize        :   The dafault batch size during training, however if the training set is smaller than this, the training set size will become the batch size 
constSampleMaxSize      :   To allow better comparison usinging the 2d plot, a consistent sample is gathered to prevent any rendomness in the samples from being misinterpreted. This value sets the max size of that sample. The sample will match the test set's size if it is smaller than this value


dataSetSpec:
This is the technical configuration of the dataset and is CRITICAL to be accurate. If this information is changed, it means the dataset has changed and this will very likely require reworking large portions of code.

imageWidth              :   The width of each image in the dataset
imageHeight             :   The height of each image in the dataset
imageChannels           :   The number of channels used for eeach image in the dataset

testSize                :   The number of test images for each category in the dataset
trainSize               :   The number of training images for each category in the dataset

strings:
This contains the list of supported languages and the reelative path to their strings.json file. Each entry should have a key indicating the language and the path to the relevant strings.json. To set the active language, this is currently done through the language key in the script portion under defaults in config.json

en  : The path to the english strings

*Note: Adding more languages is easy, but it must be added here as done for English