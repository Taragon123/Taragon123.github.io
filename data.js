/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * A class that fetches the CIFAR dataset and returns shuffled batches.
 */
export class CifarData {
    constructor(masterConfig) {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;

        this.masterConfig = masterConfig; // This will store the config json for all parts of the code to access it instead of having to load it multiple times

        const dataSetSpec = masterConfig.defaults.dataSetSpec;
        this.imageSize = dataSetSpec.imageWidth * dataSetSpec.imageHeight * dataSetSpec.imageChannels;

        this.testSetMaxSize = masterConfig.defaults.dataSetSpec.testSize;
        this.trainSetMaxSize = masterConfig.defaults.dataSetSpec.trainSize;

        this.defaultBatchSize = masterConfig.defaults.data.defaultBatchSize;
        this.constSampleMaxSize = masterConfig.defaults.data.constSampleMaxSize;

        this.testBinPathPrefix = masterConfig.defaults.data.testBinPathPrefix;
        this.trainBinPathPrefix = masterConfig.defaults.data.trainBinPathPrefix;
    }

    // Loads the data class
    // Sets some of the given values then fetches the data and labels
    async load(targets, trainScale) {
        this.targets = targets;
        this.trainingSetSize = this.trainSetMaxSize * (trainScale/100) * 2;
        this.testingSetSize = this.testSetMaxSize;
        this.batchSize = this.trainingSetSize < this.defaultBatchSize ? this.trainingSetSize : this.defaultBatchSize;

        // Fetch data and labels
        [this.testLabels, this.testImages] = await this.getLabelsAndData(true);
        [this.trainLabels, this.trainImages] = await this.getLabelsAndData(false, trainScale);
        
        // x2 to account for the two labels, will need updating if using more than 2 labels
        this.num_train_elements = this.trainSetMaxSize * 2;
        this.num_test_elements = this.testSetMaxSize * 2;
        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(this.num_train_elements);
        this.testIndices = tf.util.createShuffledIndices(this.num_test_elements);
    }

    // pathPrefix is path to dataset
    // isTest is boolean flag for if it is the test dataset, if it is not, it is considered the train set
    async getLabelsAndData(isTestSet, trainScale=null) {
        // If it's the test set use full test set size, else, use training set size scaled by percentage of traing set to be used 
        const imageCount = isTestSet ? this.testSetMaxSize : this.trainSetMaxSize * (trainScale/100);

        // Holds all data as it is read in
        let dataUint8Array; 
        const pathPrefix = isTestSet ? this.masterConfig.defaults.data.testBinPathPrefix : this.masterConfig.defaults.data.trainBinPathPrefix
        for (let targetIndex in this.targets.fine) {
            // Forms the file path to the target's data, ie "<pathPrefix>/apple-0-train.bin"
            let path = `${pathPrefix}${this.targets.fine[targetIndex].rawName}-${this.targets.fine[targetIndex].index}-${isTestSet ? 'test' : 'train'}.bin`;

            // Get the image data for this target
            let imagesUint8 = await this.getTargetData(path, imageCount)

            // If it's the first target, set the output array to the new data, else merge the new data into the output array
            if (targetIndex == 0) {
                dataUint8Array = imagesUint8;
            } else {
                dataUint8Array = this.mergeUint8Arrays(dataUint8Array, imagesUint8)
            }
        }

        // Convert the output data to be a Float32Array
        let images265 = Float32Array.from(dataUint8Array);
        // Scale all values by 1/256 (pixel values initially range from 0-255, this brings them into 0-1)
        let images = images265.map((i) => {return i/256});

        // **** If scaling to more than 2 labels, the next section will need to be updated ****
        // Create template labels
        const one = [1, 0];
        const two = [0, 1];

        // Labels output
        let labels = [];

        // Add all labels for traget 1
        for (let i=0; i < imageCount; i++) {
            labels.push(...one)
        }
        // Add all labels for target 2
        for (let j=0; j < imageCount; j++) {
            labels.push(...two)
        }

        return [labels, images]
    }

    // This is the main data loading function
    async getTargetData(path, imageCount) {
        // Get file from path (file is the data for a given single label)
        return await fetch(path)
        .then(response => {
            const reader = response.body.getReader();
            
            // This stuff is confusing, best to reference mozila guide:
            // https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream
            return new ReadableStream({
                start(controller) {
                    return pump();
                    function pump() {
                        return reader.read().then(({ done, value }) => {
                        // When no more data needs to be consumed, close the stream
                        if (done) {
                            controller.close();
                            return;
                        }
                        // Enqueue the next data chunk into our target stream
                        controller.enqueue(value);
                        return pump();
                        });
                    }
                }
            })
        })
        .then(stream => new Response(stream))   // Make response out of the stream
        .then(response => response.blob())      // Turn tthe response into a blob
        .then(blob => blob.arrayBuffer())       // Make an arrayBuffer out of the blob
        .then(
            inputBuffer => {

                // Make arrayBuffer to hold output data and be used by the DataView
                const datasetBytesBuffer = new ArrayBuffer(this.imageSize * imageCount)

                // Create DataView for input and output buffers so they are easier to work with
                let inputView = new DataView(inputBuffer);
                let outputViewData = new DataView(datasetBytesBuffer);
                
                // Chunk size is how big each image is in bytes so one image can be processed at a time
                const chunkSize = this.imageSize;
                // Count is only relevant when the training set is limited
                let count = 0;

                // For each image in the dataset
                for (let i = 0; i < inputBuffer.byteLength; i+=chunkSize) {
                    // For each byte in the image
                    for (let j=0; j < this.imageSize; j+=1) {
                        outputViewData.setUint8(count*this.imageSize+j, inputView.getUint8(i+j));
                    }

                    count++;
                    if (count >= imageCount) {  // This is used to limit the training set size
                        break;
                    }
                }
                
                // The data is output as a Unit8Array
                let imagesUint8 = new Uint8Array(datasetBytesBuffer);

                return imagesUint8
            })
        .catch(err => console.error(err));
    }

    // Helper function to merge two Uint8Arrays
    mergeUint8Arrays(arrayA, arrayB) {
        var mergedArray = new Uint8Array(arrayA.length + arrayB.length);
        mergedArray.set(arrayA);
        mergedArray.set(arrayB, arrayA.length);

        return mergedArray
    }

    // Fetches a batch of data of the given batch size
        // index is a function for getting the indexes from the shuffled indexes array
    nextBatch(batchSize, data, index) {
        const targetCount = this.targets.fine.length
        const batchImagesArray = new Float32Array(batchSize * this.imageSize);
        const batchLabelsArray = new Uint8Array(batchSize * targetCount);

        for (let i = 0; i < batchSize; i++) {
            let idx = index();

            let label = data[1].slice(idx * targetCount, idx * targetCount + targetCount);
            batchLabelsArray.set(label, i * targetCount);

            const image = data[0].slice(idx * this.imageSize, idx * this.imageSize + this.imageSize);
            batchImagesArray.set(image, i * this.imageSize);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, this.imageSize]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, targetCount]);
        return {xs, labels};
    }

    // Fetches the next batch of given batch size for training
    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;

                return this.trainIndices[this.shuffledTrainIndex];
            });
    }

    // Fetches the next batch of given batch size for testing
    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }

    // For use in generating the 2d plot
    // This fetches a consistent sample from the test set
    getConstTestSample() {
        const sampleSize = this.num_test_elements < this.constSampleMaxSize ? this.num_test_elements : this.constSampleMaxSize;
        this.constSampleIndex = 0;

        return [ 
            this.nextBatch(sampleSize, [this.testImages, this.testLabels],  () => {
                this.constSampleIndex =
                    (this.constSampleIndex + 1) % this.testIndices.length;
                return this.testIndices[this.constSampleIndex];
            }),
            sampleSize
            ];
    }
}