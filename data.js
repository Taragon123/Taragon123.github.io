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

 const IMAGE_WIDTH = 32;
 const IMAGE_HEIGHT = 32;
 const IMAGE_CHANNELS = 3;
 const IMAGE_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS];

const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;
const TEST_SIZE = 100;
const TRAIN_SIZE = 500;

const CONST_SAMPLE_MAX_SIZE = 200;

const TEST_BIN_PATH_PREFIX = './cifar-100-binary/split test bins/';
const TRAIN_BIN_PATH_PREFIX = './cifar-100-binary/split train bins/';


async function getTargetData(path, imageCount) {
    return await fetch(path)
    .then(response => {
        const reader = response.body.getReader();
        
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
    .then(stream => new Response(stream))
    .then(response => response.blob())
    .then(blob => blob.arrayBuffer())
    .then(
        inputBuffer => {
            let byteArray = new Uint8Array(inputBuffer)

            const datasetBytesBuffer = new ArrayBuffer(IMAGE_SIZE * imageCount)

            let inputView = new DataView(inputBuffer);
            let outputViewData = new DataView(datasetBytesBuffer);
              
            const chunkSize = IMAGE_SIZE;
            let count = 0;
            for (let i = 0; i < byteArray.length; i+=chunkSize) {
                for (let j=0; j < IMAGE_SIZE; j+=1) {
                    outputViewData.setUint8(count*IMAGE_SIZE+j, inputView.getUint8(i+j));
                }
                count++;
                if (count >= imageCount) {
                    break;
                }
            }
            
            let imagesUint8 = new Uint8Array(datasetBytesBuffer);

            return imagesUint8
        })
    .catch(err => console.error(err));
}

function mergeUint8Arrays(arrayA, arrayB) {
    var mergedArray = new Uint8Array(arrayA.length + arrayB.length);
    mergedArray.set(arrayA);
    mergedArray.set(arrayB, arrayA.length);

    return mergedArray
}

// Path is path to dataset
// isTest is boolean flag for if it is the test dataset, if it is not, it is considered the train set
async function getLabelsAndData(pathPrefix, targets, isTestSet, trainScale=null) {
    const imageCount = isTestSet ? TEST_SIZE : TRAIN_SIZE * (trainScale/100);

    let dataUint8Array; 
    for (let targetIndex in targets.fine) {
        let path = `${pathPrefix}${targets.fine[targetIndex].rawName}-${targets.fine[targetIndex].index}-${isTestSet ? 'test' : 'train'}.bin`;

        let imagesUint8 = await getTargetData(path, imageCount)

        if (targetIndex == 0) {
            dataUint8Array = imagesUint8;
        } else {
            dataUint8Array = mergeUint8Arrays(dataUint8Array, imagesUint8)
        }
    }

    let images265 = Float32Array.from(dataUint8Array);
    let images = images265.map((i) => {return i/256});

    const one = [1, 0];
    const two = [0, 1];
    let labels = [];

    for (let i=0; i < imageCount; i++) {
        labels.push(...one)
    }
    for (let j=0; j < imageCount; j++) {
        labels.push(...two)
    }

    return [labels, images]
}

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }

    async load(targets, trainScale) {
        this.targets = targets;
        this.trainingSetSize = TRAIN_SIZE * (trainScale/100);

        [this.testLabels, this.testImages] = await getLabelsAndData(TEST_BIN_PATH_PREFIX, targets, true);
        [this.trainLabels, this.trainImages] = await getLabelsAndData(TRAIN_BIN_PATH_PREFIX, targets, false, trainScale);
        
        this.num_train_elements = TRAIN_SIZE * 2;
        this.num_test_elements = TEST_SIZE * 2;
        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(this.num_train_elements);
        this.testIndices = tf.util.createShuffledIndices(this.num_test_elements);
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;

                return this.trainIndices[this.shuffledTrainIndex];
            });
    }

    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }

    nextBatch(batchSize, data, index) {
        const targetCount = this.targets.fine.length
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * targetCount);

        for (let i = 0; i < batchSize; i++) {
            let idx = index();

            let label = data[1].slice(idx * targetCount, idx * targetCount + targetCount);
            batchLabelsArray.set(label, i * targetCount);

            const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, targetCount]);
        return {xs, labels};
    }

    getConstTestSample() {
        const sampleSize = this.num_test_elements < CONST_SAMPLE_MAX_SIZE ? this.num_test_elements : CONST_SAMPLE_MAX_SIZE;
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
