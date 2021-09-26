import {MnistData} from './data.js';

const COARSE_FINE_MAP_PATH = "./cifar-100-binary/course_to_fine_mapping.json"
const FINE_LABEL_PATH = './cifar-100-binary/fine_label_names.txt';
const COARSE_LABEL_PATH = './cifar-100-binary/coarse_label_names.txt';

const Tags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

// Define defaults for selectable parameters
let numEpochs = 1;
let maxWrong = 10;
let clearHistoryOutput = true;
let clearHistoryInput = true;
let layerCount = 2;

let kernelSize = 5;
let filtersMultiplier = 4;
let filterPowerBase = 2;
let filterStrides = 1;

let poolSize = 2;
let poolStride = 2;

let fineLabels;
let coarseLabels;

let coarseSelction;
let fineLabelOptions;

const IMAGE_WIDTH = 32;
const IMAGE_HEIGHT = 32;
const IMAGE_CHANNELS = 3;
const IMAGE_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS];

const IMAGE_RESHAPE = [3, 32, 32];
const RESHAPE_TRANSPOSE = [1, 2, 0]
const CONCAT_RESHAPE_TRANSPOSE = [0, 2, 3, 1]

const BATCH_SIZE = 48;
const EXAMPLE_COUNT = 20;

var global_model;

const TIMER_MS = 50; // ms interval for stopwatches
var t0; // Time at button press
var t1; // Time at start of run
var t2; // Time at end of train (before plot starts)

document.getElementById('start_button').onclick = async function() {
    const targets = setup();
    global_model = await run(targets, false);
};

document.getElementById('re_fit_button').onclick = function() {
    const targets = setup();
    run(targets, true);
};

function setup() {
    const fineSelection1 = document.getElementById('element_1').value;
    const fineSelection2 = document.getElementById('element_2').value;
    const fineIndex1 = fineLabels.indexOf(fineSelection1)
    const fineIndex2 = fineLabels.indexOf(fineSelection2)

    const targets = {
        "coarse" : coarseLabels.indexOf(coarseSelction),
        "fine": [
            {
                index : fineIndex1, 
                rawName: fineSelection1, 
                displayName : fineSelection1.replaceAll("_", " ")
            },
            {
                index : fineIndex2, 
                rawName: fineSelection2, 
                displayName : fineSelection2.replaceAll("_", " ")
            }
        ]
    };

    const checkNotDefaultInt = function(name, currentVal) {
        const input = parseInt(document.getElementById(name).value);
        const output = isNaN(input) ? currentVal : input;
        return output;
    }
    
    numEpochs = checkNotDefaultInt('epochs_input', numEpochs);
    layerCount = checkNotDefaultInt('layers_input', layerCount);
    maxWrong = checkNotDefaultInt('wrong_count_input', maxWrong);

    kernelSize = checkNotDefaultInt('kernel_size_input', kernelSize);
    filtersMultiplier = checkNotDefaultInt('filter_multiplier_input', filtersMultiplier);
    filterPowerBase = checkNotDefaultInt('filter_power_input', filterPowerBase);
    filterStrides = checkNotDefaultInt('filter_strides_input', filterStrides);
    
    poolSize = checkNotDefaultInt('pool_size_input', poolSize);
    poolStride = checkNotDefaultInt('pool_strides_input', poolStride);

    return targets;
}

document.addEventListener('DOMContentLoaded', boot);

async function getLabelListsFile() {
    
    function getLabelList(path) {
        let output = fetch(path)
        .then(response => response.text())
        .then(data => { 
            return data.split(/\r?\n/)
        });
        return output;
    }

    fineLabels = await getLabelList(FINE_LABEL_PATH);
    coarseLabels = await getLabelList(COARSE_LABEL_PATH);
}

async function boot() {
    await getLabelListsFile()

    setOptions('coarse_labels', coarseLabels);
    document.getElementById('element_1').disabled = true;
    document.getElementById('element_2').disabled = true;
    document.getElementById('coarse_labels').onchange = () => {

        document.getElementById('element_1').disabled = false;
        document.getElementById('element_2').disabled = true;
        document.getElementById('element_2').value = "";
        
        fetch(COARSE_FINE_MAP_PATH)
        .then(response => response.json())
        .then(map => {
            coarseSelction = document.getElementById('coarse_labels').value
            fineLabelOptions = map[coarseSelction];

            setOptions('element_1', fineLabelOptions);
        })
    }
}

function setOptions(target, items) {
    var str = '<option value="" selected disabled hidden>-select category-</option>';
    for (var item of items) {
        if (item.length > 0 && item !== null) {
            str += `<option value="${item}">${item.replaceAll("_", " ")}</option>`
        }
<<<<<<< HEAD
    }

    if (target === 'element_1' && document.getElementById(target).value === '') {
        document.getElementById('element_1').onchange = function() {
            document.getElementById('element_2').disabled = false;
            let filteredItems = fineLabelOptions.filter(e => {return e !== document.getElementById(target).value})
            setOptions('element_2', filteredItems)
        }
    }

=======
    }

    if (target === 'element_1' && document.getElementById(target).value === '') {
        document.getElementById('element_1').onchange = function() {
            document.getElementById('element_2').disabled = false;
            let filteredItems = fineLabelOptions.filter(e => {return e !== document.getElementById(target).value})
            setOptions('element_2', filteredItems)
        }
    }

>>>>>>> d85c4b5602b38c39c6a8f62c5e99cf933477cb89
    document.getElementById(target).innerHTML = str;
}

function startTimer(timer) {
    return setInterval(updateTimer, TIMER_MS, timer);
}

function stopTimer(loadTimer) {
    clearInterval(loadTimer)
}

function updateTimer(timer) {
    let time = performance.now();
    let title;
    if (timer == "timer_load") {
        title = "Time spent loading data: ";
        time -= t0;
    } else if (timer == "timer_train") {
        title = "Time spent training the model: ";
        time -= t1;
    } else if (timer == "timer_plot") {
        title = "Time spent generating the output visualizations: ";
        time -= t2;
    }
    document.getElementById(timer).innerHTML = `${title}${(time/1000).toFixed(3)} seconds`
}

async function run(targets, isFit) {
    t0 = performance.now();
    let loadTimer = startTimer("timer_load")

    const data = new MnistData();
    await data.load(targets);
    if (clearHistoryOutput) {tfvis.visor().surfaceList.clear();}
    await showExamples(data);

    const model = getModel(data.targets, isFit);
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model); // Shows how the model is laid out
    tfvis.visor().setActiveTab("Model");
    stopTimer(loadTimer) // Stops load timer

    
    t1 = performance.now();
    let trainTimer = startTimer("timer_train")
    await train(model, data);
    stopTimer(trainTimer) // Stops train timer

    t2 = performance.now();
    let plotTimer = startTimer("timer_plot")
    await showStats(model, data);
    stopTimer(plotTimer) // Stops plot timer
    

    return model;
}

async function showExamples(data) {
    // Create a container in the visor
    const surface =
        tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});
    tfvis.visor().setActiveTab("Input Data");
    if (clearHistoryInput) {surface.drawArea.textContent = '';}

    // Get the examples
    const examples = data.nextTestBatch(EXAMPLE_COUNT);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 32x32 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape(IMAGE_RESHAPE)
                .transpose(RESHAPE_TRANSPOSE);
        });  

        const canvas = document.createElement('canvas');
        let divy = document.getElementById('empty');
        canvas.width = IMAGE_WIDTH;
        canvas.height = IMAGE_HEIGHT;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);
        // divy.appendChild(canvas);

        imageTensor.dispose();
    }
}

function getModel(targets, isFit) {
    let model = tf.sequential();

    for (let x =1; x <= layerCount; x++) {
        model = makeConvPoolLayers(model, isFit, x);
    }

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({trainable: !isFit}));

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = targets.fine.length;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));


    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

function makeConvPoolLayers(model, isFit, convLayerNum) {
    // convLayerNum set is int for the number of the layer in the sequence

    // In the first layer of our convolutional neural network we have
    // to specify the input shape. Then we specify some parameters for
    // the convolution operation that takes place in this layer.

    // Expose for tweaking
    let config = {
        kernelSize: kernelSize,
        filters: filtersMultiplier * filterPowerBase**convLayerNum,
        strides: filterStrides,
        trainable: !isFit,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    };

    if (convLayerNum === 1) {
        config.inputShape = IMAGE_SHAPE;
    }
    if (isFit) {
        config.weights = global_model.layers[(convLayerNum-1)*2].getWeights();
    }

    model.add(tf.layers.conv2d(config));

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.
    model.add(tf.layers.maxPooling2d({poolSize: [poolSize, poolSize], strides: [poolStride, poolStride], trainable: !isFit}));

    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const TRAIN_DATA_SIZE = data.num_train_elements;
    const TEST_DATA_SIZE = data.num_test_elements;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE].concat(IMAGE_RESHAPE)).transpose(CONCAT_RESHAPE_TRANSPOSE),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE].concat(IMAGE_RESHAPE)).transpose(CONCAT_RESHAPE_TRANSPOSE),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: numEpochs, // Iterations over the whole dataset
        shuffle: true,
        callbacks: fitCallbacks
    });
}

async function constPredict(model, inputData) {
    // Gets a consistent set of sample data (for the given label choices) 
    //  Sample size will either be defined by constant CONST_SAMPLE_MAX_SIZE (at top of data.js) or the entirity of the test set if it is smaller than 1000
    const [testData, sampleSize] = inputData.getConstTestSample();
    const testxs = testData.xs.reshape([sampleSize].concat(IMAGE_RESHAPE)).transpose(CONCAT_RESHAPE_TRANSPOSE);
    const labels = testData.labels;

    // Clones the existing model, without the final dense layer
    const layerOutputModel = tf.model({inputs:model.inputs, outputs: model.layers[model.layers.length - 2].output});
    // Get predition from "cloned" model
    const outerPredsFloat32Array = await layerOutputModel.predict(testxs).data();

    // Split outerPredsFloat32Array into an arrary of the 
    const perElementCount = layerOutputModel.output.shape[1];
    var splitOutPuts = [];
    for (var i = 0; i < outerPredsFloat32Array.length; i += perElementCount) {
        splitOutPuts.push(Float64Array.from(outerPredsFloat32Array.subarray(i, i + perElementCount)));
    }

    // Convert labels from binaray array based on all classes, to binary flag for label A or B
    const rawLabels = await labels.data()
    const contextLabels = []
    for (var i=0; i<rawLabels.length; i+= 2) {
        contextLabels.push(rawLabels[i] == 0 ? 0 : 1)
    }

    return [contextLabels, splitOutPuts]
}

function plot(reducedArray, contextLabels, targets) {

    // Split array into X and Y components
    // (This is done as plotly requires x and y components seperately, but this makes processing for scaling easier)
    var [xArray, yArray] = [[], []]
    reducedArray.forEach(point => {xArray.push(point[0]); yArray.push(point[1])})

    // Get min and max for each axis to allow for min max scaling
    var [minX, maxX, minY, maxY] = [
        Math.min(...xArray),
        Math.max(...xArray),
        Math.min(...yArray),
        Math.max(...yArray),
    ]

    // Create a X and Y array for each label
    var [aXArray, bXArray, aYArray, bYArray] = [[],[],[],[]]

    // Defines a function to add the given index in x and y array to the provided arrays
    //  Points are scaled using min max scaling as they are added
    const addToArray = (xSubArray, ySubArray, index) => {
        xSubArray.push(xArray[index] - minX) / (maxX - minX);
        ySubArray.push(yArray[index] - minY) / (maxY - minY);
    }
    for (var index in reducedArray) {
        // If label for index is 0, call helper to add x and y at index to A arrays, else call helper to add x and y at index to B arrays
        contextLabels[index] == 0 ? addToArray(aXArray, aYArray, index) : addToArray(bXArray, bYArray, index)
    }

    // Helper function to form templtes based on arrays of x and y points
    const makeTrace = (xPart, yPart, name) => {
        return { // Template for plotting configs
            x: xPart,
            y: yPart,
            mode: 'markers',
            type: 'scatter',
            name: name
        }
    }
    
    // Make plotly configs for each label
    const traceA = makeTrace(aXArray, aYArray, targets.fine[0].displayName)
    const traceB = makeTrace(bXArray, bYArray, targets.fine[1].displayName)

    // Plot
    var plotlyBox = document.getElementById('plotlyBox');
    Plotly.newPlot(plotlyBox, [traceA, traceB]);
}

async function show2dPlot(model, inputData) {

    // Gets labels and an array of Float64Arrays containing the output of the prediction
    const [contextLabels, splitOutPuts] = await constPredict(model, inputData);

    // Create druid object for demension redution
    var druidObj = new druid.PCA(druid.Matrix.from(splitOutPuts), 2)
    // FASTMAP > MDS > ISOMAP > UMAP > PCA

    // Druid solution returns a druid matrix, so this must be converted into an array
    //  This is the computationally intense part
    var reducedArray = druidObj.transform().to2dArray;

    // Plot the redued array using the given labels for colouring
    plot(reducedArray, contextLabels, inputData.targets)
}

function doPrediction(model, data, testDataSize = BATCH_SIZE) {

    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize].concat(IMAGE_RESHAPE)).transpose(CONCAT_RESHAPE_TRANSPOSE);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels, testData];
}

async function showStats(model, data) {
    const [preds, labels, testData] = doPrediction(model, data);
    await show2dPlot(model, data);

    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const accContainer = {name: 'Accuracy', tab: 'Evaluation'};
    const targetNameList = [data.targets.fine[0].displayName, data.targets.fine[1].displayName]

    tfvis.show.perClassAccuracy(accContainer, classAccuracy, targetNameList);
    tfvis.visor().setActiveTab("Evaluation");

    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const confusionContainer = {name: 'Confusion Matrix', tab: 'Evaluation'};

    tfvis.render.confusionMatrix(confusionContainer, {values: confusionMatrix, tickLabels: targetNameList});

    const rawPreds = await preds.data();
    const rawLabels = await labels.data();

    // Get the examples
    const numExamples = testData.xs.shape[0];
    const surface01 =
        tfvis.visor().surface({ name: `Predicted ${targetNameList[0]} labeled ${targetNameList[1]}`, tab: 'Evaluation'});
    const surface10 =
        tfvis.visor().surface({ name: `Predicted ${targetNameList[1]} labeled ${targetNameList[0]}`, tab: 'Evaluation'});

    let wrong = [0,0];
    // Create a canvas element to render each example
    // for (let index in rawPreds) {
    for (let index = 0; index < rawPreds.length; index++) {
        if (rawPreds[index] !== rawLabels[index] && wrong[rawPreds[index]]++ < maxWrong) {

            const imageTensor = tf.tidy(() => {
                // Reshape the image to 28x28 px
                return testData.xs
                    .slice([index, 0], [1, testData.xs.shape[1]])
                    .reshape(IMAGE_RESHAPE)
                    .transpose(RESHAPE_TRANSPOSE);
            });

            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;
            canvas.style = 'margin: 4px;';
            await tf.browser.toPixels(imageTensor, canvas);
            if (rawPreds[index] === 1) {
                surface10.drawArea.appendChild(canvas);
            } else {
                surface01.drawArea.appendChild(canvas);
            }

            imageTensor.dispose();
        }
        if (wrong[0] >= maxWrong && wrong[1] >= maxWrong) {
            break;
        }
    }

    labels.dispose();
}
