import {CifarData} from './data.js';

let CONFIG_PATH = './config.json';

export class ScriptObject {
    constructor() {
        this.masterConfig; // This will store the config json for all parts of the code to access it instead of having to load it multiple times

        this.language; // Sets the language for the page based on settings in config.json
        this.strings; // Cotains all strings for the UI

        // These file paths are defined in config.json
        this.courseFineMapPath;
        this.fineLabelsPath;
        this.coarseLabelsPath;

        // Instanciate global variables to hold either default or provided parameters
        // These are set using the values in config.json
        this.numEpochs;
        this.maxWrong;
        this.layerCount;

        this.kernelSize;
        this.filtersMultiplier;
        this.filterPowerBase;
        this.filterStrides;

        this.poolSize;
        this.poolStride;

        this.trainScale;

        this.exampleCount;
        this.clearHistoryOutput;
        this.clearHistoryInput;

        this.fineLabels;
        this.coarseLabels;

        this.coarseSelction;
        this.fineLabelOptions;

        this.imageWidth;
        this.imageHeight;
        this.imageChannels;
        this.imageShape;

        // These are set programatically
        this.IMAGE_RESHAPE;
        this.RESHAPE_TRANSPOSE = [1, 2, 0];
        this.CONCAT_RESHAPE_TRANSPOSE = [0, 2, 3, 1];

        // This is set after the run function finishes and holds the generated model
        this.global_model;

        this.TIMER_MS = 50; // ms interval for stopwatches
        this.t0; // Time at button press
        this.t1; // Time at start of run
        this.t2; // Time at end of train (before plot starts)
    }

    // ######################################### Startup procedures ###############################################

    // This is called when the html is loaded. This should be the first code executed in this file
    async boot() {
        this.masterConfig = await this.getJson(CONFIG_PATH);

        const defaults = this.masterConfig.defaults;
        
        // Sets the language based on the config file. Then calls setStrings to put all text in the UI
        this.language = defaults.script.language;
        this.setStrings();

        // Helper function to set
        function setPlaceholderVal(name, defaultVal) {
            document.getElementById(name).placeholder = defaultVal;
        }
        
        // Set all placeholder values to the default values as defined in config.json
        for (var input in defaults.script.inputToVarMap) {
            setPlaceholderVal(input, defaults.script[defaults.script.inputToVarMap[input]]);
        }

        // Set defauult values from config.json
        this.courseFineMapPath = defaults.script.courseFineMapPath;
        this.fineLabelsPath = defaults.script.fineLabelsPath;
        this.coarseLabelsPath = defaults.script.coarseLabelsPath;

        this.imageWidth = defaults.dataSetSpec.imageWidth;
        this.imageHeight = defaults.dataSetSpec.imageHeight;
        this.imageChannels = defaults.dataSetSpec.imageChannels;

        this.imageShape = [this.imageWidth, this.imageHeight, this.imageChannels];

        // This may encounter issues if the input images are not square or a different source file is used
            // Height and width may need to be swapped in such a case
        this.IMAGE_RESHAPE = [this.imageChannels, this.imageWidth, this.imageHeight];

        this.checkUrlParams();

        this.setLabelOptions();
    }

    // Fetches a json file from the given path
    async getJson(path) {
        let output = fetch(path)
        .then(response => response.json())
        .then(map => {return map})

        return output;
    }

    // Sets all strings in the UI from the given strings.json file (path defined in config)
    async setStrings() {
        this.strings = await this.getJson(this.masterConfig.strings[this.language]);
        
        for (let id of Object.keys(this.strings.directLabels)) {
            document.getElementById(id).innerText = this.strings.directLabels[id];
        }

        for (let id of Object.keys(this.strings.htmlEnabledLabels)) {
            document.getElementById(id).innerHTML = this.strings.htmlEnabledLabels[id];
        }

        document.getElementById('nullSelectOne').innerText = this.strings.otherLabels.nullElementSlectionText;
        document.getElementById('nullSelectTwo').innerText = this.strings.otherLabels.nullElementSlectionText;
    }

    // Checks the URL paramaters and sets up the preset options (network configurations) for the user to select from
    checkUrlParams() {
        const urlSearchParams = new URLSearchParams(window.location.search);
        const params = Object.fromEntries(urlSearchParams.entries());

        // Helper function to setup the selector for the user
        function setPresetOptionList(presetsConfig) {
            var str = "";
            for (var key of Object.keys(presetsConfig)) {
                str += `<option value="${key}">${presetsConfig[key].displayName}</option>`;
            }
        
            document.getElementById('preset_select').innerHTML = str;

            // When the preset is changed, dissable the re-fit option as a change in preset will require a different model
            document.getElementById('preset_select').onchange = function() {
                document.getElementById('reFitButton').disabled = true;
            }
        }
        
        // If the user has use the advanced key in the URL paramaters, this filters that and creates their specific configuration
        if ("advanced" in params) {
            let hideList = this.masterConfig.options.advanced
            if (params["advanced"] !== "all") {
                for (let i of params["advanced"].split(',')) {
                    if (i === "presets") {
                        setPresetOptionList(this.masterConfig.options.preset);
                    }
                    
                    let index = hideList.indexOf(i);
                    if (index > -1) {
                        hideList.splice(index, 1);
                    }
                }
            } else {
                hideList = ["presets"]
            }
            
            this.loadPreset({"hide": hideList});

        // if the user is using URL assigned presets, this hadels that
        } else if ("preset" in params && 
                    /^([1-9]\d*)$/.test(params.preset) && 
                    parseInt(params.preset) <= Object.keys(this.masterConfig.options.preset).length) {
            this.loadPreset(this.masterConfig.options.preset[`preset${params.preset}`])

        // If no URL params that we care about are found, this laods the default preset
        } else {
            this.loadPreset(this.masterConfig.options.default);
            
            setPresetOptionList(this.masterConfig.options.preset);
        }
    }

    // Loads a given preset
    // Hides the inputs that are specified to be hidden
    // Sets any values for the preset. This replaces the default values when using given preset
    loadPreset(config) {
        if ("hide" in config) {
            for (let i of config.hide) {
                for (let x of this.masterConfig.groupKeys[i]) {
                    document.getElementById(x).style.display = "none";
                }
            }
        }
        
        if ("config" in config) {
            this.setPresetValues(config.config)
        }
    }

    // Sets values from given config preset
    setPresetValues(config) {
        for (var key of Object.keys(config)) {
            document.getElementById(key).value = config[key]
        }
    }

    // Fetches the lists of the fine and coarse labels
    async getLabelListsFile() {
        
        // Helper function that fetches the given text file and returns a list of the contents split by new lines
        function getLabelList(path) {
            let output = fetch(path)
            .then(response => response.text())
            .then(data => { 
                return data.split(/\r?\n/)
            });
            return output;
        }

        this.fineLabels = await getLabelList(this.fineLabelsPath);
        this.coarseLabels = await getLabelList(this.coarseLabelsPath);
    }

    // Helper function to handel ssetting the contents of a select input
    // This function sets the input options then also conditionally enabled and dissables buttons and inputs based on what select is being setup 
    setOptions(target, items) {
        var str = `<option value="" selected disabled hidden>${this.strings.otherLabels.nullElementSlectionText}</option>`;
        for (var item of items) {
            if (item.length > 0 && item !== null) {
                str += `<option value="${item}">${item.replaceAll("_", " ")}</option>`
            }
        }

        document.getElementById(target).innerHTML = str;
    }

    // Sets the options for the user to select catergories and labels
    async setLabelOptions() {
        await this.getLabelListsFile();

        this.setOptions('coarse_labels', this.coarseLabels);

        // Dissabled the label selection until the corase label iss selected 
        document.getElementById('element_1').disabled = true;
        document.getElementById('element_2').disabled = true;

        // When the coarse lable is selected, enable selecting label 1 and dissable label 2 selection. Also dissables the start and re-fit buttons to prevent starting without labels
        document.getElementById('coarse_labels').onchange = () => {

            document.getElementById('element_1').disabled = false;
            document.getElementById('element_2').disabled = true;
            document.getElementById('element_2').value = "";
            document.getElementById('reFitButton').disabled = true;
            document.getElementById('startButton').disabled = true;
            
            // Uses the coarse to fine label map to get the fine labels relevant to the selected coarse label
            fetch(this.courseFineMapPath)
            .then(response => response.json())
            .then(map => {
                this.coarseSelction = document.getElementById('coarse_labels').value
                this.fineLabelOptions = map[this.coarseSelction];

                this.setOptions('element_1', this.fineLabelOptions);
            })
        }
    }

    // #################################### Once the start or re-fit button is pressed ################################################

    // This is the main logic handeler for running the training or re-fits
    // Called when either the start or re-fit buttons are clicked
    async run(targets, isFit) {
        
        // Dissables the main inputs when training starts
        this.dissableMainInputs(true);

        // Starts the timer for loading the data
        this.t0 = performance.now();
        let loadTimer = this.startTimer("timer_load")

        // Load the data
        const data = new CifarData(this.masterConfig);
        await data.load(targets, this.trainScale);
        // Clears the example history
        if (this.clearHistoryOutput) {tfvis.visor().surfaceList.clear();}
        // Tell the user how many items are being used for training
        document.getElementById("traingSizeLabel").innerHTML = this.strings.otherLabels.trainScaleDisplay.replace(this.strings.otherLabels.trainScaleReplaceKey, data.trainingSetSize);
        
        await this.showExamples(data);

        // Creates the AI model
        const model = this.getModel(data.targets, isFit);
        // Shows how the model is setup
        tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model); 
        tfvis.visor().setActiveTab("Model");
        // Stops timer for loading the data
        this.stopTimer(loadTimer) 

        // Starts the timer for training the model
        this.t1 = performance.now();
        let trainTimer = this.startTimer("timer_train")
        // Calls the core training function
        await this.train(model, data);
        // Stops the timer for training the model
        this.stopTimer(trainTimer) 

        // STarts the timer for generating the 2d visualizations plot
        this.t2 = performance.now();
        let plotTimer = this.startTimer("timer_plot")
        await this.showStats(model, data);
        this.stopTimer(plotTimer) // Stops plot timer
        
        // Enables the main inputs once training ends
        this.dissableMainInputs(false);

        return model;
    }

    // Helper function to dissable and enable the main inputs
    // Currently ignores any additional inputs shown via advanced config options
    dissableMainInputs(doDissable) {
        document.getElementById('element_1').disabled = doDissable;
        document.getElementById('element_2').disabled = doDissable;
        document.getElementById('coarse_labels').disabled = doDissable;
        document.getElementById('preset_select').disabled = doDissable;

        document.getElementById('startButton').disabled = doDissable;
        document.getElementById('reFitButton').disabled = doDissable;
    }

    // Called at the start of the run
    setup() {
        // Get seelected labels
        const fineSelection1 = document.getElementById('element_1').value;
        const fineSelection2 = document.getElementById('element_2').value;
        const fineIndex1 = this.fineLabels.indexOf(fineSelection1);
        const fineIndex2 = this.fineLabels.indexOf(fineSelection2);

        // Setup targets object
        const targets = {
            "coarse" : this.coarseLabels.indexOf(this.coarseSelction),
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

        // Checks all inputs
        this.checkInputs();

        return targets;
    }

    // checks all inputs and replaces the default value if an input variable is found
    checkInputs() {
        // Check if preset is being used
        let presetKey = document.getElementById("preset_select").value;
        if (presetKey !== "") { // If preset is being used
            // Set values from preset
            this.setPresetValues(this.masterConfig.options.preset[presetKey].config);
        }

        const defaults = this.masterConfig.defaults.script

        // Helper function to reduce duplication
        // Grabs the input value, then returns the input value if there is one or the default if there is not
        function checkNotDefaultInt(name, currentVal) {
            const input = parseInt(document.getElementById(name).value);
            const output = isNaN(input) ? currentVal : input;
            return output;
        }
        
        // Set all default values as defined in config.json
        for (var input in defaults.inputToVarMap) {
            this[defaults.inputToVarMap[input]] = checkNotDefaultInt(input, defaults[defaults.inputToVarMap[input]]);
        }

        // These options are only set from the config file
        this.exampleCount = defaults.exampleCount;
        this.clearHistoryOutput = defaults.clearHistoryOutput;
        this.clearHistoryInput = defaults.clearHistoryInput;
    }

    // Util function to start timers
    startTimer(timer) {
        return setInterval(this.updateTimer, this.TIMER_MS, timer);
    }

    // Util function to stop timers
    stopTimer(timer) {
        clearInterval(timer)
    }

    // Called every x milliseconds as defined by TIMER_MS
    // Updates the display of the currently running timer
    updateTimer(timer) {
        const otherLabels = scriptObject.strings.otherLabels;
        let time = performance.now();
        let title;
        if (timer == "timer_load") {
            title = otherLabels.loadTimerText;
            time -= scriptObject.t0;
        } else if (timer == "timer_train") {
            title = otherLabels.trainTimerText;
            time -= scriptObject.t1;
        } else if (timer == "timer_plot") {
            title = otherLabels.plotTimerText;
            time -= scriptObject.t2;
        }
        // Remember to devide by 1000 to get seconds and round to 3dp to be nice to users :)
        document.getElementById(timer).innerHTML = `${title}${(time/1000).toFixed(3)} seconds`
    }

    // Grabs a random sample of the input data and displays it to the user
    async showExamples(data) {
        // Create a container in the visor
        const surface =
            tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});
        tfvis.visor().setActiveTab("Input Data");
        if (this.clearHistoryInput) {surface.drawArea.textContent = '';}

        // Get the examples
        const examples = data.nextTestBatch(this.exampleCount);
        const numExamples = examples.xs.shape[0];

        // Create a canvas element to render each example
        for (let i = 0; i < numExamples; i++) {
            const imageTensor = tf.tidy(() => {
                // Reshape the image
                return examples.xs
                    .slice([i, 0], [1, examples.xs.shape[1]])
                    .reshape(this.IMAGE_RESHAPE)
                    .transpose(this.RESHAPE_TRANSPOSE);
            });  

            const canvas = document.createElement('canvas');
            canvas.width = this.imageWidth;
            canvas.height = this.imageHeight;
            canvas.style = 'margin: 4px;';
            await tf.browser.toPixels(imageTensor, canvas);
            surface.drawArea.appendChild(canvas);

            imageTensor.dispose();
        }
    }

    // Builds the AI model based on the input configs
    getModel(targets, isFit) {
        let model = tf.sequential();

        for (let x =1; x <= this.layerCount; x++) {
            model = this.makeConvPoolLayers(model, isFit, x);
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

    // Helper function to make convolutional layers
    makeConvPoolLayers(model, isFit, convLayerNum) {
        // convLayerNum is int for the number of the layer in the sequence

        let config = {
            kernelSize: this.kernelSize,
            filters: this.filtersMultiplier * this.filterPowerBase**convLayerNum,
            strides: this.filterStrides,
            trainable: !isFit,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        };

        // In the first layer of our convolutional neural network we have
        // to specify the input shape. Then we specify some parameters for
        // the convolution operation that takes place in this layer.
        if (convLayerNum === 1) {
            config.inputShape = this.imageShape;
        }
        if (isFit) {
            config.weights = this.global_model.layers[(convLayerNum-1)*2].getWeights();
        }

        model.add(tf.layers.conv2d(config));

        // The MaxPooling layer acts as a sort of downsampling using max values
        // in a region instead of averaging.
        model.add(tf.layers.maxPooling2d({poolSize: [this.poolSize, this.poolSize], strides: [this.poolStride, this.poolStride], trainable: !isFit}));

        return model;
    }

    // The core training function
    async train(model, data) {

        // Setup tfVis display for training
        const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
        const container = {
            name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
        };
        const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

        // Get data setup for training 
        const TRAIN_DATA_SIZE = data.num_train_elements;
        const TEST_DATA_SIZE = data.num_test_elements;

        const [trainXs, trainYs] = tf.tidy(() => {
            const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
            return [
                d.xs.reshape([TRAIN_DATA_SIZE].concat(this.IMAGE_RESHAPE)).transpose(this.CONCAT_RESHAPE_TRANSPOSE),
                d.labels
            ];
        });

        const [testXs, testYs] = tf.tidy(() => {
            const d = data.nextTestBatch(TEST_DATA_SIZE);
            return [
                d.xs.reshape([TEST_DATA_SIZE].concat(this.IMAGE_RESHAPE)).transpose(this.CONCAT_RESHAPE_TRANSPOSE),
                d.labels
            ];
        });

        // model.fit calls the training loop
        return model.fit(trainXs, trainYs, {
            batchSize: data.batchSize,
            validationData: [testXs, testYs],
            epochs: this.numEpochs, // Iterations over the whole dataset
            shuffle: true,
            callbacks: fitCallbacks
        });
    }

    // Once training is done, this is called to show how the model preforms
        // This will also attempt to call show2dPlot, but only if the output of the model is small enough, unless manual override is enabled (both configured in config.json)
    async showStats(model, data) {
        // Gets prediction for the test dataset
        const [preds, labels, testData] = this.doPrediction(model, data, data.testingSetSize);

        // Checks if demensional reduction is feasable on the output size
            // If so, runs show2dPlot
        const numParams = model.layers[model.layers.length - 2].outputShape[1];
        let drConfig = this.masterConfig.defaults.script.demensionReduction;
        if (numParams <= drConfig.sizeLimit || drConfig.ignoreLimit) {
            await this.show2dPlot(model, data);
        }

        // Get accuracy stats
        const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
        const accContainer = {name: 'Accuracy', tab: 'Evaluation'};
        const targetNameList = [data.targets.fine[0].displayName, data.targets.fine[1].displayName]

        // Use tfVis to show the accuracy stats
        tfvis.show.perClassAccuracy(accContainer, classAccuracy, targetNameList);
        tfvis.visor().setActiveTab("Evaluation");

        // Generate the confusion matrix
        const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
        const confusionContainer = {name: 'Confusion Matrix', tab: 'Evaluation'};

        // Use tfVis to show confusion matrix
        tfvis.render.confusionMatrix(confusionContainer, {values: confusionMatrix, tickLabels: targetNameList});

        this.showWrongExamples(preds, labels, testData, targetNameList);
    }

    // Helper function to make predictions
    doPrediction(model, data, testDataSize) {

        const testData = data.nextTestBatch(testDataSize);
        const testxs = testData.xs.reshape([testDataSize].concat(this.IMAGE_RESHAPE)).transpose(this.CONCAT_RESHAPE_TRANSPOSE);
        const labels = testData.labels.argMax(-1);
        const preds = model.predict(testxs).argMax(-1);

        testxs.dispose();
        return [preds, labels, testData];
    }

    // Once the training is done, this is called to generate a 2d estimation plot of the model's raw output
    async show2dPlot(model, inputData) {

        // Gets labels and an array of Float64Arrays containing the output of the prediction
        const [contextLabels, splitOutPuts] = await this.constPredict(model, inputData);

        // Create druid object for demension redution
        var druidObj = new druid.PCA(druid.Matrix.from(splitOutPuts), 2)

        // Druid solution returns a druid matrix, so this must be converted into an array
        //  This is the computationally intense part
        var reducedArray = druidObj.transform().to2dArray;

        // Plot the redued array using the given labels for colouring
        this.plot(reducedArray, contextLabels, inputData.targets)
    }

    // For making the 2d plot after training, this grabs a constant sample to prevent any major changes in the plot from creating confusion
    async constPredict(model, inputData) {
        // Gets a consistent set of sample data (for the given label choices) 
        //  Sample size will either be defined by constant constSampleMaxSize (set in config.json) or the entirity of the test set if it is smaller than 1000
        const [testData, sampleSize] = inputData.getConstTestSample();
        const testxs = testData.xs.reshape([sampleSize].concat(this.IMAGE_RESHAPE)).transpose(this.CONCAT_RESHAPE_TRANSPOSE);
        const labels = testData.labels;

        // Clones the existing model, without the final dense layer
        const layerOutputModel = tf.model({inputs:model.inputs, outputs: model.layers[model.layers.length - 2].output});
        // Get predition from cloned model
        const predsFloat32Array = await layerOutputModel.predict(testxs).data();

        // Split predsFloat32Array into an arrary of arrays for demension reduction 
        const perElementCount = layerOutputModel.output.shape[1];
        var splitOutPuts = [];
        for (var i = 0; i < predsFloat32Array.length; i += perElementCount) {
            splitOutPuts.push(Float64Array.from(predsFloat32Array.subarray(i, i + perElementCount)));
        }

        // Convert labels from binaray array based on all classes, to binary flag for label A or B
        // ie: [1, 0, 1, 0, 0, 1] => [0, 0, 1]
        const rawLabels = await labels.data()
        const contextLabels = []
        for (var i=0; i<rawLabels.length; i+= 2) {
            contextLabels.push(rawLabels[i] == 0 ? 0 : 1)
        }

        return [contextLabels, splitOutPuts]
    }

    // Takes the demensionally reduced array and the corresponding labels, then generates a nice 2d plot graph of it
    plot(reducedArray, contextLabels, targets) {

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

    // Shows some examples of the wrong predictions (under the confusion matrix)
    async showWrongExamples(preds, labels, testData, targetNameList) {
        const rawPreds = await preds.data();
        const rawLabels = await labels.data();

        // Setup the tsVis surface to hold each example set
        const surface01 =
            tfvis.visor().surface({ name: `Predicted ${targetNameList[0]} labeled ${targetNameList[1]}`, tab: 'Evaluation'});
        const surface10 =
            tfvis.visor().surface({ name: `Predicted ${targetNameList[1]} labeled ${targetNameList[0]}`, tab: 'Evaluation'});

        // This array is used to count how many of each label have been displayed to prevent the max count being only for one label
        let wrong = [0,0];
        // Create a canvas element to render each example
        for (let index = 0; index < rawPreds.length; index++) {
            if (rawPreds[index] !== rawLabels[index] && wrong[rawPreds[index]]++ < this.maxWrong) {

                const imageTensor = tf.tidy(() => {
                    // Reshape the image to 28x28 px
                    return testData.xs
                        .slice([index, 0], [1, testData.xs.shape[1]])
                        .reshape(this.IMAGE_RESHAPE)
                        .transpose(this.RESHAPE_TRANSPOSE);
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
            if (wrong[0] >= this.maxWrong && wrong[1] >= this.maxWrong) {
                break;
            }
        }

        labels.dispose();
    }
}

// ###################### Globals ################################################################

let scriptObject = new ScriptObject();

// ######################################### Set html listeners ###############################################

// Set click listener for the start button
document.getElementById('startButton').onclick = async function() {
    const targets = scriptObject.setup();
    scriptObject.global_model = await scriptObject.run(targets, false);
};

// Set click listener for the re-fit button
document.getElementById('reFitButton').onclick = function() {
    const targets = scriptObject.setup();
    scriptObject.run(targets, true);
};

// If the training scale input is shown, this function prevents it from exceeding 100 or going below 1
document.getElementById('train_scale_input').oninput = function() {
    if (document.getElementById('train_scale_input').value > 100) {
        document.getElementById('train_scale_input').value = 100;
    }
    if (document.getElementById('train_scale_input').value < 1) {
        document.getElementById('train_scale_input').value = 1;
    }
};

 // Wehn label 1 is selected:
    // Dissable start and re-fit buttons (prevent starting without 2 labels)
    // Enable label 2 selector
    // Set the options for label 2
document.getElementById('element_1').onchange = function() {
    document.getElementById('reFitButton').disabled = true;
    document.getElementById('startButton').disabled = true;

    document.getElementById('element_2').disabled = false;
    let filteredItems = scriptObject.fineLabelOptions.filter(e => {return e !== document.getElementById('element_1').value})
    scriptObject.setOptions('element_2', filteredItems)
}

// When label 2 is selected, enable the start button, but only enable the re-fiit button if a model from a pervious run exists
document.getElementById('element_2').onchange = function() {
    document.getElementById('reFitButton').disabled = typeof scriptObject.global_model === 'undefined';
    document.getElementById('startButton').disabled = false;
}

// Setup boot function to be called when the html doc has loaded
document.addEventListener('DOMContentLoaded', scriptObject.boot());