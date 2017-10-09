using System;
using System.Collections.Generic;
using System.Data;
using static NeuralNetworkLibrary.ActivationFunctionUtilities;
using static NeuralNetworkLibrary.Enumerations;
using MathNet.Numerics.LinearAlgebra;
using static NeuralNetworkLibrary.NeuralNetExceptions;

namespace NeuralNetworkLibrary {

    /// <summary>
    /// A class representation of Feed Forward Neural Network, a "model" so-to-speak, that predicts
    /// output given a set of input. 
    /// This holds a collection of <see cref="Neuron"/>s and has methods such as training and backpropagation. 
    /// </summary>
    public class NeuralNetwork {

        /// <summary>
        /// Gets the training intput data in a form of <see cref="DataTable"/>.
        /// </summary>
        public DataTable TrainingInputData { get; private set; }

        /// <summary>
        /// Gets the number of inputs in this network
        /// </summary>
        public int InputCount { get; private set; }

        /// <summary>
        /// Gets the training output data as list of floating points.
        /// </summary>
        public DataTable TrainingOutputData { get; private set; }

        /// <summary>
        /// Gets the number of output neurons in this network.
        /// </summary>
        public int OutputCount { get; private set; }

        /// <summary>
        /// Gets or sets the hidden layers
        /// </summary>
        public List<List<HiddenNeuron>> HiddenLayers { get; private set; }

        /// <summary>
        /// Gets the number of hidden layers
        /// </summary>
        public int HiddenLayersCount { get; private set; }

        /// <summary>
        /// Holder of the output activation derived from the forward propagation of this network.
        /// </summary>
        public List<OutputNeuron> OutputNeurons { get; private set; }

        /// <summary>
        /// Gets or sets the value of learning rate.
        /// </summary>
        public float LearningRate { get; set; }

        /// <summary>
        /// Gets or sets the value of epoch / training episode.
        /// </summary>
        public int Epoch { get; set; }

        /// <summary>
        /// Represents whether the output from the training output data is
        /// a classification vector.
        /// </summary>
        public bool IsOutputClassifierVector { get; private set; }

        /// <summary>
        /// Creates an instance of Neural Network.
        /// </summary>
        public NeuralNetwork() {
            this.HiddenLayers = new List<List<HiddenNeuron>>();
            this.LearningRate = 1;
            this.Epoch = 2000;
            this.IsOutputClassifierVector = false;
        }

        /// <summary>
        /// Analyzes the datatable parameter as input data for training. The number of columns
        /// in the datatable schema represents the number of input buffer or input neurons
        /// in this network.
        /// </summary>
        /// <param name="dtInput">Expects a table or matrix of preprocessed data for training purposes</param>
        public void SetTrainingInputData(DataTable dtInput) {
            if (dtInput == null) throw new NullReferenceException();
            if (dtInput.Rows.Count < 4) throw new NotEnoughTrainingDataException("Training data must contain at least 4 rows.");
            if (dtInput.Columns.Count < 2) throw new NotEnoughTrainingDataException("Training data must contain at least 2 columns.");

            this.TrainingInputData = dtInput.Copy() as DataTable;
            this.InputCount = this.TrainingInputData.Columns.Count;
        }

        /// <summary>
        /// Analyzes the datatable parameter as output data for training. The number of columns
        /// in the datatable schema represents the number of output neurons in this network
        /// </summary>
        /// <param name="dtOutput">Expects a 1xn table of preprocessed data for training purposes</param>
        public void SetTrainingOutputData(DataTable dtOutput) {
            if (dtOutput == null) throw new NullReferenceException();
            if (dtOutput.Rows.Count != this.TrainingInputData.Rows.Count) throw new InvalidOutputDataForTrainingException("Output data must contain the same number of rows as the input training data.");
            if (dtOutput.Columns.Count < 1) throw new NotEnoughTrainingDataException("Output data must contain at least 1 column.");
            if (this.HiddenLayers == null) throw new NullReferenceException("Hidden layer(s) is not yet set.");
            if (this.HiddenLayers.Count < 1) throw new NullReferenceException("Hidden layer(s) is not yet ready. Hidden layer(s) must contain at least one (1) layer of hidden neurons.");

            this.TrainingOutputData = dtOutput.Copy() as DataTable;
            this.OutputCount = this.TrainingOutputData.Columns.Count;

            var onSynapse = this.HiddenLayers[this.HiddenLayers.Count - 1].Count;
            this.OutputNeurons = new List<OutputNeuron>();
            foreach (DataColumn col in dtOutput.Columns) {
                this.OutputNeurons.Add(new OutputNeuron(onSynapse) {
                    ActivationFunction = ActivationFunctions.Sigmoid
                });
            }
        }

        /// <summary>
        /// Adds a "collection" or Layer of <see cref="HiddenNeuron"/>s to this network.
        /// </summary>
        /// <param name="numOfHiddenNeurons">The total number of desired hidden neurons in this layer</param>
        /// <param name="activationFunction">The chosen activation function</param>
        /// <exception cref="NotEnoughHiddenNeuronsException">Exception is thrown when hidden neurons is less than three (3).</exception>
        public void AddHiddenLayer(int numOfHiddenNeurons, ActivationFunctions activationFunction) {
            if (this.HiddenLayers == null) throw new NullReferenceException("The hidden layers property is not yet instantiated.");
            if (numOfHiddenNeurons < 2) throw new NotEnoughHiddenNeuronsException("The hidden neurons must be greater than two (2).");

            var hnSynapse = 0;
            if (this.HiddenLayers.Count == 0)
                hnSynapse = this.TrainingInputData.Columns.Count;
            else hnSynapse = this.HiddenLayers[this.HiddenLayers.Count - 1].Count;
            List<HiddenNeuron> hn = new List<HiddenNeuron>();
            for (int i = 0; i < numOfHiddenNeurons; i++) {
                hn.Add(new HiddenNeuron(hnSynapse) {
                    ActivationFunction = activationFunction
                });
            }
            this.HiddenLayers.Add(hn);
            this.HiddenLayersCount++;
        }

        /// <summary>
        /// Runs a single instance of feed forward algorithm
        /// </summary>
        public void FeedForward(float[] inputs) {
            for (int i = 0; i < this.HiddenLayersCount; i++) {
                if (i == 0) {
                    for (int j = 0; j < this.HiddenLayers[i].Count; j++) {
                        this.HiddenLayers[i][j].Input = inputs;
                        this.HiddenLayers[i][j].Activate();
                    }
                } else {
                    var _inputs = new float[this.HiddenLayers[i - 1].Count];
                    for (int j = 0; j < this.HiddenLayers[i].Count; j++) {
                        _inputs[j] = this.HiddenLayers[i - 1][j].Activation;
                    }
                    for (int j = 0; j < this.HiddenLayers[i].Count; j++) {
                        this.HiddenLayers[i][j].Input = _inputs;
                        this.HiddenLayers[i][j].Activate();
                    }
                }
            }
            int lastLayer = this.HiddenLayersCount - 1;
            var _lastLayerActivations = new float[this.HiddenLayers[lastLayer].Count];
            for (int j = 0; j < this.HiddenLayers[lastLayer].Count; j++) {
                _lastLayerActivations[j] = this.HiddenLayers[lastLayer][j].Activation;
            }
            for (int i = 0; i < this.OutputCount; i++) {
                this.OutputNeurons[i].Input = _lastLayerActivations;
                this.OutputNeurons[i].Activate();
            }

            Console.Write("\nGuess: ");
            for (int i = 0; i < this.OutputNeurons.Count; i++) Console.Write($"{this.OutputNeurons[i].Activation:0.0000}\t");
        }

        /// <summary>
        /// Runs a single instance of back propagation algorithm
        /// </summary>
        public void BackPropagation(float[] outputs) {

            Console.Write("\nAnswer: ");
            for (int i = 0; i < outputs.Length; i++) Console.Write($"{outputs[i]:0.0000}\t");

            for (int i = 0; i < this.OutputNeurons.Count; i++) {
                this.OutputNeurons[i].Error = SigmoidDerivative(this.OutputNeurons[i].Activation) * (outputs[i] - this.OutputNeurons[i].Activation);
                this.OutputNeurons[i].AdjustWeights(applyLearningRate: true);
            }

            float[] backsignal = new float[this.OutputCount];
            for (int i = 0; i < this.OutputCount; i++) {
                backsignal[i] = this.OutputNeurons[i].Error;
            }

            var c = this.HiddenLayersCount - 1;
            for (int i = c; i >= 0; i--) {
                if (i == c) {
                    for (int j = 0; j < this.HiddenLayers[i].Count; j++) {
                        for (int k = 0; k < backsignal.Length; k++) {
                            this.HiddenLayers[i][j].Error = SigmoidDerivative(this.HiddenLayers[i][j].Activation) * backsignal[k] * this.OutputNeurons[k].Weights[j];
                            this.HiddenLayers[i][j].AdjustWeights(applyLearningRate: true);
                        }
                    }
                } else {
                    backsignal = new float[this.HiddenLayers[i + 1].Count];
                    for (int f = 0; f < backsignal.Length; f++)
                        backsignal[f] = this.HiddenLayers[i + 1][f].Error;
                    for (int j = 0; j < this.HiddenLayers[i].Count; j++) {
                        for (int k = 0; k < backsignal.Length; k++) {
                            this.HiddenLayers[i][j].Error = SigmoidDerivative(this.HiddenLayers[i][j].Activation) * backsignal[k] * this.HiddenLayers[i + 1][k].Weights[j];
                            this.HiddenLayers[i][j].AdjustWeights(applyLearningRate: true);
                        }
                    }
                }
            }
        }


        /// <summary>
        /// Parses the datatable and trains the network for given amount of epochs
        /// </summary>
        /// <exception cref="InvalidCastException"></exception>
        public void Train() {
            for (int i = 0; i < this.Epoch; i++) {
                for (int r = 0; r < this.TrainingInputData.Rows.Count; r++) {
                    var inp = new float[this.InputCount];
                    var oup = new float[this.OutputCount];
                    for (int j = 0; j < this.InputCount; j++) {
                        if (float.TryParse(this.TrainingInputData.Rows[r][j].ToString(), out float parser)) {
                            inp[j] = parser;
                        } else {
                            throw new InvalidCastException("Cannot parse the data from training data table into float");
                        }
                    }
                    for (int j = 0; j < this.OutputCount; j++) {
                        if (float.TryParse(this.TrainingOutputData.Rows[r][j].ToString(), out float parser)) {
                            oup[j] = SigmoidActivation(parser);
                        } else {
                            throw new InvalidCastException("Cannot parse the data from training data table into float");
                        }
                    }

                    FeedForward(inp);
                    BackPropagation(oup);
                }
            }

        }

    }
}
