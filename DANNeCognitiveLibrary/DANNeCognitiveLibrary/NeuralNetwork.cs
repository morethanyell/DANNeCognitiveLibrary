using System;
using System.Collections.Generic;
using System.Data;
using static DANNeCognitiveLibrary.ActivationFunctionUtilities;

namespace DANNeCognitiveLibrary {

    /// <summary>
    /// A class that holds a collection of <see cref="Neuron"/>s and has methods of training an Artificial Neural Network model. 
    /// This class is serializable, which means its current state in the memory can be translated into a file on disk 
    /// using a <see cref="System.Runtime.Serialization.Formatters.Binary.BinaryFormatter"/>
    /// </summary>
    [Serializable]
    public class NeuralNetwork {

        /// <summary>
        /// Creates an instance of a Neural Network
        /// </summary>
        public NeuralNetwork() { }

        /// <summary>
        /// The DataTable container for the training data. It is <b>IMPORTANT</b> to note
        /// that this DataTable assumes the last column (or field) as the
        /// expected output value
        /// </summary>
        public DataTable TrainingDataTable { get; internal set; }

        /// <summary>
        /// Gets the number of input neurons in this Neural Network
        /// </summary>
        public int TotalInputs { get; internal set; }

        /// <summary>
        /// Gets or sets the List of input neurons in this Neural Network
        /// </summary>
        public List<Neuron> Inputs { get; set; }

        /// <summary>
        /// Gets the number of hidden neurons in this Neural Network
        /// </summary>
        public int TotalHiddenNeurons { get; internal set; }

        /// <summary>
        /// Gets or sets the List of hidden neurons in this Neural Network
        /// </summary>
        public List<HiddenNeuron> HiddenNeurons { get; set; }

        /// <summary>
        /// Gets or sets the list of output neuron(s) in this Neural Network
        /// </summary>
        public List<OutputNeuron> OutputNeurons { get; set; }

        /// <summary>
        /// Gets the number of hidden layers in this Neural Network
        /// </summary>
        public int TotalHiddenLayers { get; internal set; }

        /// <summary>
        /// Gets the number of output neurons in this Neural Network
        /// </summary>
        public int TotalOutputs { get; internal set; }

        /// <summary>
        /// Gets or sets the number epochs to be used in training this Neural Network
        /// </summary>
        public int Epochs { get; set; }

        /// <summary>
        /// The learning rate applied during the adjustment of the <see cref="Weights"/> 
        /// and <see cref="Bias"/>. As an optional variable during instantiation, 
        /// this will have a default value of 1.0.
        /// </summary>
        public float LearningRate { get; set; }

        /// <summary>
        /// Sets the training data from a <see cref="DataTable"/>
        /// </summary>
        /// <param name="dt">A prepared <see cref="DataTable"/></param>
        public void SetTrainingData(DataTable dt) {
            this.TrainingDataTable = dt.Copy() as DataTable;
        }

        /// <summary>
        /// Sets the training data from a CSV file
        /// </summary>
        /// <param name="excel_file_path">The exact location of the CSV file</param>
        public void SetTrainingData(string csvFilePath) {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Sets the training data from an Excel file
        /// </summary>
        /// <param name="msExcelFilePath">The exact location of the CSV file</param>
        public void SetTrainingData(string msExcelFilePath, int worksheetIndex = 0) {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Runs a feed forward propagation - used for initial assessment as the method
        /// uses the pre-defined training dataset in this network object
        /// </summary>
        public void SingleInstanceFeedForward() {
            // Running through each row of the training data
            foreach (DataRow row in this.TrainingDataTable.Rows) {

                // Set the result as an adhoc variable
                var expectedResult = (float)row[this.TrainingDataTable.Columns.Count - 1];

                // Fire or activate every hidden neuron in this network
                foreach (HiddenNeuron hn in this.HiddenNeurons) {
                    hn.Inputs = new float[this.TrainingDataTable.Columns.Count - 1];
                    // Avoiding the last column as it is always assumed (in this context) 
                    // that the last column is the expected output
                    for (int j = 0; j < this.TrainingDataTable.Columns.Count - 1; j++) {
                        hn.Inputs[j] = (float)row[j];
                        hn.Activate();
                    }
                }

                // Fire the output neuron(s) and print the prediction
                foreach (OutputNeuron on in this.OutputNeurons) {
                    on.Inputs = new float[this.HiddenNeurons.Count];
                    for (int k = 0; k < this.HiddenNeurons.Count; k++) {
                        on.Inputs[k] = this.HiddenNeurons[k].FofX;
                        on.Activate();
                    }
                    Console.WriteLine("\n\tSingle Feedforward Prediction (untrained model)");
                    Console.WriteLine($"\n\t{row[0]} XOR {row[1]} = Expected: {expectedResult} | Observed: {on.FofX:0.00} | Error: {(expectedResult - on.FofX) * 100:0.00}%");
                }
            }
        }

        /// <summary>
        /// Runs a feed forward propagation - used for testing data
        /// </summary>
        /// <param name="testDataTable">The new test dataset from the user</param>
        public void SingleInstanceFeedForward(DataTable testDataTable) {
            // Running through each row of the training data
            foreach (DataRow row in testDataTable.Rows) {

                // Set the result as an adhoc variable
                var expectedResult = (float)row[testDataTable.Columns.Count - 1];

                // Fire or activate every hidden neuron in this network
                foreach (HiddenNeuron hn in this.HiddenNeurons) {
                    hn.Inputs = new float[testDataTable.Columns.Count - 1];
                    // Avoiding the last column as it is always assumed (in this context) 
                    // that the last column is the expected output
                    for (int j = 0; j < testDataTable.Columns.Count - 1; j++) {
                        hn.Inputs[j] = (float)row[j];
                        hn.Activate();
                    }
                }

                // Fire the output neuron(s) and print the prediction
                foreach (OutputNeuron on in this.OutputNeurons) {
                    on.Inputs = new float[this.HiddenNeurons.Count];
                    for (int k = 0; k < this.HiddenNeurons.Count; k++) {
                        on.Inputs[k] = this.HiddenNeurons[k].FofX;
                        on.Activate();
                    }
                    Console.WriteLine("\n\tSingle Feedforward Prediction (trained model)");
                    Console.WriteLine($"\n\t{row[0]} XOR {row[1]} = Expected: {expectedResult} | Observed: {on.FofX:0.00} | Error: {(expectedResult - on.FofX) * 100:0.00}%");
                }
            }
        }

        /// <summary>
        /// Trains the neural network using backpropagation
        /// </summary>
        public void StartTraining(bool applyLearningRate = false) {

            // Validations
            if (this.HiddenNeurons == null) throw new Exception("Invalid network built. Zero or null hidden neurons detected");
            if (this.OutputNeurons == null) throw new Exception("Invalid network built. Zero or null output neurons detected");
            if (this.HiddenNeurons.Count < 1) throw new Exception("Invalid network built. Zero or null hidden neurons detected");
            if (this.OutputNeurons.Count < 1) throw new Exception("Invalid network built. Zero or null output neurons detected");
            if (this.TrainingDataTable == null) throw new Exception("Invalid network built. Zero or null training data detected");

            // Train the model over and over again until number of epoch
            for (int i = 0; i < this.Epochs; i++) {

                // Running through each row of the training data
                foreach (DataRow row in this.TrainingDataTable.Rows) {

                    // Set the result as an adhoc variable
                    var expectedResult = (float)row[this.TrainingDataTable.Columns.Count - 1];

                    // Fire or activate every hidden neuron in this network
                    foreach (HiddenNeuron hn in this.HiddenNeurons) {
                        hn.Inputs = new float[this.TrainingDataTable.Columns.Count - 1];
                        // Avoiding the last column as it is always assumed (in this context) 
                        // that the last column is the expected output
                        for (int j = 0; j < this.TrainingDataTable.Columns.Count - 1; j++) {
                            hn.Inputs[j] = (float)row[j];
                            hn.Activate();
                        }
                    }

                    // Fire the output neuron(s) and print the prediction
                    foreach (OutputNeuron on in this.OutputNeurons) {
                        on.Inputs = new float[this.HiddenNeurons.Count];
                        for (int k = 0; k < this.HiddenNeurons.Count; k++) {
                            on.Inputs[k] = this.HiddenNeurons[k].FofX;
                            on.Activate();
                        }

                        Console.WriteLine($"\t{i + 1},{row[0]},{row[1]},{expectedResult},{on.FofX},{on.Error}"); // CSV like output
                    }

                    // Start of Back Propagation - calculate the error of the output neuron(s) then adjust the weights
                    foreach (OutputNeuron on in this.OutputNeurons) {
                        on.Error = SigmoidDerivative(on.FofX) * (expectedResult - on.FofX);
                        on.AdjustWeights(applyLearningRate, this.LearningRate);

                        // Calculate the error of the hidden neurons then adjust the weights
                        for (int l = 0; l < on.TotalSynapses; l++) {
                            this.HiddenNeurons[l].Error = SigmoidDerivative(this.HiddenNeurons[l].FofX) * on.Error * on.Weights[l];
                            this.HiddenNeurons[l].AdjustWeights(applyLearningRate, this.LearningRate);
                        }
                    }
                }
            }

            // Print the current state of each neurons
            foreach (HiddenNeuron hn in this.HiddenNeurons) {
                Console.Write($"\t{hn.Name} | Weight: ");
                foreach (float w in hn.Weights) Console.Write(w + ", ");
                Console.Write($"Bias: {hn.Bias}\n");
            }

            foreach (OutputNeuron on in this.OutputNeurons) {
                Console.Write($"\t{on.Name} | Weight: ");
                foreach (float w in on.Weights) Console.Write(w + ", ");
                Console.Write($"Bias: {on.Bias}\n");
            }

        }

    }
}
