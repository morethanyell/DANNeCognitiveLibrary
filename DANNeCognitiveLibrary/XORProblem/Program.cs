using DANNeCognitiveLibrary;
using System;
using System.Collections.Generic;
using System.Data;
using static DANNeCognitiveLibrary.Enumerations;

namespace XORProblem {
    class Program {
        static void Main(string[] args) {

            // XOR truth table Neural Network
            // by Daniel L. Astillero
            // Using DANNe Cognitive Library

            // Step 1. Create instances of the input neurons and set the input type
            // Normalize the values (or not--this is just the Truth Table XOR problem)
            Console.WriteLine("\nStep 1. Retrieve the real-world input value froma CSV file, or hard code it. Normalize the values (or not--this is just the Truth Table XOR problem)");

            DataTable inputTable = new DataTable("Inputs");

            DataColumn leftcol = new DataColumn("Left") { DataType = typeof(float) };
            DataColumn rightcol = new DataColumn("Right") { DataType = typeof(float) };
            DataColumn anscol = new DataColumn("Answer") { DataType = typeof(float) };

            inputTable.Columns.Add(leftcol); inputTable.Columns.Add(rightcol); inputTable.Columns.Add(anscol);

            var r = inputTable.NewRow();
            r[0] = 0f; r[1] = 0f; r[2] = 0f;
            inputTable.Rows.Add(r);

            r = inputTable.NewRow();
            r[0] = 1f; r[1] = 0f; r[2] = 1f;
            inputTable.Rows.Add(r);

            r = inputTable.NewRow();
            r[0] = 0f; r[1] = 1f; r[2] = 1f;
            inputTable.Rows.Add(r);

            r = inputTable.NewRow();
            r[0] = 1f; r[1] = 1f; r[2] = 0f;
            inputTable.Rows.Add(r);

            leftcol.Dispose(); rightcol.Dispose(); anscol.Dispose();

            // Step 2.1. Display the data on console
            Console.WriteLine("\n\tL\tXOR\tR\t=\tA");
            Console.WriteLine("        ---------------------------------");

            foreach (DataRow row in inputTable.Rows) {
                Console.WriteLine($"\t{row[0]}\tXOR\t{row[1]}\t=\t{row[2]}");
            }

            // Step 3. Adhoc variable for number of synapses - 2 for each neuron
            Console.WriteLine("\nStep 3. Adhoc variable for number of synapses - 2");
            var synapses = 2;

            // Step 4. Initialize two or more hidden neurons
            Console.WriteLine("\nStep 4. Initialize two or more hidden neurons");

            var hidN1L2 = new HiddenNeuron(synapses) {
                Name = "1st Hidden Neuron",
                MemberOfLayer = 2,
                Randomizer = Randomizers.MersenneTwister,
                InputType = InputTypes.Vector,
            };

            hidN1L2.InitializeWeights(); hidN1L2.InitializeBias();

            var hidN2L2 = new HiddenNeuron(synapses) {
                Name = "2nd Hidden Neuron",
                MemberOfLayer = 2,
                Randomizer = Randomizers.MersenneTwister,
                InputType = InputTypes.Vector,
            };

            hidN2L2.InitializeWeights(); hidN2L2.InitializeBias();

            var hidN3L2 = new HiddenNeuron(synapses) {
                Name = "3rd Hidden Neuron",
                MemberOfLayer = 2,
                Randomizer = Randomizers.MersenneTwister,
                InputType = InputTypes.Vector,
            };

            hidN3L2.InitializeWeights(); hidN3L2.InitializeBias();

            var hidN4L2 = new HiddenNeuron(synapses) {
                Name = "4th Hidden Neuron",
                MemberOfLayer = 2,
                Randomizer = Randomizers.MersenneTwister,
                InputType = InputTypes.Vector,
            };

            hidN4L2.InitializeWeights(); hidN4L2.InitializeBias();


            // Step 4.1 Print the initial weights
            Console.WriteLine();
            for (int i = 0; i < hidN1L2.Weights.Length; i++) Console.WriteLine($"\tHidden Neuron 1 Weight #{i}\t: " + hidN1L2.Weights[i]);
            Console.WriteLine("\tHidden Neuron 1 Bias\t\t: " + hidN1L2.Bias);
            for (int i = 0; i < hidN2L2.Weights.Length; i++) Console.WriteLine($"\tHidden Neuron 2 Weight #{i}\t: " + hidN2L2.Weights[i]);
            Console.WriteLine("\tHidden Neuron 2 Bias\t\t: " + hidN2L2.Bias);
            for (int i = 0; i < hidN3L2.Weights.Length; i++) Console.WriteLine($"\tHidden Neuron 3 Weight #{i}\t: " + hidN3L2.Weights[i]);
            Console.WriteLine("\tHidden Neuron 3 Bias\t\t: " + hidN3L2.Bias);
            for (int i = 0; i < hidN4L2.Weights.Length; i++) Console.WriteLine($"\tHidden Neuron 4 Weight #{i}\t: " + hidN4L2.Weights[i]);
            Console.WriteLine("\tHidden Neuron 4 Bias\t\t: " + hidN4L2.Bias);

            // Step 5. Initialize one output neuron
            Console.WriteLine("\nStep 5. Initialize one output neuron");
            var outN1 = new OutputNeuron(4) {
                Name = "Lone Output Neuron",
                MemberOfLayer = 3,
                Randomizer = Randomizers.MersenneTwister,
                InputType = InputTypes.Vector
            };

            outN1.InitializeWeights(); outN1.InitializeBias();

            // Step 5.1 Print the initial weights
            for (int i = 0; i < outN1.Weights.Length; i++) Console.WriteLine($"\tOutputNeuron Weight #{i}\t: " + outN1.Weights[i]);
            Console.WriteLine("\tOutput Neuron Bias\t: " + outN1.Bias);

            // Step 6. Create a Artificial Neural Network (ANN) out of the neurons from abovementioned
            Console.WriteLine("\nStep 6. Create an Artificial Neural Network (ANN) out of the neurons from abovementioned");
            var ann = new NeuralNetwork() { Epochs = 2000 };

            // Step 6.1. Initialize properties of the ANN, such as Epoch, List of Hidden Neurons, Training Data, etc...
            Console.WriteLine("\nStep 6.1. Initialize properties of the ANN, such as Epoch, List of Hidden Neurons, Training Data, etc...");

            // Set training data
            ann.SetTrainingData(inputTable);

            // Set number of desired hidden layers
            ann.TotalHiddenLayers = 1;

            // Set or add the hidden neurons
            ann.HiddenNeurons = new List<HiddenNeuron> {
                hidN1L2,
                hidN2L2,
                hidN3L2,
                hidN4L2
            };

            // Set add the output neuron(s)
            ann.OutputNeurons = new List<OutputNeuron> {
                outN1
            };

            // Set the weights matrices
            ann.SetWeightsMatrices();

            // Step 7. Make a feed forward just to check everything's working
            Console.WriteLine("\nStep 7. Run a feed forward just to check everything's working");

            Console.WriteLine("\nStep 8. Set the epoch and then start the training.");

            ann.Epochs = 10000;
            ann.LearningRate = 0.1f;
            ann.StartTraining(applyLearningRate: true);

            Console.WriteLine("\nStep 9. Done with training! Now we can test the model.");

            while (true) {

                try {
                    Console.Write("\n\tEnter left number: ");
                    var left = float.Parse(Console.ReadLine());
                    Console.Write("\tEnter right number: ");
                    var right = float.Parse(Console.ReadLine());
                    Console.Write("\tEnter expected answer: ");
                    var ans = float.Parse(Console.ReadLine());

                    using (DataTable dt = new DataTable("test")) {
                        DataColumn leftcol1 = new DataColumn("Left") { DataType = typeof(float) };
                        DataColumn rightcol1 = new DataColumn("Right") { DataType = typeof(float) };
                        DataColumn anscol1 = new DataColumn("Answer") { DataType = typeof(float) };

                        dt.Columns.Add(leftcol1); dt.Columns.Add(rightcol1); dt.Columns.Add(anscol1);


                        var r1 = dt.NewRow();
                        r1[0] = left; r1[1] = right; r1[2] = ans;
                        dt.Rows.Add(r1);

                        ann.SingleInstanceFeedForward(dt);
                    }

                } catch {
                    break;
                }

            }

            Console.ReadKey();

        }
    }
}

