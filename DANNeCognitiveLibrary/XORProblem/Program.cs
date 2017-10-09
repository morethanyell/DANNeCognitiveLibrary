using GenericParsing;
using NeuralNetworkLibrary;
using System;
using System.Data;
using System.Threading.Tasks;
using static NeuralNetworkLibrary.Enumerations.ActivationFunctions;
namespace XORProblem {
    class Program {
        static void Main(string[] args) {

            // Store the training data into a datatable

            var inpgetter = GetCSV(@"C:\Users\daniel.l.astillero\Documents\Visual Studio 2017\Projects\AccentureCognitiveLibrary\DATASETS\Pandora Dataset - Input Normalized.csv");
            var outgetter = GetCSV(@"C:\Users\daniel.l.astillero\Documents\Visual Studio 2017\Projects\AccentureCognitiveLibrary\DATASETS\Pandora Dataset - Output Normalized.csv");

            Task.WaitAll(inpgetter, outgetter);

            var dtInput = inpgetter.Result as DataTable;

            var dtOutput = outgetter.Result as DataTable;

            var ffnn = new NeuralNetwork();

            ffnn.SetTrainingInputData(dtInput);

            ffnn.AddHiddenLayer(numOfHiddenNeurons: 22, activationFunction: Sigmoid);

            ffnn.AddHiddenLayer(numOfHiddenNeurons: 22, activationFunction: Sigmoid);

            ffnn.SetTrainingOutputData(dtOutput);

            ffnn.LearningRate = 0.2f;

            ffnn.Epoch = 100000;

            ffnn.Train();

            Console.WriteLine("\n\nDone with training.");

            while (true) {
                Console.ReadKey();
            }

        }

        private static async Task<DataTable> GetCSV(string filepath) {
            DataTable retval = await Task.Run(() => {
                var adapter = new GenericParserAdapter(filepath);
                adapter.FirstRowHasHeader = true;
                DataTable dt = adapter.GetDataTable();
                return dt;
            });
            return retval;
        }

    }
}

