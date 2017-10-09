using System;

namespace NeuralNetworkLibrary {

    /// <summary>
    /// Implements a <see cref="Neuron"/> as <b>Output Neuron</b> using the implementation provided by the class Neuron.
    /// </summary>
    public class OutputNeuron : Neuron {

        /// <summary>
        /// Creates an instance of a Output Neuron.
        /// </summary>
        public OutputNeuron(int _synapses) {
            this.TotalSynapses = _synapses;
            this.Weights = new float[this.TotalSynapses];
            this.InitializeWeights();
            this.InitializeBias();
        }

    }
}
