using System;

namespace NeuralNetworkLibrary {

    /// <summary>
    /// Implements a <see cref="Neuron"/> as <b>Hidden Neuron</b> using the implementation provided by the class Neuron.
    /// </summary>
    public class HiddenNeuron : Neuron {

        /// <summary>
        /// Creates an instance of a Hidden Neuron.
        /// </summary>
        public HiddenNeuron(int _synapses) {
            this.TotalSynapses = _synapses;
            this.Weights = new float[this.TotalSynapses];
            this.InitializeWeights();
            this.InitializeBias();
        }
    }
}
