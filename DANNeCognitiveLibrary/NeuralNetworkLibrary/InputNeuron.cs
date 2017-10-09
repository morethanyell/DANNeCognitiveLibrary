using System;

namespace NeuralNetworkLibrary {

    /// <summary>
    /// A class that holds the value of the input signal in a feed-forward neural network.
    /// It is expected that the value is already normalized.
    /// </summary>
    public class InputNeuron {

        /// <summary>
        /// The real-world value, usually from tranining data or use-case test data
        /// </summary>
        public float Value { get; set; }

        /// <summary>
        /// Creates an instance of the
        /// </summary>
        public InputNeuron() { }

        /// <summary>
        /// Creates an instance of the
        /// </summary>
        /// <param name="value">Initializes the value of this input neuron</param>
        public InputNeuron(float value) { this.Value = value; }

    }
}
