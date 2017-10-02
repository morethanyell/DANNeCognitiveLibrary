using System;

namespace DANNeCognitiveLibrary {

    /// <summary>
    /// Implements a <see cref="Neuron"/> as <b>Output Neuron</b> using the implementation provided by the class Neuron.
    /// This class is serializable, which means its current state in the memory can be translated into a file on 
    /// disk using a <see cref="System.Runtime.Serialization.Formatters.Binary.BinaryFormatter"/>
    /// </summary>
    [Serializable]
    public class OutputNeuron : Neuron {

        /// <summary>
        /// Creates an instance of a Output Neuron.
        /// </summary>
        public OutputNeuron(int _synapses) {
            this.TotalSynapses = _synapses;
            this.Weights = new float[this.TotalSynapses];
        }

    }
}
