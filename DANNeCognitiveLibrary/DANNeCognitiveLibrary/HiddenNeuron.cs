using System;

namespace DANNeCognitiveLibrary {

    /// <summary>
    /// Implements a <see cref="Neuron"/> as <b>Hidden Neuron</b> using the implementation provided by the class Neuron.
    /// This class is serializable, which means its current state in the memory can be translated into a file on 
    /// disk using a <see cref="System.Runtime.Serialization.Formatters.Binary.BinaryFormatter"/>
    /// </summary>
    [Serializable]
    public class HiddenNeuron : Neuron {

        /// <summary>
        /// Creates an instance of a Hidden Neuron.
        /// </summary>
        public HiddenNeuron(int _synapses) {
            this.TotalSynapses = _synapses;
            this.Weights = new float[this.TotalSynapses];
        }
    }
}
