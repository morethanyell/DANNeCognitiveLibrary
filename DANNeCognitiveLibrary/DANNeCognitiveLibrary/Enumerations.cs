using System;

namespace DANNeCognitiveLibrary {

    /// <summary>
    /// A collection of Enumerative values used as tools in this library.
    /// </summary>
    public static class Enumerations {

        /// <summary>
        /// The type of incoming signal to a <see cref="Neuron"/>.
        /// </summary>
        public enum InputTypes {
            /// <summary>
            /// A single-value input
            /// </summary>
            Scalar = 0,
            /// <summary>
            /// A one-dimensional input array - comprised of a chain of scalar values
            /// </summary>
            Vector = 1,
            /// <summary>
            /// A two-dimensional input array - comprised of row and columns
            /// </summary>
            Matrix = 2
        }

        /// <summary>
        /// Allows you to choose what type of randomizer is to be used for
        /// initializing <see cref="Neuron.Weights"/> and <see cref="Neuron.Bias"/>
        /// </summary>
        public enum Randomizers {
            /// <summary>
            /// .Net Library <see cref="System.Random"/>
            /// </summary>
            Random = 0,
            /// <summary>
            /// .Net Library <see cref="System.Security.Cryptography"/>
            /// </summary>
            Cryptographic = 1,
            /// <summary>
            /// From the open source Mathematics and Scientific library <see cref="MathNet"/>, check <see cref="Numerics.Random.MersenneTwister"/> documentation / website at https://numerics.mathdotnet.com. The developer reserves the right to favor this type as the default PRNG in this project
            /// </summary>
            MersenneTwister = 2
        }

        /// <summary>
        /// Activation options for neurons
        /// </summary>
        public enum ActivationFunctions {
            /// <summary>
            /// S-like curve function that maps a value into a floating number between 0 and 1
            /// </summary>
            Sigmoid = 0,
            /// <summary>
            /// S-like curve function that maps a value into a floating number between -1 and 1
            /// </summary>
            Softmax = 1,
            /// <summary>
            /// See <see cref="Math.Tanh(double)"/>
            /// </summary>
            HyperbolicTangent = 2
        }

    }
}
