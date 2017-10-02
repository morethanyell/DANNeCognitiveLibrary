using System;
using static DANNeCognitiveLibrary.Enumerations;
using static DANNeCognitiveLibrary.ActivationFunctionUtilities;

namespace DANNeCognitiveLibrary {

    /// <summary>
    /// Represents the abstraction of an Artificial Neural Network object called a "<b>Neuron</b>" from 
    /// which all implementations of Neuron derive.
    /// </summary>
    public abstract class Neuron {

        /// <summary>
        /// Protected constructor function.
        /// </summary>
        protected Neuron() { }

        /// <summary>
        /// Accessory property that gives the neuron a name.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the layer heirarchical layer where this neuron belongs
        /// </summary>
        public int MemberOfLayer { get; set; }

        /// <summary>
        /// Gets or sets the type of input as defined in <see cref="Enumerations.InputTypes"/>.
        /// </summary>
        public InputTypes InputType { get; set; }

        /// <summary>
        /// Gets or sets the type of <see cref="Enumerations.ActivationFunctions"/> used by this neuron
        /// </summary>
        public ActivationFunctions ActivationFunction { get; set; }

        /// <summary>
        /// The incoming signal as float vector input with the assumption that such signals are already 
        /// normalized and/or Sigmoidal ahead of time.
        /// </summary>
        public float[] Inputs { get; set; }

        /// <summary>
        /// The value or result of the <b>f</b>(x) that was activated using
        /// the an activation function.
        /// </summary>
        public float FofX { get; private set; }

        /// <summary>
        /// The number of synapses / weights braning from the incoming signals.
        /// </summary>
        public int TotalSynapses { get; set; }

        /// <summary>
        /// Synapse or the weight of this intance of <see cref="Neuron"/>. This is synonymous to the 
        /// <i>m</i> in the function <i>f(x) = mx + b</i>.
        /// </summary>
        public float[] Weights { get; internal set; }

        /// <summary>
        /// The bias constant that is added to this function. This is synonymous to the 
        /// <i>b</i> in the function <i>f(x) = mx + b</i>.
        /// </summary>
        public float Bias { get; internal set; }

        /// <summary>
        /// The current error rate of this neuron.
        /// </summary>
        public float Error { get; internal set; }

        /// <summary>
        /// The type of randomizer to be used in this instance.
        /// </summary>
        public Randomizers Randomizer { get; set; }

        /// <summary>
        /// Assigns random value to the weights.
        /// </summary>
        public void InitializeWeights() {
            switch (this.Randomizer) {
                case Randomizers.Random: // Simple PRNG
                    for (int i = 0; i < this.Weights.Length; i++)
                        this.Weights[i] = (float)(new Random(DateTime.Now.Millisecond).NextDouble()); // call the next double given the seed
                    break;
                case Randomizers.Cryptographic: // Cryptographic PRNG
                    for (int i = 0; i < this.Weights.Length; i++) {
                        byte[] bytes = new byte[16]; // initialize a 16-length empty byte array
                        using (var rng = new System.Security.Cryptography.RNGCryptoServiceProvider()) {  // initialize an RNGCryptoServiceProvider
                            rng.GetNonZeroBytes(bytes); // fill the byte array with non-zero bytes from the RNGCryptoServiceProvider
                            int floatation = 0; // ad-hoc variable for concatation purposes
                            foreach (byte b in bytes) floatation += (int)b; // concatenate each byte to the adhoc integer
                            float randomFloat = float.Parse("0." + floatation.ToString()); // create a float by parsing "0." + the adhoc integer
                            this.Weights[floatation] = randomFloat; // assign the new float to the weights
                        }
                    }
                    break;
                case Randomizers.MersenneTwister: // MersenneTwister PRNG
                    for (int i = 0; i < this.Weights.Length; i++)
                        this.Weights[i] = (float)(MathNet.Numerics.Random.MersenneTwister.Default.NextDouble()); // call the next MersenneTwister double
                    break;
                default: // Favored MersenneTwister
                    for (int i = 0; i < this.Weights.Length; i++)
                        this.Weights[i] = (float)(MathNet.Numerics.Random.MersenneTwister.Default.NextDouble()); // call the next MersenneTwister double
                    break;
            }
        }

        /// <summary>
        /// Assigns a random value to the bias.
        /// </summary>
        public void InitializeBias() {
            switch (this.Randomizer) {
                case Randomizers.Random: // Simple PRNG
                    this.Bias = (float)(new Random(DateTime.Now.Millisecond).NextDouble()); // call the next double given the 
                    break;
                case Randomizers.Cryptographic: // Cryptographic PRNG
                    byte[] bytes = new byte[16]; // initialize a 16-length empty byte array
                    using (var rng = new System.Security.Cryptography.RNGCryptoServiceProvider()) {  // initialize an RNGCryptoServiceProvider
                        rng.GetNonZeroBytes(bytes); // fill the byte array with non-zero bytes from the RNGCryptoServiceProvider
                        int floatation = 0; // ad-hoc variable for concatation purposes
                        foreach (byte b in bytes) floatation += (int)b; // concatenate each byte to the adhoc integer
                        float randomFloat = float.Parse("0." + floatation.ToString()); // create a float by parsing "0." + the adhoc integer
                        this.Bias = randomFloat; // assign the new float to the weights
                    }
                    break;
                case Randomizers.MersenneTwister: // MersenneTwister PRNG
                    this.Bias = (float)(MathNet.Numerics.Random.MersenneTwister.Default.NextDouble()); // call the next MersenneTwister double
                    break;
                default: // Favored MersenneTwister
                    this.Bias = (float)(MathNet.Numerics.Random.MersenneTwister.Default.NextDouble()); // call the next MersenneTwister double
                    break;
            }
        }

        /// <summary>
        /// Adjusts all the <see cref="Weights"/> of this neuron based on the <see cref="Error"/>.
        /// </summary>
        /// <param name="applyLearningRate">Optional parameter that will either apply
        /// the <see cref="LearningRateAlpha"/> or not.</param>
        public void AdjustWeights(bool applyLearningRate = false, float learningRate = 1.0f) {
            if (applyLearningRate) {
                for (int i = 0; i < this.Weights.Length; i++) {
                    this.Weights[i] += this.Inputs[i] * (this.Error * learningRate);
                }
            } else {
                for (int i = 0; i < this.Weights.Length; i++) {
                    this.Weights[i] += this.Inputs[i] * this.Error;
                }
            }
        }

        /// <summary>
        /// Fires this neuron or sets the f(x) value of this by activiating the weights, input, value,
        /// and/or bias if needed
        /// </summary>
        public void Activate() {
            float fofx = 0;
            for (int i = 0; i < this.TotalSynapses; i++) fofx += (this.Weights[i] * this.Inputs[i]) + this.Bias;
            switch (this.ActivationFunction) {
                case ActivationFunctions.Sigmoid:
                    this.FofX = SigmoidActivation(fofx);
                    break;
                case ActivationFunctions.HyperbolicTangent:
                    this.FofX = HyperbolicTangent(fofx);
                    break;
                default:
                    this.FofX = SigmoidActivation(fofx);
                    break;
            }
        }

    }
}
