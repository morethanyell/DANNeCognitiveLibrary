using System;
using static NeuralNetworkLibrary.Enumerations;
using static NeuralNetworkLibrary.ActivationFunctionUtilities;

namespace NeuralNetworkLibrary {

    /// <summary>
    /// Represents the abstraction of an Artificial Neural Network object called a "<b>Neuron</b>" from 
    /// which all implementations of Neuron derive.
    /// </summary>
    public abstract class Neuron {

        const int MIN_WEIGHT_RANGE = -2;
        const int MAX_WEIGHT_RANGE = 2;

        /// <summary>
        /// Protected constructor function.
        /// </summary>
        protected Neuron() { this.Id = new Guid(); }

        /// <summary>
        /// Accessory property that gives the neuron a name.
        /// </summary>
        public Guid Id { get; }

        /// <summary>
        /// Gets or sets the type of <see cref="Enumerations.ActivationFunctions"/> used by this neuron
        /// </summary>
        public ActivationFunctions ActivationFunction { get; set; }

        /// <summary>
        /// The incoming signal as float vector input with the assumption that such signals are already 
        /// normalized and/or Sigmoidal ahead of time.
        /// </summary>
        public float[] Input { get; set; }

        /// <summary>
        /// The value or result of the <b>f</b>(x) that was activated using
        /// the an activation function.
        /// </summary>
        public float Activation { get; private set; }

        /// <summary>
        /// The number of synapses / weights braning from the incoming signals.
        /// </summary>
        public int TotalSynapses { get; set; }

        /// <summary>
        /// Synapse or the weight of this intance of <see cref="Neuron"/>. This is synonymous to the 
        /// <i>m</i> in the function <i>f(x) = mx + b</i>.
        /// </summary>
        public float[] Weights { get; set; }

        /// <summary>
        /// The bias constant that is added to this function. This is synonymous to the 
        /// <i>b</i> in the function <i>f(x) = mx + b</i>.
        /// </summary>
        public float Bias { get; private set; }

        /// <summary>
        /// The current error rate of this neuron.
        /// </summary>
        public float Error { get; set; }

        /// <summary>
        /// Assigns random value to the weights.
        /// </summary>
        public void InitializeWeights() {
            for (int i = 0; i < this.Weights.Length; i++) {
                this.Weights[i] = (float)(MathNet.Numerics.Random.MersenneTwister.Default.NextDouble()); // call the next MersenneTwister double
                this.Weights[i] += MathNet.Numerics.Random.MersenneTwister.Default.Next(MIN_WEIGHT_RANGE, MAX_WEIGHT_RANGE);
            }
        }

        /// <summary>
        /// Assigns a random value to the bias.
        /// </summary>
        public void InitializeBias() {
            this.Bias = (float)(MathNet.Numerics.Random.MersenneTwister.Default.NextDouble()); // call the next MersenneTwister double
            this.Bias += MathNet.Numerics.Random.MersenneTwister.Default.Next(MIN_WEIGHT_RANGE, MAX_WEIGHT_RANGE);
        }

        /// <summary>
        /// Adjusts all the <see cref="Weights"/> of this neuron based on the <see cref="Error"/>.
        /// </summary>
        /// <param name="applyLearningRate">Optional parameter that will either apply
        /// the <see cref="LearningRateAlpha"/> or not.</param>
        public void AdjustWeights(bool applyLearningRate = false, float learningRate = 1.0f) {
            if (applyLearningRate)
                for (int i = 0; i < this.Weights.Length; i++)
                    this.Weights[i] += this.Input[i] * (this.Error * learningRate);
            else
                for (int i = 0; i < this.Weights.Length; i++)
                    this.Weights[i] += this.Input[i] * this.Error;
        }

        /// <summary>
        /// Fires this neuron or sets the f(x) value of this by activiating the weights, input, value,
        /// and/or bias if needed
        /// </summary>
        public void Activate() {
            float fofx = 0;
            for (int i = 0; i < this.TotalSynapses; i++)
                fofx += (this.Weights[i] * this.Input[i]) + this.Bias;

            switch (this.ActivationFunction) {
                case ActivationFunctions.Sigmoid:
                    this.Activation = SigmoidActivation(fofx);
                    break;
                case ActivationFunctions.HyperbolicTangent:
                    this.Activation = HyperbolicTangent(fofx);
                    break;
                case ActivationFunctions.ReLU:
                    this.Activation = ReLU(fofx);
                    break;
                default:
                    this.Activation = SigmoidActivation(fofx);
                    break;
            }
        }

        /// <summary>
        /// Translates this object into a string that follows the format: Name (Total Weights: x)
        /// </summary>
        /// <returns></returns>
        public override string ToString() {
            return $"{this.Id} (Total Weights: {this.Weights.Length})";
        }

    }
}
