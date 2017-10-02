using System;

namespace DANNeCognitiveLibrary {

    /// <summary>
    /// A collection of activation functions used as utility or tool in this library
    /// </summary>
    public static class ActivationFunctionUtilities {

        /// <summary>
        /// Calculates the Sigmoid of x, used to activate hidden layers
        /// </summary>
        public static float SigmoidActivation(float x) {
            return (float)(1 / (1 + Math.Exp(-x)));
        }

        /// <summary>
        /// Getting the derivative of X, used to activate neurons
        /// </summary>
        public static float SigmoidDerivative(float x) {
            return (float)(x * (1 - x));
        }

        /// <summary>
        /// Calculates the hyperbolic tangent <see cref="Math.Tanh(double)"/>
        /// of the given parameter
        /// </summary>
        public static float HyperbolicTangent(float x) {
            return (float)Math.Tanh(x);
        }

        /// <summary>
        /// Calculates the Sigmoid of x, used to activate hidden layers
        /// </summary>
        public static float SigmoidActivation(double x) {
            return (float)(1 / (1 + Math.Exp(-x)));
        }

        /// <summary>
        /// Getting the derivative of X, used to activate neurons
        /// </summary>
        public static float SigmoidDerivative(double x) {
            return (float)(x * (1 - x));
        }

        /// <summary>
        /// Calculates the hyperbolic tangent <see cref="Math.Tanh(double)"/>
        /// of the given parameter
        /// </summary>
        public static float HyperbolicTangent(double x) {
            return (float)Math.Tanh(x);
        }

    }
}
