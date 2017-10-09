using System;

namespace NeuralNetworkLibrary {

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
        /// Calculates the hyperbolic tangent <see cref="Math.Tanh(float)"/>
        /// of the given parameter
        /// </summary>
        public static float HyperbolicTangent(float x) {
            return (float)Math.Tanh(x);
        }

        /// <summary>
        /// Calculates the hyperbolic tangent <see cref="Math.Tanh(double)"/>
        /// of the given parameter
        /// </summary>
        public static float HyperbolicTangent(double x) {
            return (float)Math.Tanh(x);
        }

        /// <summary>
        /// Calculates the Softmax activation of a given vector
        /// </summary>
        /// <returns></returns>
        public static float[] Softmax(float[] x) {
            var sumx = 0f;
            float[] retval = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
                x[i] = (float)Math.Exp(x[i]);
            for (int i = 0; i < x.Length; i++)
                sumx += x[i];
            for (int i = 0; i < x.Length; i++)
                retval[i] = x[i] / sumx;
            return retval;
        }

        /// <summary>
        /// Returns the larger of two single-precision floating-point numbers
        /// </summary>
        public static float ReLU(float x) {
            return Math.Max(0, x);
        }

    }


}
