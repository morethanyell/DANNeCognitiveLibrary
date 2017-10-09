using System;

namespace NeuralNetworkLibrary {

    /// <summary>
    /// A collection of exceptions or error handlers in this Neural Network Library
    /// </summary>
    public sealed class NeuralNetExceptions {

        /// <summary>
        /// Represents an error describing that the network is building a 
        /// layer with not enough hidden neurons. (There must be at least two or more Hidden Neurons per layer).
        /// </summary>
        [Serializable]
        public class NotEnoughHiddenNeuronsException : Exception {

            /// <summary>
            /// Error message
            /// </summary>
            public string _message { get; private set; }

            /// <summary>
            /// Overriding the message of Exception class
            /// </summary>
            public NotEnoughHiddenNeuronsException() { }

            /// <summary>
            /// Creates an instance of NotEnoughHiddenNeuronsException
            /// </summary>
            public NotEnoughHiddenNeuronsException(string message) { this._message = message; }

            /// <summary>
            /// Overriding the message of Exception class
            /// </summary>
            public override string Message => this._message;
        }

        /// <summary>
        /// Represents an error describing that the network has not enough training data.
        /// </summary>
        [Serializable]
        public class NotEnoughTrainingDataException : Exception {

            /// <summary>
            /// Error message
            /// </summary>
            public string _message { get; private set; }

            /// <summary>
            /// Creates an instance of NotEnoughTrainingDataException
            /// </summary>

            public NotEnoughTrainingDataException() { }

            /// <summary>
            /// Creates an instance of NotEnoughTrainingDataException
            /// </summary>
            /// <param name="message">Custom message</param>
            public NotEnoughTrainingDataException(string message) { this._message = message; }

            /// <summary>
            /// Overriding the message of Exception class
            /// </summary>
            public override string Message => this._message;
        }

        /// <summary>
        /// Represents an error describing that the network is building a vector of output that is invalid.
        /// </summary>
        /// <example>A DataTable is set as the expected output value of a feed forward neural network
        /// however such table contains zero (0) or more than one rows. An output must be a vector or a 1xN
        /// DataTable, where N is the number of columns.</example>
        [Serializable]
        public class InvalidOutputDataForTrainingException : Exception {

            /// <summary>
            /// Error message
            /// </summary>
            public string _message { get; private set; }

            /// <summary>
            /// Creates an instance of InvalidOutputDataForTrainingException
            /// </summary>

            public InvalidOutputDataForTrainingException() { }

            /// <summary>
            /// Creates an instance of InvalidOutputDataForTrainingException
            /// </summary>
            /// <param name="message">Custom message</param>
            public InvalidOutputDataForTrainingException(string message) { this._message = message; }

            /// <summary>
            /// Overriding the message of Exception class
            /// </summary>
            public override string Message => this._message;
        }

    }


}
