using System;
using System.Collections.Generic;

namespace NeuralNetwork101
{
    public class Neuron
    {
        public Neuron(int index, int numberOutputs)
        {
            Index = index;
            var rnd = new Random();
            for (int i = 0; i < numberOutputs; ++i)
            {
                _connections.Add(new Connection { Weight = rnd.NextDouble() });
            }
        }
        public int Index { get; }

        public double OutputValue { get; private set; }

        public double Gradient { get; private set; }

        public List<Connection> Connections { get { return _connections; } }

        public void FeedForward(Layer prevLayer)
        {
            double sum = 0.0;
            for (int neuronIndex = 0; neuronIndex < prevLayer.Count; ++neuronIndex)
            {
                sum += prevLayer[neuronIndex].OutputValue * 
                       prevLayer[neuronIndex]._connections[Index].Weight;
            }

            OutputValue = TransferFunction(sum);
        }

        private static double TransferFunction(double value)
        {
            return Math.Tanh(value);
        }

        private static double TransferFunctionDerivative(double value)
        {
            return 1 - Math.Pow(value, 2.0); // Approximated by Taylor Expansion
        }

        public void CalcOutputGradients(double targetValue)
        {
            double delta = targetValue - OutputValue;
            Gradient = delta * TransferFunctionDerivative(OutputValue);
        }

        public void CalcHiddenGradients(Layer nextLayer)
        {
            double delta = sumDeltaOfWeights(nextLayer);
            Gradient = delta * TransferFunctionDerivative(OutputValue);
        }

        private double sumDeltaOfWeights(Layer nextLayer)
        {
        }

        public void UpdateInputWeights(Layer prevLayer)
        {

        }

        private readonly List<Connection> _connections = new List<Connection>();
    }
}
