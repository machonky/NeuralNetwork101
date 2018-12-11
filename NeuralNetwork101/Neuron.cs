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
        public double OutputValue { get; set; }

        public List<Connection> Connections { get { return _connections; } }
        private readonly List<Connection> _connections = new List<Connection>();

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

        public void CalcOutputGradients(double v)
        {
        }

        public void CalcHiddenGradients(Layer nextLayer)
        {
        }

        internal void UpdateConnectionWeights(Layer prevLayer)
        {
            throw new NotImplementedException();
        }
    }
}
