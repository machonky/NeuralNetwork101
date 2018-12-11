using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork101
{
    public class NeuralNetwork
    {
        public NeuralNetwork(Topology topology)
        {
            foreach(var count in topology.LayerNeuronCounts)
            {
                var newLayer = new Layer();
                _layers.Add(newLayer);

                for (int neuronIndex = 0; neuronIndex <= count; ++neuronIndex)
                {
                    newLayer.Add(new Neuron(neuronIndex, count));
                }

                newLayer.Last().OutputValue = 1.0; // Bias neuron set to 1.0;
            }
        }

        public void FeedForward(List<double> inputValues)
        {
            if (inputValues.Count != _layers[0].Count - 1)
            {
                throw new ArgumentException("Mismatched input value sizes");
            }

            for (int i = 0; i < inputValues.Count; ++i)
            {
                _layers[0][i].OutputValue = inputValues[i];
            }

            for (int layerIndex = 1; layerIndex < _layers.Count; ++layerIndex)
            {
                Layer prevLayer = _layers[layerIndex - 1];
                for (int neuronIndex = 0; neuronIndex < _layers[layerIndex].Count - 1; ++neuronIndex)
                {
                    _layers[layerIndex][neuronIndex].FeedForward(prevLayer);
                }
            }
        }

        public event EventHandler<double> ErrorCalculated;

        public void BackPropagate(List<double> targetValues)
        {
            // Calculate overall net error
            Layer outputLayer = _layers.Last();
            Error = CalculateRmsError(targetValues, outputLayer);
            // Let observers know of an update
            ErrorCalculated?.Invoke(this, Error);

            CalcOutputLayerGradients(targetValues, outputLayer);
            CalcHiddenLayerGradients();
            UpdateConnectionWeights();
        }

        private void UpdateConnectionWeights()
        {
            // For all layers from output to first hidden layer - update connection weights.
            for (int layerIndex = _layers.Count - 1; layerIndex > 0; --layerIndex)
            {
                Layer layer = _layers[layerIndex];
                Layer prevLayer = _layers[layerIndex - 1];
                for (int neuronIndex = 0; neuronIndex < layer.Count - 1; ++neuronIndex)
                {
                    layer[neuronIndex].UpdateConnectionWeights(prevLayer);
                }
            }
        }

        private void CalcHiddenLayerGradients()
        {
            for (int layerIndex = _layers.Count - 2; layerIndex > 0; --layerIndex)
            {
                Layer hiddenLayer = _layers[layerIndex];
                Layer nextLayer = _layers[layerIndex + 1];
                for (int neuronIndex = 0; neuronIndex < hiddenLayer.Count; ++neuronIndex)
                {
                    hiddenLayer[neuronIndex].CalcHiddenGradients(nextLayer);
                }
            }
        }

        private static void CalcOutputLayerGradients(List<double> targetValues, Layer outputLayer)
        {
            for (int neuronIndex = 0; neuronIndex < outputLayer.Count - 1; ++neuronIndex)
            {
                outputLayer[neuronIndex].CalcOutputGradients(targetValues[neuronIndex]);
            }
        }

        private static double CalculateRmsError(List<double> targetValues, Layer outputLayer)
        {
            double error = 0.0;
            for (int neuronIndex = 0; neuronIndex < outputLayer.Count - 1; ++neuronIndex)
            {
                double delta = targetValues[neuronIndex] - outputLayer[neuronIndex].OutputValue;
                error += delta * delta;
            }
            error /= outputLayer.Count - 1;
            return Math.Sqrt(error);
        }

        public List<double> GetResults()
        {
            var result = new List<double>();
            return result;
        }

        private readonly List<Layer> _layers = new List<Layer>();

        public double Error { get; private set; }
    }
}
