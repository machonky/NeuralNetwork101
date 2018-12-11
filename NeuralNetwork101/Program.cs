using System.Collections.Generic;

namespace NeuralNetwork101
{
    class Program
    {
        static void Main(string[] args)
        {
            var topology = new Topology(3,3,1);
            var network = new NeuralNetwork(topology);

            var inputValues = new List<double>();
            network.FeedForward(inputValues);

            var targetValues = new List<double>();
            network.BackPropagate(targetValues);

            List<double> results = network.GetResults();
        }
    }
}
