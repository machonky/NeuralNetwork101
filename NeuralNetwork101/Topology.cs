namespace NeuralNetwork101
{
    public class Topology
    {
        public Topology(params int[] layerNeuronCounts)
        {
            LayerNeuronCounts = layerNeuronCounts;
        }

        public int[] LayerNeuronCounts { get; }
    }
}
