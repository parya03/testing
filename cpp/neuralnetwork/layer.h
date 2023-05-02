/**
 * Layer class
*/

// typedef struct Wire {
//     double weight;
//     Node *start;
//     Node *end;
// } wire_t;


class Layer {
    public:
        Layer(int node_num, double *weight_array, double *bias_array); // To give weights and biases
        Layer(int node_num, Layer* prev, bool isInput); // Randomize weights and biases
        void calcValues(); // Calculates activation values based off previous node

        Layer *prev_layer;
        int numNodes;
        double *weights; // Array holds weights of connections from previous layer
        double *biases; // Biases of every node
        double *activation_values; // Values of every node after calculation
        double *prev_activation_values; // Values of previous layer's calculation
        bool isInput; // "Should this layer do it's own activation value calculations?"
    private:

};