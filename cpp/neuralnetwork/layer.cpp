/**
 * Layer class
*/

#include "layer.h"
#include <iostream>
#include <ctime>
#include <cstdlib>

/**
 * Some math functions:
*/

// ReLU activation function
inline double relu(double x) {
    return x > 0 ? x : 0;
}
// Find ReLU of whole matrix
void reluMatrix(double a[], double b[], uint32_t a_b_size) {
    for(uint32_t i = 0; i < a_b_size; i++) {
        b[i] = relu(a[i]);
    }
}
// Multiply 2 matrices
// TODO: OpenCL
void multiplyMatrices(
        double a[],
        const uint32_t a_rows,
        const uint32_t a_cols,
        double b[],
        const uint32_t b_cols,
        double c[]) {
    // https://en.wikipedia.org/wiki/Row-_and_column-major_order

    int a_size = a_rows * a_cols;
    
    int c_index = 0;

    // Debugging printfs
    // for(int i = 0; i < a_rows*a_cols; ++i) {
    //     printf("%lf ", a[i]);
    //     if(i % a_cols == 0)
    //         printf("\n");
    // }
    // for(int i = 0; i < a_rows*b_cols; ++i) {
    //     printf("%lf ", b[i]);
    //     if(i % b_cols == 0)
    //         printf("\n");
    // }
    // printf("\n");
    // printf("-------------------------------------\n");
    
    for(int a_current = 0; a_current < a_size; a_current += a_cols) { //Iterate through A row by row
        //For each row, start iterating through columns in B
        for(int b_current = 0; b_current < b_cols; ++b_current) {
            double buffer = 0;
            // Iterate through numbers to multiply together
            for(int i = 0; i < a_cols; ++i) {
                buffer += (a[i + a_current] * b[(i*b_cols) + b_current]);
            }
            c[c_index] = buffer;

            //printf("%d - %lf ", c_index, buffer);
            
            ++c_index;
        }
        //printf("\n");
    }

    // Debugging printfs
    // for(int i = 0; i < a_rows*b_cols; ++i) {
    //     printf("%lf ", c[i]);
    //     if(i % b_cols == 0)
    //         printf("\n");
    // }
    // printf("\n");
}
// Add matrices
void addMatrices(double a[], double b[], uint32_t a_b_size, double c[]) {
    for(uint32_t i = 0; i < a_b_size; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Layer class
*/

Layer::Layer(int node_num, double *weight_array, double *bias_array) {
    this->numNodes = node_num;
    this->weights = weight_array;
    this->biases = bias_array;
    this->activation_values = new double[node_num];
}

Layer::Layer(int node_num, Layer *prev, bool isInput) {

    // Allocate arrays and set variables
    this->prev_layer = prev;
    this->numNodes = node_num;
    this->weights = isInput ? NULL : new double[node_num * (prev->numNodes)];
    this->biases = new double[node_num];
    this->activation_values = new double[node_num];
    this->prev_activation_values = (isInput ? NULL : prev->activation_values);
    this->isInput = isInput;

    // Randomize weights and biases
    std::srand(time(0));
    if(!isInput) {
        for(int i = 0; i < (node_num * (prev->numNodes)); i++) {
            weights[i] = ((double)std::rand())/RAND_MAX;
        }
        for(int i = 0; i < (node_num); i++) {
            biases[i] = ((double)std::rand())/RAND_MAX;
        }
    }
}

// Calculate the neuron values of this layer
void Layer::calcValues() {
    if(this->isInput) return; // Can't do calculation if this is input node (you set the value, no previous nodes)
    // Multiply the weights matrix with previous layer's activation values
    multiplyMatrices(this->weights, this->prev_layer->numNodes, this->numNodes, prev_activation_values, 1, this->activation_values);

    // Add the biases of each node
    addMatrices(this->activation_values, this->biases, this->numNodes, this->activation_values);

    // Take the ReLU of each value
    reluMatrix(this->activation_values, this->activation_values, this->numNodes);
}