#include <iostream>
#include "layer.h"

#define LEARNING_RATE 0.3

Layer input(2, NULL, true); // 0 previous node neurons because no previous node
Layer h1(2, &input, false);
Layer h2(2, &h1, false);
Layer output(1, &h2, false);

int main() {
    double inputvalues[2] = {1, 1};

    input.activation_values = inputvalues;

    while(1) {
        h1.calcValues();
        h2.calcValues();
        output.calcValues();

        for(int i = 0; i < output.numNodes; i++) {
            std::cout << output.activation_values[i] << std::endl;
        }
    }

    return 0;
}