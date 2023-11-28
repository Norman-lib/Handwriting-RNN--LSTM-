
#ifndef UTILS_H
#define UTILS_H

#include <cmath>

// Hyperbolic tangent function
double tanh(double x) {
    return std::tanh(x);
}

// Derivative of hyperbolic tangent function
double tanh_derivative(double x) {
    double tanhX = std::tanh(x);
    return 1 - tanhX * tanhX;
}

// Sigmoid function
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x) {
    double sigmoidX = sigmoid(x);
    return sigmoidX * (1 - sigmoidX);
}

#endif // UTILS_H
