#pragma once
#include<iostream>
#include<Eigen/Dense>
#include <random> 
#include <cmath>
#include"utils.h"

using namespace std;
using namespace Eigen;

class LSTM {
public:
	LSTM(int input_size, int hidden_size, int output_size, int num_epochs, float learning_rate );
	MatrixXd forward(MatrixXd x);
	void backward(MatrixXd inputs, MatrixXd errors);
	void init();
	void test();

private:
	#pragma region Private variables
	int input_size, hidden_size, output_size, num_epochs, learning_rate;
	MatrixXd wf, bf, wi, bi, wc, wco, bc, wo, bo, wy, by;
	MatrixXd activation_outputs, candidate_gates, output_gates, forget_gates, input_gates, cell_states, hidden_states, concat_inputs, concat_gates;

	#pragma endregion

	#pragma region Private functions
	MatrixXd init_weights(int input_size, int output_size);
	double sigmoid(double x);
	double tanh(double x);
	#pragma endregion
};