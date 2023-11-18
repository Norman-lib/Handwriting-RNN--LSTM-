#pragma once
#include<iostream>
#include<Eigen/Dense>
#include <random> 

using namespace std;
using namespace Eigen;

class LSTM {
public:
	LSTM(int input_size, int hidden_size, int output_size, int num_epochs, float learning_rate );
	void init();
	void test();
private:
	#pragma region Private variables
	int input_size, hidden_size, output_size, num_epochs, learning_rate;
	MatrixXd wf, bf, wi, bi, wc, bc, wo, bo, wy, by;

	#pragma endregion

	#pragma region Private functions
	MatrixXd init_weights(int input_size, int output_size);
	double sigmoid(double x);
	#pragma endregion
};