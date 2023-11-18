#include"LSTM.h"

LSTM::LSTM(int input_size, int hidden_size, int output_size, int num_epochs, float learning_rate ) {
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->num_epochs = num_epochs;
    this->learning_rate = learning_rate;
};

// self.wf = initWeights(input_size, hidden_size)
//         self.bf = np.zeros((hidden_size, 1))

//         # Input Gate
//         self.wi = initWeights(input_size, hidden_size)
//         self.bi = np.zeros((hidden_size, 1))

//         # Candidate Gate
//         self.wc = initWeights(input_size, hidden_size)
//         self.bc = np.zeros((hidden_size, 1))

//         # Output Gate
//         self.wo = initWeights(input_size, hidden_size)
//         self.bo = np.zeros((hidden_size, 1))

//         # Final Gate
//         self.wy = initWeights(hidden_size, output_size)
//         self.by = np.zeros((output_size, 1))

void LSTM::init(){
    this->wi = init_weights(input_size, hidden_size);
    this->bi = MatrixXd::Zero(hidden_size, 1);

    this->wf = init_weights(input_size, hidden_size);
    this->bf = MatrixXd::Zero(hidden_size, 1);

    this->wc = init_weights(input_size, hidden_size);
    this->bc = MatrixXd::Zero(hidden_size, 1);

    this->wo = init_weights(input_size, hidden_size);
    this->bo = MatrixXd::Zero(hidden_size, 1);

    this->wy = init_weights(hidden_size, output_size);
    this->by = MatrixXd::Zero(output_size, 1);
}

MatrixXd LSTM:: init_weights(int input_size, int output_size) {
    // Seed the random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    // Initialize the weight matrix
    Eigen::MatrixXd weights(output_size, input_size);

    // Fill the weight matrix with random values in the specified range
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights(i, j) = dis(gen);
        }
    }

    // Scale the weights by sqrt(6 / (input_size + output_size))
    double scale = std::sqrt(6.0 / (input_size + output_size));
    weights *= scale;

    return weights;
}

double LSTM::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void LSTM::test(){
    cout<<"testing"<<endl;
    cout<<init_weights(2,2)<<endl;
}
