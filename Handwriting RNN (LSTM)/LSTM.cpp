#include"LSTM.h"

LSTM::LSTM(int input_size, int hidden_size, int output_size, int num_epochs, float learning_rate ) {
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->num_epochs = num_epochs;
    this->learning_rate = learning_rate;
};

void LSTM::init(){
    this->wi = init_weights(input_size, hidden_size);
    this->bi = MatrixXd::Zero(hidden_size, 1);

    this->wf = init_weights(input_size, hidden_size);
    this->bf = MatrixXd::Zero(hidden_size, 1);

    this->wc = init_weights(input_size, hidden_size);
    this->bc = MatrixXd::Zero(hidden_size, 1);

    this->wo = init_weights(input_size, hidden_size);
    this->wco = init_weights(input_size, hidden_size);
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
    double scale = sqrt(6.0 / (input_size + output_size));
    weights *= scale;

    return weights;
}


MatrixXd LSTM::forward(MatrixXd x) {
    // Initialize the cell state and hidden state
    MatrixXd cell_state = MatrixXd::Zero(hidden_size, 1);
    MatrixXd hidden_state = MatrixXd::Zero(hidden_size, 1);

    // Initialize the big matrices
    concat_gates = MatrixXd::Zero(hidden_size * 4, x.cols());
    input_gates = MatrixXd::Zero(hidden_size, x.cols());
    forget_gates = MatrixXd::Zero(hidden_size, x.cols());
    candidate_gates = MatrixXd::Zero(hidden_size, x.cols());
    output_gates = MatrixXd::Zero(hidden_size, x.cols());
    cell_states = MatrixXd::Zero(hidden_size, x.cols());
    hidden_states = MatrixXd::Zero(hidden_size, x.cols());

    MatrixXd outputs = MatrixXd::Zero(output_size, x.cols());

    // Iterate through the sequence
    for (int p = 0; p < x.cols(); ++p) {
        // Concatenate the input vector and hidden state
        MatrixXd concat_input = MatrixXd::Zero(input_size + hidden_size, 1);
        concat_input << x.col(p), hidden_state;

        // Calculate the forget gate
        MatrixXd forget_gate = (wf * concat_input + bf).unaryExpr(&sigmoid);

        // Calculate the input gate
        MatrixXd input_gate = (wi * concat_input + bi).unaryExpr(&sigmoid);

        // Calculate the candidate cell state
        MatrixXd candidate_gate = (wc * concat_input + bc).unaryExpr(&tanh);

        // Calculate the cell state
        cell_state = forget_gate.cwiseProduct(cell_state) + input_gate.cwiseProduct(candidate_gate);

        // Calculate the output gate
        MatrixXd output_gate = (wo * concat_input + wco * cell_state + bo).unaryExpr(&sigmoid);

        // Calculate the hidden state
        hidden_state = output_gate.cwiseProduct(cell_state.unaryExpr(&tanh));

        // Add the values to the big matrices
        concat_gates.col(p) = concat_input;
        forget_gates.col(p) = forget_gate;
        input_gates.col(p) = input_gate;
        candidate_gates.col(p) = candidate_gate;
        output_gates.col(p) = output_gate;
        cell_states.col(p) = cell_state;
        hidden_states.col(p) = hidden_state;

        outputs.col(p) = (wy * hidden_state + by).unaryExpr(&sigmoid); 
    }

    // Calculate the output vector
    // Return the big matrices
    return outputs;
}

void  LSTM::backward(MatrixXd x, MatrixXd errors) {
    // Initialize the gradients for weights and biases
    MatrixXd d_wf = MatrixXd::Zero(hidden_size * 4, input_size + hidden_size);
    MatrixXd d_wi = MatrixXd::Zero(hidden_size * 4, input_size + hidden_size);
    MatrixXd d_wc = MatrixXd::Zero(hidden_size * 4, input_size + hidden_size);
    MatrixXd d_wo = MatrixXd::Zero(hidden_size * 4, input_size + hidden_size);
    MatrixXd d_wy = MatrixXd::Zero(output_size, hidden_size);
    MatrixXd d_wco = MatrixXd::Zero(hidden_size * 4, hidden_size);
    MatrixXd d_bf = MatrixXd::Zero(hidden_size * 4, 1);
    MatrixXd d_bi = MatrixXd::Zero(hidden_size * 4, 1);
    MatrixXd d_bc = MatrixXd::Zero(hidden_size * 4, 1);
    MatrixXd d_bo = MatrixXd::Zero(hidden_size * 4, 1);
    MatrixXd d_by = MatrixXd::Zero(output_size, 1);

    // Initialize the gradients for hidden state and cell state
    MatrixXd d_hidden_state = MatrixXd::Zero(hidden_size, 1);
    MatrixXd d_cell_state = MatrixXd::Zero(hidden_size, 1);

    // Iterate through the sequence in reverse order
    for (int p = x.cols() - 1; p >= 0; --p) {
        MatrixXd error = errors.row(p);
        // Calculate the gradient for the output layer
        MatrixXd d_output = error.cwiseProduct(hidden_states.row(p).transpose());
        d_wy += d_output;
        d_by += error;

        // Calculate the gradient for the hidden state
        MatrixXd d_hidden = wy.transpose() * error + d_hidden_state;

        // Calculate the gradient for the output gate
        MatrixXd d_output_gate = (d_hidden * cell_states.row(p)).unaryExpr(&tanh);
        d_output_gate = d_output_gate.cwiseProduct(output_gates.col(p).unaryExpr(&sigmoid_derivative));

        // Calculate the gradient for the cell state
        MatrixXd d_cell = wo.transpose() * d_output_gate + d_cell_state;
        d_cell += d_hidden.cwiseProduct(output_gates.col(p).unaryExpr(&tanh_derivative));
        d_cell = d_cell.cwiseProduct(forget_gates.col(p).unaryExpr(&sigmoid_derivative));

        // Calculate the gradient for the forget gate
        MatrixXd d_forget_gate = d_cell.cwiseProduct(cell_states.col(p - 1));
        d_forget_gate = d_forget_gate.cwiseProduct(forget_gates.col(p).unaryExpr(&sigmoid_derivative));

        // Calculate the gradient for the input gate
        MatrixXd d_input_gate = d_cell.cwiseProduct(candidate_gates.col(p));
        d_input_gate = d_input_gate.cwiseProduct(input_gates.col(p).unaryExpr(&sigmoid_derivative));

        // Calculate the gradient for the candidate gate
        MatrixXd d_candidate_gate = d_cell.cwiseProduct(input_gates.col(p));
        d_candidate_gate = d_candidate_gate.cwiseProduct(candidate_gates.col(p).unaryExpr(&tanh_derivative));

        // Calculate the gradient for the concatenated input
        MatrixXd d_concat_input = wf.transpose() * d_forget_gate +
                                  wi.transpose() * d_input_gate +
                                  wc.transpose() * d_candidate_gate +
                                  wo.transpose() * d_output_gate;

        // Update the gradients for weights and biases
        //TODO : change this
        d_wf += d_forget_gate * d_concat_input.col(p).transpose();
        d_wi += d_input_gate * d_concat_input.col(p).transpose();
        d_wc += d_candidate_gate * d_concat_input.col(p).transpose();
        d_wo += d_output_gate * d_concat_input.col(p).transpose();
        d_wco += d_output_gate * d_concat_input.col(p).transpose();
        d_bf += d_forget_gate;
        d_bi += d_input_gate;
        d_bc += d_candidate_gate;
        d_bo += d_output_gate;

        // Update the gradients for hidden state and cell state
        d_hidden_state = d_concat_input.bottomRows(hidden_size);
        d_cell_state = forget_gates.col(p).cwiseProduct(d_cell);
    }

    // Return the gradients for weights and biases
   // return std::make_tuple(d_wf, d_wi, d_wc, d_wo, d_wco, d_bf, d_bi, d_bc, d_bo, d_by);
}

// MatrixXd LSTM :: forward(MatrixXd x) {
//     // Initialize the cell state and hidden state
//     MatrixXd cell_state = MatrixXd::Zero(hidden_size, 1);
//     MatrixXd hidden_state = MatrixXd::Zero(hidden_size, 1);

//     // Iterate through the sequence
//     for (int p = 0; p < x.cols(); ++p) {
//         // Concatenate the input vector and hidden state
//         MatrixXd concat_input = MatrixXd::Zero(input_size + hidden_size, 1);
//         concat_input << x.col(p), hidden_state;

//         // Calculate the forget gate
//         MatrixXd forget_gate = (wf * concat_input + bf).unaryExpr(&sigmoid);

//         // Calculate the input gate
//         MatrixXd input_gate = (wi * concat_input + bi).unaryExpr(&sigmoid);

//         // Calculate the candidate cell state
//         MatrixXd candidate_gate = (wc * concat_input + bc).unaryExpr(&tanh);

//         // Calculate the cell state
//         cell_state = forget_gate.cwiseProduct(cell_state) + input_gate.cwiseProduct(candidate_gate);

//         // Calculate the output gate
//         MatrixXd output_gate = (wo * concat_input + wco * cell_state  + bo).unaryExpr(&sigmoid);

//         // Calculate the hidden state
//         hidden_state = output_gate.cwiseProduct(cell_state.unaryExpr(&tanh));
//     }

//     // Calculate the output vector
//     MatrixXd outputs = (wy * hidden_state + by).unaryExpr(&sigmoid);

//     return outputs;
// }

void LSTM::test(){
    cout<<"testing"<<endl;
    cout<<init_weights(2,2)<<endl;
}
