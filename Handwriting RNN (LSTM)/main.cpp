#include<LSTM.h>

int main(){
    cout<<"hello"<<endl;
    Matrix<float, 2,2> mat;
    mat.setZero();
    cout<< mat;

   // LSTM lstm(1,1,1,1,1);
    //lstm.init();
   // lstm.test();

    int hidden_size = 10;
    int input_size = 5;

    MatrixXd hidden_state = MatrixXd::Zero(hidden_size, 1);

    MatrixXd x = MatrixXd::Ones(input_size, 1);

    MatrixXd concat_input = MatrixXd::Zero(input_size + hidden_size, 1);
    concat_input << x, hidden_state;
    cout << concat_input << endl;
    return 0;
}

