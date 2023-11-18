#include<LSTM.h>

int main(){
    cout<<"hello"<<endl;
    Matrix<float, 2,2> mat;
    mat.setZero();
    cout<< mat;

    LSTM lstm(1,1,1,1,1);
    lstm.init();
    lstm.test();
    return 0;
}

