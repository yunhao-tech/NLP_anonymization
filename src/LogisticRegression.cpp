#include "LogisticRegression.h"

LogisticRegression::LogisticRegression(int nClasses, int maxIter, double eps=0.001, int miniBatch=100, double learning_rate=0.5)
{ // the default value to be discussed
    nClasses_ = nClasses;
    maxIter_ = maxIter;
    eps_ = eps;
    miniBatch_ = miniBatch;
    learning_rate_ = learning_rate;

}

LogisticRegression::~LogisticRegression(){}

void LogisticRegression::initWeights(int length){}

vector<vector<double> > LogisticRegression::loadFeatures(string fileName){}

vector<int> LogisticRegression::loadLabels(string fileName){}

void LogisticRegression::train(vector<vector<double> > features, vector<int> labels){}

int LogisticRegression::predict(vector<double> feature){}

double LogisticRegression::predict_proba(vector<double> feature){}

double LogisticRegression::eval(vector<vector<double> > features, vector<int> labels){}


int main(){ //test package Eigen
    Vector2d v1, v2;
    v1 << 5, 6;
    cout  << "v1 = " << endl << v1 << endl;
    v2 << 4, 5 ;
    Matrix2d result = v1*v2.transpose();
    cout << "result: " << endl << result << endl;
}
