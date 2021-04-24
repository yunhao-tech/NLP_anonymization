#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
#include "../eigen-3.4-rc1/Eigen/Dense"

using namespace std;
using namespace Eigen;

class LogisticRegression
{
    public:
        LogisticRegression(int nClasses, int maxIter, double eps, int miniBatch, double learning_rates);
        virtual ~LogisticRegression();
        void initWeights(int length); //initialize the weights_ of network, the parameter length is that of one feature
        vector<vector<double> > loadFeatures(string fileName); // load the word embeddings
        vector<int> loadLabels(string fileName); //load the labels
        void train(vector<vector<double> > features, vector<int> labels); //train the model using gradient descent
        int predict(vector<double> feature); //give the classifier result
        double predict_proba(vector<double> feature); //give the predicted probability for class 1
        double eval(vector<vector<double> > features, vector<int> labels); //evaluate the model's accuracy

    private:
        int nClasses_; //number of classes to classifier
        int maxIter_; //maximum number of iterations
        double eps_; //the stop criterion
        int miniBatch_; //size of mini batch data
        double learning_rate_; //parameter alpha in gradient descent
        vector<double> weights_; //the weights in logistic regression unit

};

#endif // LOGISTICREGRESSION_H
