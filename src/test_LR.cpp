#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <ctime>


#include "Dataset.hpp"
#include "LogistiqueRegression.hpp"
#include "ConfusionMatrix.hpp"

using std::cout;
using std::endl;

int main(int argc, const char* argv[]){
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " <train_data_file> <train_label_file> <test_data_file> <test_label_file> <lambda>" << endl;
        return 1;
    }
    cout << endl << "Reading training dataset and its labels " << argv[1] << " " << argv[2] <<" ..." << endl << endl;
    Dataset train_dataset(argv[1]);
    Dataset test_dataset(argv[3]);
    double lambda = std::stod(argv[5]);

    std::clock_t t = std::clock();
    //load training labels
    std::vector<int> train_labels;
    std::ifstream fin(argv[2]);
	if (fin.fail()) {
		std::cout<<"Cannot read from file "<< argv[2] <<" !"<<std::endl;
		exit(1);
	}
    std::string line; 
    while (getline(fin, line)){
        int val = std::stoi(line);
        train_labels.push_back(val);
    }
    fin.close();

    //load test labels
    std::vector<int> test_labels;
    std::ifstream fin_test(argv[4]);
	if (fin_test.fail()) {
		std::cout<<"Cannot read from file "<< argv[4] <<" !"<<std::endl;
		exit(1);
	}
    std::string line2; 
    while (getline(fin_test, line2)){
        int val = std::stoi(line2);
        test_labels.push_back(val);
    }
    fin_test.close();

    train_dataset.Show(false);  // only dimensions and samples
    assert((train_dataset.GetDim() == test_dataset.GetDim()));  // otherwise doesn't make sense


    LogistiqueRegression lr(&train_dataset, train_labels, lambda);

    // ConfusionMatrix
    ConfusionMatrix confusion_matrix;
	
    // Starts predicting
 	std::cout<< "Prediction and Confusion Matrix filling" <<std::endl;
    int dim = test_dataset.GetDim();
    for (int i=0; i < test_dataset.GetNbrSamples(); ++i) {
        std::vector<double> sample = test_dataset.GetInstance(i);
        Eigen::VectorXd query(dim);
        int true_label = test_labels[i];  // To not leave it uninitialized + will error in AddPrediction
        for (int j=0; j < dim; ++j){
            query(j) = sample[j];
        }
        int predicted_label = lr.Estimate(query);
        confusion_matrix.AddPrediction(true_label, predicted_label);
    }
    
    t = std::clock() - t;

    cout <<endl
         <<"execution time: "
         <<(t*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    confusion_matrix.PrintEvaluation();
    
	return 0;

}