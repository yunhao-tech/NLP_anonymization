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


void load_labels(std::string fileName, std::vector<int> &labels){
    std::ifstream fin(fileName);
	if (fin.fail()) {
		std::cout<<"Cannot read from file "<< fileName << "!" <<std::endl;
		exit(1);
	}
    std::string line; 
    while (getline(fin, line)){
        int val = std::stoi(line);
        labels.push_back(val);
    }
    fin.close();
}

int main(int argc, const char* argv[]){
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <train_data_file> <test_data_file> <lambda>" << std::endl;
        return 1;
    }
    std::cout << std::endl << "Reading training dataset and test dataset" << argv[1] << " " << argv[2] <<" ..." << std::endl << std::endl;
    Dataset train_dataset(argv[1]);
    Dataset test_dataset(argv[2]);
    double lambda = std::stod(argv[3]);

    std::clock_t t_start = std::clock();
    //load training labels
    std::vector<int> train_labels_O_vs_rest;
    load_labels("multinomial_datasets/train_labels_O_vs_rest.csv", train_labels_O_vs_rest);
    std::vector<int> train_labels_PER_vs_rest; 
    load_labels("multinomial_datasets/train_labels_PER_vs_rest.csv", train_labels_PER_vs_rest);
    std::vector<int> train_labels_MISC_vs_rest;
    load_labels("multinomial_datasets/train_labels_MISC_vs_rest.csv", train_labels_MISC_vs_rest);
    std::vector<int> train_labels_LOC_vs_rest;
    load_labels("multinomial_datasets/train_labels_LOC_vs_rest.csv", train_labels_LOC_vs_rest);
    std::vector<int> train_labels_ORG_vs_rest;
    load_labels("multinomial_datasets/train_labels_ORG_vs_rest.csv", train_labels_ORG_vs_rest);

    //load test labels
    std::vector<int> test_labels_O_vs_rest;
    load_labels("multinomial_datasets/testa_labels_O_vs_rest.csv", test_labels_O_vs_rest);
    std::vector<int> test_labels_PER_vs_rest;
    load_labels("multinomial_datasets/testa_labels_PER_vs_rest.csv", test_labels_PER_vs_rest);
    std::vector<int> test_labels_MISC_vs_rest;
    load_labels("multinomial_datasets/testa_labels_MISC_vs_rest.csv", test_labels_MISC_vs_rest);
    std::vector<int> test_labels_LOC_vs_rest;
    load_labels("multinomial_datasets/testa_labels_LOC_vs_rest.csv", test_labels_LOC_vs_rest);
    std::vector<int> test_labels_ORG_vs_rest;
    load_labels("multinomial_datasets/testa_labels_ORG_vs_rest.csv", test_labels_ORG_vs_rest);
    std::vector<int> test_true_labels;
    load_labels("multinomial_datasets/testa_true_labels.csv", test_true_labels);

    train_dataset.Show(false);  // only dimensions and samples
    assert((train_dataset.GetDim() == test_dataset.GetDim()));  // otherwise doesn't make sense

    LogistiqueRegression lr_O_vs_rest(&train_dataset, train_labels_O_vs_rest, lambda);
    LogistiqueRegression lr_PER_vs_rest(&train_dataset, train_labels_PER_vs_rest, lambda);
    LogistiqueRegression lr_MISC_vs_rest(&train_dataset, train_labels_MISC_vs_rest, lambda);
    LogistiqueRegression lr_LOC_vs_rest(&train_dataset, train_labels_LOC_vs_rest, lambda);
    LogistiqueRegression lr_ORG_vs_rest(&train_dataset, train_labels_ORG_vs_rest, lambda);
	
    // ConfusionMatrix
    ConfusionMatrix confusion_matrix_O;
    ConfusionMatrix confusion_matrix_PER;
    ConfusionMatrix confusion_matrix_MISC;
    ConfusionMatrix confusion_matrix_LOC;
    ConfusionMatrix confusion_matrix_ORG;


    /*label "O" -> 0
    //label "PER" -> 1
    //label "MISC" -> 2
    //label "LOC" -> 3
    //label "ORG" -> 4
    */
    // Starts predicting
 	std::cout<< "Starting prediction" <<std::endl;
    int dim = test_dataset.GetDim();

    int num_err = 0;
    int n = test_dataset.GetNbrSamples();
    for (int i=0; i < n; ++i) {
        std::vector<double> xi = test_dataset.GetInstance(i);
        Eigen::VectorXd xi_ = Eigen::VectorXd::Map(xi.data(), xi.size());
        
        int true_label_O = test_labels_O_vs_rest[i];
        int predicted_label_O = lr_O_vs_rest.Estimate(xi_);
        confusion_matrix_O.AddPrediction(true_label_O, predicted_label_O);
        
        int true_label_PER = test_labels_PER_vs_rest[i];
        int predicted_label_PER = lr_PER_vs_rest.Estimate(xi_);
        confusion_matrix_PER.AddPrediction(true_label_PER, predicted_label_PER);
        
        int true_label_MISC = test_labels_MISC_vs_rest[i];
        int predicted_label_MISC = lr_MISC_vs_rest.Estimate(xi_);
        confusion_matrix_MISC.AddPrediction(true_label_MISC, predicted_label_MISC);

        int true_label_LOC = test_labels_LOC_vs_rest[i];
        int predicted_label_LOC = lr_LOC_vs_rest.Estimate(xi_);
        confusion_matrix_LOC.AddPrediction(true_label_LOC, predicted_label_LOC);

        int true_label_ORG = test_labels_ORG_vs_rest[i];
        int predicted_label_ORG = lr_ORG_vs_rest.Estimate(xi_);
        confusion_matrix_ORG.AddPrediction(true_label_ORG, predicted_label_ORG);

        //calculate the delta for each binary classifier
        
        std::vector<double> delta; 
        double delta_O = lr_O_vs_rest.Estimate_double(xi_);
        double delta_PER = lr_PER_vs_rest.Estimate_double(xi_);
        double delta_MISC = lr_MISC_vs_rest.Estimate_double(xi_);
        double delta_LOC = lr_LOC_vs_rest.Estimate_double(xi_);
        double delta_ORG = lr_ORG_vs_rest.Estimate_double(xi_);
        delta.push_back(delta_O);
        delta.push_back(delta_PER);
        delta.push_back(delta_MISC);
        delta.push_back(delta_LOC);
        delta.push_back(delta_ORG);

        //calculate the maximum delta and its index
        double  delta_max = 0.0;
        int index = 0;
        for (int j=0; j<5; j++){
            if (delta[j] > delta_max){
                delta_max = delta[j];
                index = j;
            }
        }
        if (index != test_true_labels[i]){
            num_err++;
        }
    }
    std::clock_t t_end = std::clock();
    std::cout << std::endl
         <<"CPU time used: "
         <<1000 * (t_end - t_start)/CLOCKS_PER_SEC
         <<"ms\n\n";

    double precision_total = 1 - (double)num_err / n;
    double f_score_macro = (confusion_matrix_O.f_score() + confusion_matrix_PER.f_score() 
                            + confusion_matrix_MISC.f_score() + confusion_matrix_LOC.f_score() + confusion_matrix_ORG.f_score()) / 5;
    double precision_mean = (confusion_matrix_O.precision() + confusion_matrix_PER.precision() 
                            + confusion_matrix_MISC.precision() + confusion_matrix_LOC.precision() + confusion_matrix_ORG.precision()) / 5;
    double detection_rate_mean = (confusion_matrix_O.detection_rate() + confusion_matrix_PER.detection_rate() 
                            + confusion_matrix_MISC.detection_rate() + confusion_matrix_LOC.detection_rate() + confusion_matrix_ORG.detection_rate()) / 5;
    double f_score_micro = 2 * precision_mean * detection_rate_mean / (precision_mean + detection_rate_mean);

    confusion_matrix_O.PrintEvaluation();
    confusion_matrix_PER.PrintEvaluation();
    confusion_matrix_MISC.PrintEvaluation();
    confusion_matrix_LOC.PrintEvaluation();
    confusion_matrix_ORG.PrintEvaluation();


    std::cout << std::endl
        << "The total precision of multinomial logistic regression is "
        << precision_total << std::endl
        << "The macro f score is "
        << f_score_macro << std::endl
        << "The micro f score is "
        << f_score_micro << std::endl;
    
	return 0;

}