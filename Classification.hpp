#include "Dataset.hpp"
#include "eigen-3.4-rc1/Eigen/Dense"

#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

/** 
	The Classification class is an abstract class that will be the basis of the KnnClassification classe.
*/
class Classification{
protected:
    /**
      The pointer to a dataset.
    */
	Dataset* m_dataset;
    /**
     * a vector storing the labels
    */
  std::vector<int> m_labels;
public:
    /**
      The constructor sets private attributes dataset (as a pointer) and the column to do classification on (as an int).
    */
	Classification(Dataset* dataset, std::vector<int> labels);
    /**
      The dataset getter.
    */
	Dataset* getDataset();

    /**
      The Estimate method is virtual: it (could) depend(s) on the Classification model(s) implemented (here we use only the KnnClassification class).
    */
	virtual int Estimate(const Eigen::VectorXd & x , double threshold=0.5) const = 0;
};

#endif //CLASSIFICATION_HPP
