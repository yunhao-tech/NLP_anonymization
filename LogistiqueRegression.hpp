#include "eigen-3.4-rc1/Eigen/Dense"
#include "Dataset.hpp"
#include "Classification.hpp"

#ifndef LOGISTIQUEREGRESSION_HPP
#define LOGISTIQUEREGRESSION_HPP
/**
  The LogistiqueRegression class inherits from the Classification class, stores the coefficient and provides a bunch of specific methods.
*/
class LogistiqueRegression : public Classification {
private:
    /**
      The LogistiqueRegression coefficient.
    */
	Eigen::VectorXd* m_beta;
  
public:
    /**
      The linear regression method fits a linear regression coefficient to col_regr using the provided Dataset. It calls setCoefficients under the hood.
     @param dataset a pointer to a dataset
     @param m_col_regr the integer of the column index of Y
    */
	LogistiqueRegression(Dataset* dataset, std::vector<int> labels, double lambda);
    /**
      The destructor (frees m_beta).
    */
    ~LogistiqueRegression();
    /**
      The setter method of the private attribute m_beta which is called by LinearRegression.
    */
	void SetCoefficients(double lambda);
  /**
   *  calculation of descent in one iteration
   */
  Eigen::VectorXd Descent(Eigen::VectorXd & beta, double lambda);
    /**
      The getter method of the private attribute m_beta.
    */
	const Eigen::VectorXd* GetCoefficients() const;
    /**
      The estimate method outputs the predicted Y for a given point x.
     @param x the point for which to estimate Y.
    */
	int Estimate(const Eigen::VectorXd & x, double threshold=0.5) const;
  double Estimate_double( const Eigen::VectorXd & x) const;
  
};

#endif 