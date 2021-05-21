#include "LogistiqueRegression.hpp"
#include "Dataset.hpp"
#include "Classification.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

const double defaultEps = 8;

LogistiqueRegression::LogistiqueRegression( Dataset* dataset, std::vector<int> labels, double lambda) 
: Classification(dataset, labels) {
	SetCoefficients(lambda);
}

LogistiqueRegression::~LogistiqueRegression() {
	m_beta->resize(0);
	delete m_beta;
}

double Sigmoid(double t) {
    return 1 / (1 + exp(-t));
}

Eigen::VectorXd LogistiqueRegression::Descent(Eigen::VectorXd & beta, double lambda){
	int d = m_dataset->GetDim();
	int n = m_dataset->GetNbrSamples();

	//dérivée de beta
	Eigen::VectorXd derive(d+1);
	//matrice hessienne de beta
	Eigen::MatrixXd hessian(d+1,d+1);

	for (int i=0; i<n; i++){
		//std::vector to Eigen::VectorXd 
		std::vector<double > xi = m_dataset->GetInstance(i);
  		Eigen::VectorXd xi_ = Eigen::VectorXd::Map(xi.data(), xi.size());
		
		Eigen::VectorXd Xi(d+1);
		Xi[0]=1;
		Xi.tail(d) = xi_;
		if( m_labels[i] == 0 ){
			derive += -Sigmoid(Xi.transpose()*beta) * Xi;
		}
		else{
			derive += (1-Sigmoid(Xi.transpose()*beta)) * Xi;
		}
		hessian -= Sigmoid(Xi.transpose()*beta)*(1-Sigmoid(Xi.transpose()*beta))*Xi*Xi.transpose();
	}

	//Ici on utilise regularized logistic regression avec le paramètre lambda
	derive -= 2*lambda*beta;

	Eigen::MatrixXd I(d+1,d+1);
	I.setIdentity();
	hessian -= 2*lambda*I;
	return (-hessian).llt().solve(derive);
}


void LogistiqueRegression::SetCoefficients(double lambda) {
 	int d = m_dataset->GetDim();
 	//int n = m_dataset->GetNbrSamples();

 	//Initialisation de beta
 	Eigen::VectorXd beta(d+1);
 	beta.setOnes();

 	//Répéter jusqu'à la convergence de beta, i.e. beta=beta_next
 	Eigen::VectorXd beta_next(d+1);
 	beta_next = beta + Descent(beta,lambda);
 	while((beta_next - beta).norm() > defaultEps){
  		beta = beta_next;
  		beta_next = beta + Descent(beta,lambda);
 	}
	m_beta = new Eigen::VectorXd(d+1);
	*m_beta = beta;
}

const Eigen::VectorXd* LogistiqueRegression::GetCoefficients() const {
	if (!m_beta) {
		std::cout<<"Coefficients have not been allocated."<<std::endl;
		return NULL;
	}
	return m_beta;
}


int LogistiqueRegression::Estimate( const Eigen::VectorXd & x, double threshold ) const {
	
	int d = m_dataset->GetDim();
	Eigen::VectorXd vec_x(d+1);
	vec_x(0) = 1;
	vec_x.tail(d) = x;

    double result = Sigmoid(vec_x.transpose() * (*m_beta));
    if (result >= threshold)
        return 1; 
	else 
        return 0;
}

double LogistiqueRegression::Estimate_double( const Eigen::VectorXd & x) const {
	int d = m_dataset->GetDim();
	Eigen::VectorXd vec_x(d+1);
	vec_x(0) = 1;
	vec_x.tail(d) = x;

    double result = Sigmoid(vec_x.transpose() * (*m_beta));
	return result;
}
