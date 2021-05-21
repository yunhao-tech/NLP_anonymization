#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

#ifndef DATASET_HPP
#define DATASET_HPP
/**
  The Dataset class encapsulates a dataset in a vector of vectors and provides a kind of interface to manipulate it.
*/
class Dataset {
    public:
        /**
          The constructor needs the path of the file as a string.
        */
        Dataset(const char* file);
        /**
          The constructor of Dataset from a vector of vector (needed for RandomProjection).
        */
        Dataset(const std::vector<std::vector<double> > &vector_of_vector);
        /**
          Standard destructor
        */
        ~Dataset();
        /**
          The Show method displays the number of instances and columns of the Dataset.
          @param verbose If set to True, the Dataset is also printed.
        */
        void Show(bool verbose) const;
        /**
         Returns a constance reference to an instance.
        @param i Instance number (= row) to get.
        */
    	  const std::vector<double>& GetInstance(int i);
        /**
          The getter to the number of instances / samples.
        */
    	  int GetNbrSamples();
        /**
          The getter to the dimension of the dataset.
        */
    	  int GetDim();

    private:
        /**
          The dimension of the dataset.
        */
		    int m_dim;
        /**
          The number of instances / samples.
        */
		    int m_nsamples;
        /**
          The dataset is stored as a vector of double vectors.
        */
        std::vector<std::vector<double> > m_instances;
};
#endif //DATASET_HPP