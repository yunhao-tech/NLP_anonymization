#include "Classification.hpp"
#include "Dataset.hpp"

Classification::Classification(Dataset* dataset, std::vector<int> labels) {
    m_dataset = dataset;
    m_labels = labels;
}

Dataset* Classification::getDataset(){
    return m_dataset;
}

