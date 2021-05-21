EIGEN=/usr/local/eigen-3.3.7

CC=g++

all: test_LR

INCLUDES=-I$(EIGEN)
CXXFLAGS=$(INCLUDES) -std=c++11 -g -O2 -Wall -Wextra

LDFLAGS=

%: %.o
	$(CC)  $(CXXFLAGS) $^ $(LDFLAGS) -o $@

Dataset.o: Dataset.cpp Dataset.hpp
LogistiqueRegression.o: LogistiqueRegression.cpp LogistiqueRegression.hpp
Classification.o: Classification.cpp Classification.hpp Dataset.hpp
test_LR.o: test_LR.cpp LogistiqueRegression.hpp ConfusionMatrix.hpp 
ConfusionMatrix.o: ConfusionMatrix.cpp ConfusionMatrix.hpp

test_LR: test_LR.o Classification.o ConfusionMatrix.o Dataset.o LogistiqueRegression.o
test_LR_multinomial: test_LR_multinomial.o Classification.o ConfusionMatrix.o Dataset.o LogistiqueRegression.o

.PHONY: all clean
