# Hyperparameter-Optimization-of-Machine-Learning-Algorithms

This code provides a hyper-parameter optimization implementation for machine learning algorithms, as described in the paper "On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice".  
This paper and code will help industrial users, data analysts, and researchers to better develop machine learning models by identifying the proper hyper-parameter configurations effectively.

## Paper
[On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice](https://www.google.ca/) 
### Quick Navigation
**Section 3**: Common hyper-parameters of machine learning algorithms  
**Section 4**: Hyper-parameter optimization techniques introduction  
**Section 5**: How to choose optimization techniques for different machine learning models  
**Section 6**: Common Python libraries/tools  
**Section 7**: Experimental results (sample code in "HPO_Regression.ipynb" and "HPO_Classification.ipynb")  
**Section 8**: Open challenges and future research directions  
**Summary table for Sections 3-6**: Table 2:  A comprehensive overview of common ML models, their hyper-parameters, suit-able optimization techniques, and available Python libraries  
**Summary table for Sections 8**: Table 10:  The open challenges and future directions of HPO research  

## Implementation
Sample code for hyper-parameter optimization implementation for machine learning algorithms is provided in this repository.  

**Sample code for Regression problems**  
[HPO_Regression.ipynb]  
Dataset used: [Boston-Housing]  
**Sample code for Classification problems**  
[HPO_Classification.ipynb]  
Dataset used: [MNIST]  

**Machine learning algorithms used:**  
* Random forest (RF)
* Support vector machine (SVM)
* K-nearest neighbor (KNN)  

**HPO algorithms used:**  
* Grid search
* Random search
* Hyperband
* Bayesian Optimization with Gaussian Processes (BO-GP)
* Bayesian Optimization with Tree-structured Parzen Estimator (BO-TPE)
* Particle swarm optimization (PSO)
* Genetic algorithm (GA).  

**Python libraries requirements** 
* Python 3.5  
* scikit-learn  
* hyperband  
* scikit-optimize  
* hyperopt  
* optunity  
* DEAP  
* TPOT  

## Citation
If you find this repository useful in your research, please cite:  
L. Yang and A. Shami, "On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice," Neurocomputing, 2020.
