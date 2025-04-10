# HR Employee Attrition and Department best-fit Prediction Model (DU AI Bootcamp: Module 19 Challenge Due April 9th, 2025)

## Project Overview
This project implements a multi-output neural network to predict employee attrition and department classification simultaneously. 
Using TensorFlow and Keras, the model analyzes various employee attributes to identify patterns that may lead to attrition and determine department placement.

## Dataset
The dataset contains employee information with various features including:
- Demographics (Age, Distance from Home)
- Professional attributes (Education, Job Satisfaction, Work-Life Balance)
- Career metrics (Years at Company, Years Since Last Promotion)
- Engagement indicators (Overtime, Stock Option Level)

The model predicts two target variables:
1. **Attrition** - Whether an employee is likely to leave (Yes/No)
2. **Department** - The department an employee belongs to (Sales, Research & Development, Human Resources)

## Project Structure
The project is organized into two main parts:

### Part 1: Data Preprocessing
- Data loading and exploration
- Feature selection (10 key attributes)
- Train-test split
- Feature encoding (converting categorical data to numeric)
- Data scaling using StandardScaler
- One-hot encoding for target variables

### Part 2: Neural Network Development
- Creation of a multi-output neural network architecture with:
  - Shared base layers to learn common patterns
  - Separate branches for attrition and department prediction
  - Softmax activation functions for classification outputs
- Model compilation with categorical cross-entropy loss
- Training for 100 epochs
- Model evaluation on test data

## Model Architecture
```
Model: "functional"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_layer (InputLayer)    [(None, 10)]              0         
                                                                 
 dense (Dense)               (None, 64)                704       
                                                                 
 dense_1 (Dense)             (None, 128)               8,320     
                                                                 
 dense_3 (Dense)             (None, 32)                4,128     
                                                                 
 dense_2 (Dense)             (None, 32)                4,128     
                                                                 
 attrition_output (Dense)    (None, 2)                 66        
                                                                 
 department_output (Dense)   (None, 3)                 99                                                
=================================================================
Total params: 17,445 (68.14 KB)
Trainable params: 17,445 (68.14 KB)
Non-trainable params: 0 (0.00 B)
```

## Results
The model achieved:
- Attrition prediction accuracy: ~84%
- Department classification accuracy: ~54%

## Analysis Summary

### Metric Evaluation
Accuracy may not be the best metric for this data, especially for attrition prediction. The data is imbalanced (more "No" than "Yes" responses), so it's possible to achieve high accuracy by simply predicting "No" for all employees. Better metrics might include:
- Precision and recall
- F1 score
- ROC-AUC
- Confusion matrix analysis

### Choice of Activation Functions
Softmax activation was selected for both output layers because:
- Both tasks are multi-class classification problems
- Softmax probabilities sum to 1, making it appropriate for mutually exclusive classes
- Softmax works well with categorical cross-entropy loss function

### Potential Improvements
The model could be improved by:
- Adding more features or engineering new features
- Using class weights to better handle imbalanced data
- Implementing regularization techniques to prevent overfitting
- Hyperparameter tuning
- Applying different model architectures
- Using cross-validation instead of a single train-test split

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

## Usage
1. Install the required dependencies:
```
pip install tensorflow pandas numpy scikit-learn
```

2. Run the Jupyter notebook:
```
jupyter notebook attrition.ipynb
```

## License
licensed as "MPL-2.0 license", this project is available for educational purposes.

## Acknowledgments
- The dataset is sourced from IBM HR Analytics
- This project was completed as part of the DU AI Bootcamp and was a course assignment (Challenge 19).
