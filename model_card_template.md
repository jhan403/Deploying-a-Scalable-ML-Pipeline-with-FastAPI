# Model Card

## Model Details
This Random Forest Classifier model is trained on the US Census dataset to predict if a subjects income is less than or greater than $50K.

## Intended Use
This model is a class assignment and is for demonstration purposes only.

## Training Data
This model uses 80% of the US Census dataset for training and 20% for testing. Features include: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary.

## Evaluation Data
The testing dataset is used to evaluate the performance of the model using all the same features as the training dataset.

## Metrics
Precision: 0.7392 | Recall: 0.6321 | F1: 0.6814

## Ethical Considerations
This dataset contains features such as age, race, sex, and education that can introduce bias. All results should be inspected for bias. 

## Caveats and Recommendations
This model is for demonstration only and care should be taken if used in a real-world scenerio.
