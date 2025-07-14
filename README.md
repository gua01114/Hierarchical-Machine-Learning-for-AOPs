# Paper title and data availability
This repository contains codes for the paper entitled " Hierarchical Machine Learning-based Prediction for Ultrasonic Degradation of Organic Pollutants using Sonocatalysts" authored by Heewon Jeong, Byung-Moon Jun, Hyo Gyeom Kim, Yeomin Yoon, Kyung Hwa Cho. This repository contains the sample data set and python codes used in this study. 

# Introduction
This study developed a two-stage hierarchical machine learning model that explicitly incorporates the role of H₂O₂ in predicting degradation efficiency in ultrasound-based AOPs using sonocatalysts. The first stage predicts H₂O₂ concentration generated during ultrasonic reactions. The second stage then estimates degradation efficiency using the predicted H₂O₂ levels and other experimental conditions. This stepwise structure reflects the mechanistic sequence of ROS-mediated degradation. 

# Files in this Repository
∙ *_model.py: Files named “_model.py” were used to build a hierarchical machine learning model utilizing 12 machine learning algorithms. Applied machine learning algorithms were AdaBoost (ADB), CatBoost (CB), decision tree (DT), extra trees (ET), gradient boosting decision tree (GBDT), histogram-based gradient boosting (HGB), K-nearest neighbors (KNN), light gradient boosting machine (LGBM), random forest (RF), stochastic gradient descent (SGD), support vector regression (SVR), and XGBoost (XGB). They can be identified by each abbreviations at the beginning of the file name. All the required detailed processes, such as optimizing hyperparameters, training models, and cross validation, are included in the code files. 

∙ *_SHAP.py: Files named “_SHAP.py” were used to apply SHAP analysis for model interpretation.

∙ *_PDP.py: Files named “_PDP.py” were used to apply partial dependence plots for model interpretation.

∙ *_data.xlsx: Files named "_data.xlsx" were the sample data sets for AOPs using MOF as sonocatalyst. The data were collected by our experiments. The detailed experimental processes have been described in our previous publication. The entire data will be made available upon request.

∙ *.pkl: Files in .pkl format correspond to the CB-based models evaluated as the optimal models in this study. The numbers "1" and "2" in the filenames indicate the first and second stages of the hierarchical modeling approach, respectively.

# Findings
This study established a machine learning framework that integrates ROS into predictive modeling for AOPs. The propsosed approach enhanced the mechanistic understanding and optimization of ultrasound-based AOPs with sonocatalysts.

# Correspondance
If you feel any dificulties in executing these codes, please contact me through email on gua01114@gmail.com. Thank you
