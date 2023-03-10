# DM_hw1

Code for homework 1 of Data Mining.

### Results

合并前

|  Model  | Accuracy |  AUC  | Precision | Recall | F1 Score |
| :------: | :------: | :---: | :-------: | :----: | :------: |
|   KNN   |  56.93  | 56.71 |   58.50   | 61.15 |  59.80  |
|   SVM   |  54.51  | 52.60 |   53.82   | 92.67 |  68.10  |
|    DT    |  57.55  | 57.42 |   59.36   | 60.13 |  63.39  |
|    LR    |  58.56  | 58.07 |   59.00   | 68.48 |  63.39  |
|    RS    |  62.79  | 62.91 |   65.82   | 60.25 |  62.91  |
|    RF    |  66.60  | 66.27 |   66.44   | 73.23 |  69.67  |
| XGBoost |  65.73  | 65.52 |   66.42   | 69.91 |  68.12  |
| Lightgbm |  66.85  | 66.57 |   66.98   | 72.42 |  69.59  |

合并后

|  Model  | Accuracy |  AUC  | Precision | Recall | F1 Score |
| :------: | :------: | :---: | :-------: | :----: | :------: |
|   KNN   |  56.59  | 56.35 |   58.40   | 60.88 |  59.61  |
|   SVM   |  55.88  | 54.22 |   55.20   | 85.70 |  67.15  |
|    DT    |  57.63  | 57.51 |   59.75   | 59.69 |  59.72  |
|  Linear  |  47.89  | 50.48 |   87.27   |  1.14  |   2.25   |
|    LR    |  59.31  | 58.68 |   59.56   | 70.69 |  64.65  |
|    RS    |  62.03  | 62.22 |   65.60   | 58.53 |  61.86  |
|    RF    |  66.73  | 66.39 |   66.90   | 72.78 |  69.72  |
| XGBoost |  65.26  | 65.05 |   66.29   | 69.17 |  67.70  |
| Lightgbm |  66.60  | 66.28 |   66.90   | 72.30 |  69.50  |

加入归一化后

|  Model  | Accuracy |  AUC  | Precision | Recall | F1 Score |
| :------: | :------: | :---: | :-------: | :----: | :------: |
|   KNN   |  56.75  | 59.61 |   61.63   | 62.30 |  61.97  |
|   SVM   |  63.43  | 63.01 |   63.72   | 70.83 |  67.09  |
|    DT    |  57.63  | 57.51 |   59.75   | 59.69 |  59.72  |
|  Linear  |  47.89  | 50.48 |   87.27   |  1.14  |   2.25   |
|    LR    |  62.96  | 62.58 |   63.43   | 69.93 |  66.52  |
|    RS    |  62.03  | 62.22 |   65.60   | 58.53 |  61.86  |
|    RF    |  66.73  | 66.39 |   66.90   | 72.78 |  69.72  |
| XGBoost |  65.26  | 65.05 |   66.29   | 69.17 |  67.70  |
| Lightgbm |  66.60  | 66.28 |   66.90   | 72.30 |  69.50  |
|   MLP   |  61.09  | 60.49 |   61.07   | 71.85 |  66.03  |

爬取新闻

|  Model  |     Input     | Accuracy |  AUC  | Precision | Recall | F1 Score |
| :-----: | :------------: | :------: | :---: | :-------: | :----: | :------: |
|  BERT  |     Title     |  52.61  | 50.15 |   52.70   | 97.13 |  68.33  |
|  BERT  | Title+Abstract |  52.24  | 50.01 |   52.63   | 92.52 |  67.09  |
|  BERT  | Title+Content |  52.64  | 50.42 |   52.85   | 92.68 |  67.32  |
| RoBERTa |     Title     |  52.53  | 49.93 |   52.59   | 99.29 |  68.76  |
| RoBERTa | Title+Abstract |  52.96  | 52.10 |   54.20   | 68.53 |  68.96  |
| RoBERTa | Title+Content |  52.96  | 52.10 |   54.20   | 68.53 |  68.96  |

自己手写模型vs调用Sklearn

| Model | Accuracy |  AUC  | Precision | Recall | F1 Score |
| :---: | :------: | :---: | :-------: | :----: | :------: |
|  SVM  | 63.43 |  63.01  | 63.72 |   70.83   | 67.09 | 
| mySVM | 52.48 |  49.95  | 52.60 |   98.00   | 68.46 | 
|  LR  | 62.96 |  62.58  | 63.43 |   69.93   | 66.52 | 
| myLR | 58.33 |  56.46  | 56.37 |   92.04   | 69.92 | 

.
