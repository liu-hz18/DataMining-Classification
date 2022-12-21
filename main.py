import json
import sklearn
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def SVMModel():
	return svm.SVC(probability=True, random_state=42)

def DecisionTreeModel():
	return DecisionTreeClassifier(random_state=42)

def LogisticRegressionModel():
	return LogisticRegression(random_state=42)

def RandomSubspaceModel():
	return BaggingClassifier(random_state=42)

def RandomForestModel():
	return RandomForestClassifier(random_state=42, n_estimators=200)

def XGBoostModel():
	return XGBClassifier(random_state=42)

def LightgbmModel():
	return LGBMClassifier(random_state=42)

def KNearestNeighborsModel():
	return KNeighborsClassifier()

def MetricFunc(label, pred):
	return {'Accuracy': accuracy_score(label, pred), 'AUC': roc_auc_score(label, pred), 'Precision':precision_score(label, pred), 'Recall':recall_score(label, pred), 'F1 Score':f1_score(label, pred)}

def SklearnMain(train_data, test_data):
	ModelDict = {'SVM': SVMModel, 'DT': DecisionTreeModel, 'LR': LogisticRegressionModel, 'RS': RandomSubspaceModel, 'RF': RandomForestModel, 'XGBoost': XGBoostModel, 'Lightgbm': LightgbmModel, 'KNN':KNearestNeighborsModel}
	ModelDict = {'KNN':KNearestNeighborsModel}
	for model_name in ModelDict.keys():
		model = ModelDict[model_name]()
		model.fit(train_data['data'], train_data['label'])
		pred = model.predict(test_data['data'])
		res = MetricFunc(test_data['label'], pred)
		print('{}: {}'.format(model_name, res))

if __name__ == '__main__':
	data_path = 'data/'
	train_data = json.load(open(data_path + 'train.json', 'r'))
	test_data = json.load(open(data_path + 'test.json', 'r'))
	
	SklearnMain(train_data, test_data)

