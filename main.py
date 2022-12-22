import json
import sklearn
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from models.mlp import MultiLayerPerceptronClassifier
from models.LR import myLRModel
from models.SVM import mySVMModel
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def SVMModel():
	return svm.SVC(probability=True, random_state=42)

def DecisionTreeModel():
	return DecisionTreeClassifier(random_state=42)

def LogisticRegressionModel():
	return LogisticRegression(random_state=42)

def LinearRegressionModel():
	return LinearRegression()

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

def MultiLayerPerceptronModel():
	return MultiLayerPerceptronClassifier(random_state=42, lr=1e-4, h1=256, h2=64, epoch=2, batch_size=64)

def MyLRModel():
	return myLRModel(random_state=42)

def MySVMModel():
	return mySVMModel(random_state=42)

def Norm(train_data, test_data):
	N = len(train_data[0])
	mx, mn = [-1e9 for _ in range(N)], [1e9 for _ in range(N)]
	for data in [train_data, test_data]:
		for line in data:
			for i in range(N):
				mx[i] = max(mx[i], line[i])
				mn[i] = min(mn[i], line[i])
	for j in range(N): mx[j] = mx[j] - mn[j]
	for i in range(len(train_data)): 
		for j in range(N): train_data[i][j] = (train_data[i][j] - mn[j]) / mx[j]
	for i in range(len(test_data)): 
		for j in range(N): test_data[i][j] = (test_data[i][j] - mn[j]) / mx[j]
	return train_data, test_data

def MetricFunc(label, pred):
	pred = pred.astype(int)
	return {'Accuracy': accuracy_score(label, pred), 'AUC': roc_auc_score(label, pred), 'Precision':precision_score(label, pred), 'Recall':recall_score(label, pred), 'F1 Score':f1_score(label, pred)}

def SklearnMain(train_data, test_data):
	ModelDict = {'KNN':KNearestNeighborsModel, 'SVM': SVMModel, 'DT': DecisionTreeModel, 'LR': LogisticRegressionModel, 'Linear':LinearRegressionModel, 'RS': RandomSubspaceModel, 'RF': RandomForestModel, 'XGBoost': XGBoostModel, 'Lightgbm': LightgbmModel, 'MLP':MultiLayerPerceptronModel}
	ModelDict = {'myLR': MyLRModel}
	ModelDict = {'mySVM': MySVMModel}
	need_norm = ['SVM', 'LR', 'MLP', 'KNN', 'Linear', 'myLR', 'mySVM']
	norm_train_data, norm_test_data = Norm(train_data['data'], test_data['data'])
	for model_name in ModelDict.keys():
		model = ModelDict[model_name]()
		if model_name in need_norm:
			model.fit(norm_train_data, train_data['label'])
			pred = model.predict(norm_test_data)
		else:
			model.fit(train_data['data'], train_data['label'])
			pred = model.predict(test_data['data'])
		res = MetricFunc(test_data['label'], pred)
		print('{}: {}'.format(model_name, res))

if __name__ == '__main__':
	data_path = 'data/'
	train_data = json.load(open(data_path + 'train.json', 'r'))
	test_data = json.load(open(data_path + 'test.json', 'r'))
	
	SklearnMain(train_data, test_data)

