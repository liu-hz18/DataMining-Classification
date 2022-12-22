import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

metrics = {
    "accuracy": accuracy_score,
    "AUC": roc_auc_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}


model_list = []
date_list = []
acc_list = []
auc_list = []
precision_list = []
recall_list = []
f1_list = []


def predict(train_data, test_data, name):
    model_zoo = {
        "LR": LogisticRegression,
        "DT": DecisionTreeClassifier,
        "XGBoost": XGBClassifier,
        "RF": RandomForestClassifier,
    }
    
    print(f"{name}")
    for model_name in model_zoo.keys():
        model_results = []
        print(f"\t{model_name}")
        if model_name == "LR":
            model = model_zoo[model_name](random_state=42, max_iter=5000)
        elif model_name == "DT":
            model = model_zoo[model_name](random_state=42)
            # model = model_zoo[model_name](random_state=42)
        elif model_name == "RF":
            model = model_zoo[model_name](random_state=42, n_estimators=200)
        else:
            model = model_zoo[model_name](random_state=42)
        model.fit(train_data["data"], train_data["label"])
        pred = model.predict(test_data["data"])
        print(model.get_params())
        # if model_name == "LR":
        #     coef = model.coef_[0]
        #     print("LR coef: ", coef)
        #     plt.bar(list(range(len(coef))), coef)
        #     plt.title(f"Coef of Logistic Regression ({name})")
        #     plt.show()
        # if model_name == "DT":
        #     from sklearn import tree
        #     tree.export_graphviz(model, out_file=open(name + ".dot", "w"))
        #     print(os.system(f"dot -Tpdf {name}.dot -o {name}.pdf"))
        for m in metrics.keys():
            score = metrics[m](test_data["label"], pred)
            model_results.append(score)
            print(f"\t\t{m}: {score}")
        model_list.append(model_name)
        date_list.append(name)
        acc_list.append(model_results[0])
        auc_list.append(model_results[1])
        precision_list.append(model_results[2])
        recall_list.append(model_results[3])
        f1_list.append(model_results[4])


def data_split(data):
    L = len(data["data"])
    print(L)
    idxs_1 = np.random.choice(L, int(L * 0.01), replace=False)
    idxs_10 = np.random.choice(L, int(L * 0.1), replace=False)
    idxs_30 = np.random.choice(L, int(L * 0.3), replace=False)
    idxs_60 = np.random.choice(L, int(L * 0.6), replace=False)

    subdata_1 = {
        "col_names": data["col_names"][1:],
        "data": data["data"][idxs_1, 1:],
        "label": data["label"][idxs_1],
    }
    subdata_10 = {
        "col_names": data["col_names"][1:],
        "data": data["data"][idxs_10, 1:],
        "label": data["label"][idxs_10],
    }
    subdata_30 = {
        "col_names": data["col_names"][1:],
        "data": data["data"][idxs_30, 1:],
        "label": data["label"][idxs_30],
    }
    subdata_60 = {
        "col_names": data["col_names"][1:],
        "data": data["data"][idxs_60, 1:],
        "label": data["label"][idxs_60],
    }
    subdata_100 = {
        "col_names": data["col_names"][1:],
        "data": data["data"][:, 1:],
        "label": data["label"][:],
    }
    return subdata_1, subdata_10, subdata_30, subdata_60, subdata_100


if __name__ == "__main__":
    data_path = "./data"
    train_data = json.load(open(os.path.join(data_path, "train.json"), "r"))
    
    train_data["data"] = np.array(train_data["data"])
    train_data["label"] = np.array(train_data["label"])

    train_data["data"] = np.delete(train_data["data"], 24754, axis=0)
    train_data["label"] = np.delete(train_data["label"], 24754, axis=0)
    
    subdata_1, subdata_10, subdata_30, subdata_60, subdata_100 = data_split(train_data)


    test_data = json.load(open(os.path.join(data_path, "test.json"), 'r'))
    test_data["data"] = np.array(test_data["data"])
    test_data["label"] = np.array(test_data["label"])
    test_data["data"] = test_data["data"][:, 1:]

    predict(subdata_1, test_data, name="1%")
    predict(subdata_10, test_data, name="10%")
    predict(subdata_30, test_data, name="30%")
    predict(subdata_60, test_data, name="60%")
    predict(subdata_100, test_data, name="100%")

    df = pd.DataFrame({"model": model_list, "date": date_list, "accuracy": acc_list, "AUC": auc_list, "precision": precision_list, "recall": recall_list, "f1": f1_list})
    print(df.head())

    lr_data = df[df["model"] == "LR"]
    lr_data
    print(lr_data.head())
    lr_data.set_index('date').plot.barh(figsize=(10,5), xlabel="size", title="Logistic Regression")
    plt.show()

    dt_data = df[df["model"] == "DT"]
    print(dt_data.head())
    dt_data.set_index('date').plot.barh(figsize=(10,5), xlabel="size", title="Decision Tree")
    plt.show()

    xgb_data = df[df["model"] == "XGBoost"]
    print(xgb_data.head())
    xgb_data.set_index('date').plot.barh(figsize=(10,5), xlabel="size", title="XGBoost")
    plt.show()

    rf_data = df[df["model"] == "RF"]
    print(rf_data.head())
    rf_data.set_index('date').plot.barh(figsize=(10,5), xlabel="size", title="Random Forest")
    plt.show()
