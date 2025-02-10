import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import dagshub
# dagshub.init(repo_owner='nipun5', repo_name='AquaPredict_MLFLOW_Dagshub', mlflow=True)


mlflow.set_experiment("water_exp3")
# mlflow.set_tracking_uri("https://dagshub.com/nipun5/AquaPredict_MLFLOW_Dagshub.mlflow")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
data = pd.read_csv("water_potability.csv")

from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle
X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

n_estimators = 500
max_depth=10
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    clf.fit(X_train,y_train)

    # save 
    pickle.dump(clf,open("model.pkl","wb"))

    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    model = pickle.load(open('model.pkl',"rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_score = f1_score(y_test,y_pred)

    mlflow.log_metric("acc",acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score",f1_score)

    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("max_depth",max_depth)

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(clf,"RandomForestClassifier")

    mlflow.log_artifact(__file__)

    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")

    mlflow.set_tag("author","AquaPredict")
    mlflow.set_tag("model","RF")

    print("acc",acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score",f1_score)