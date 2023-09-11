import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

st.title('Machine Learning Algorithm')

st.write("""# Explore different classifier 
         Which one is the best?""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine dataset'))

st.write(dataset_name)

classifier_name = st.sidebar.selectbox('Select Dataset', ('KNN', 'SVM', 'Random Forest', 'Logistic Regression'))

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()

    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()

    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target
    return x,y

x, y = get_dataset(dataset_name)

st.write('Shape of Dataset', x.shape)
st.write('Number of Classes', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()

    if clf_name == 'KNN':
        K = st.sidebar.selectbox('K',(np.arange(1,20,1)))
        params["K"] = K

    elif clf_name == 'SVM':
        C = st.sidebar.slider("C", 0.01,15.0)
        kernel = st.sidebar.selectbox('Kernel',('linear','poly','rbf','sigmoid'))
        gamma = st.sidebar.slider('gamma',0.1,20.0)
        params["C"] = C
        params["kernel"] = kernel
        params["gamma"] = gamma

    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth',2,50)
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        criterion = st.sidebar.selectbox('criterion',('gini','entropy'))
        params["max_depth"] = max_depth
        params["n_estimators"] =n_estimators
        params["criterion"] = criterion

    else:
        penalty = st.sidebar.selectbox('penalty',('l1','l2'))
        C = st.sidebar.slider("C", 0.01,5.0)
        params["C"] = C
        params["penalty"] = penalty

    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == 'SVM':
        clf = SVC(C=params["C"],kernel=params["kernel"],gamma=params["gamma"])

    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"],criterion=params["criterion"],random_state=100)

    else:
        clf = LogisticRegression(C=params["C"],penalty=params["penalty"])

    return clf

clf = get_classifier(classifier_name, params)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=100)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)

st.write(f'classifier = {classifier_name}')
st.write(f'accuracy = {acc}')

pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)






