from flask import Flask, render_template, request, url_for, redirect
import os
import csv
import pandas as pd
from ensembles import RandomForestMSE, GradientBoostingMSE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
dict = []
train = None
val = None
loss_val = None
loss_test = None

@app.route("/", methods=['GET', 'POST'])
def index():

    return render_template('index.html')


@app.route("/random_forest", methods=['GET', 'POST'])
def idk():
    global dict
    global train
    global val
    if request.method == 'POST':
        dict = list(request.form.to_dict().values())
        print(dict)
        f_train = request.files['train']
        f_val = request.files['val']
        if f_train:
            train = pd.read_csv(f_train)
        if f_val:
            val = pd.read_csv(f_val)

        return redirect(url_for('cringe'))
        #render_template('about_rf.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2])
    return render_template('random_forest.html')


def preprocess_data(tr, vl):
    size_tr = tr.shape[0]
    # size_vl = vl.shape[0]
    df = pd.concat([tr, vl])
    df.drop(['id'], axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df.drop("date", axis=1, inplace=True)
    categorical = ['year', 'month', 'day', 'view', 'condition',
                   'grade', 'yr_built', 'yr_renovated', 'zipcode', 'waterfront']
    y = np.log1p(df['price'].values)
    X = df.drop(['price'], axis=1)
    numerical = np.setdiff1d(X.columns, categorical)
    X_ohe = pd.get_dummies(X, columns=categorical)
    print(X_ohe.shape)
    scaler = MinMaxScaler()
    # print(X_ohe)
    X_ohe[numerical] = scaler.fit_transform(X_ohe[numerical])

    return X_ohe[:size_tr], y[:size_tr], X_ohe[size_tr:], y[size_tr:]


@ app.route("/random_forest/about", methods=['GET', 'POST'])
def cringe():
    global dict
    global train
    global val
    global loss_val
    if request.method == 'POST':
        train_data = train
        val_data = val
        X_train, y_train, X_val, y_val = preprocess_data(train_data, val_data)
        # X_val, y_val = preprocess_data(val_data)
        model = RandomForestMSE(n_estimators=int(
            dict[0]), feature_subsample_size=int(dict[1]), max_depth=int(dict[2]))
        print('****************************')
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
        model.fit(X_train.values, y_train, X_val.values,
                  y_val, is_loss_all=False)
        print(model.loss)
        loss_val = model.loss
        return redirect(url_for('koala'))
    return render_template('about_rf.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2])


@ app.route("/random_forest/predict", methods=['GET', 'POST'])
def koala():
    global dict
    global train
    global loss_test
    if request.method == 'POST':
        train_data = train
        f_test = request.files['test']
        if f_test:
            test_data = pd.read_csv(f_test)
            X_train, y_train, X_test, y_test = preprocess_data(
                train_data, test_data)
            # X_val, y_val = preprocess_data(val_data)
            model = RandomForestMSE(n_estimators=int(
                dict[0]), feature_subsample_size=int(dict[1]), max_depth=int(dict[2]))
            print('****************************')
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            model.fit(X_train.values, y_train, X_test.values,
                      y_test, is_loss_all=False)
            print(model.loss)
            loss_test = model.loss
            return redirect(url_for('panda'))
    return render_template('predict_rf.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2], loss_val=loss_val)


@ app.route("/random_forest/predict/results", methods=['GET', 'POST'])
def panda():
    global dict
    global train
    global loss_test
    if request.method == 'POST':
        return redirect(url_for('index'))
    
    return render_template('final_rf.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2], loss_val=loss_val, loss_test=loss_test)


@app.route("/gradient_boosting", methods=['GET', 'POST'])
def idk_gb():
    global dict
    global train
    global val
    if request.method == 'POST':
        dict = list(request.form.to_dict().values())
        print(dict)
        f_train = request.files['train']
        f_val = request.files['val']
        if f_train:
            train = pd.read_csv(f_train)
        if f_val:
            val = pd.read_csv(f_val)

        return redirect(url_for('cringe_gb'))
        #render_template('about_rf.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2])
    return render_template('gradient_boosting.html')




@ app.route("/gradient_boosting/about", methods=['GET', 'POST'])
def cringe_gb():
    global dict
    global train
    global val
    global loss_val
    if request.method == 'POST':
        train_data = train
        val_data = val
        X_train, y_train, X_val, y_val = preprocess_data(train_data, val_data)
        # X_val, y_val = preprocess_data(val_data)
        model = GradientBoostingMSE(n_estimators=int(
            dict[0]), feature_subsample_size=int(dict[1]), max_depth=int(dict[2]), learning_rate=float(dict[3]))
        print('****************************')
        print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
        model.fit(X_train.values, y_train, X_val.values,
                  y_val, is_loss_all=False)
        print(model.loss)
        loss_val = model.loss
        return redirect(url_for('koala_gb'))
    return render_template('about_gb.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2], lr=dict[3])


@ app.route("/gradient_boosting/predict", methods=['GET', 'POST'])
def koala_gb():
    global dict
    global train
    global loss_test
    if request.method == 'POST':
        train_data = train
        f_test = request.files['test']
        if f_test:
            test_data = pd.read_csv(f_test)
            X_train, y_train, X_test, y_test = preprocess_data(
                train_data, test_data)
            # X_val, y_val = preprocess_data(val_data)
            model = GradientBoostingMSE(n_estimators=int(
                dict[0]), feature_subsample_size=int(dict[1]), max_depth=int(dict[2]), learning_rate=float(dict[3]))
            print('****************************')
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            model.fit(X_train.values, y_train, X_test.values,
                      y_test, is_loss_all=False)
            print(model.loss)
            loss_test = model.loss
            return redirect(url_for('panda_gb'))
    return render_template('predict_gb.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2], lr=dict[3], loss_val=loss_val)


@ app.route("/gradient_boosting/predict/results", methods=['GET', 'POST'])
def panda_gb():
    global dict
    global train
    global loss_test
    if request.method == 'POST':
        return redirect(url_for('index'))
    
    return render_template('final_gb.html', num_trees=dict[0], dim_trees=dict[1], dep_trees=dict[2], lr=dict[3], loss_val=loss_val, loss_test=loss_test)