"""
这个文件是用来写各种算法的

author: kai
"""
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
#print('Tensorflow Version: {}'.format(tf.__version__))
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'# 只显示 Error
import warnings
warnings.filterwarnings("ignore")
from random import seed


class all_algorithme(object):
    """
    sk_LinearRegression：线性模型


    """

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test


    def sk_LinearRegression(self): #线性模型
        linear = LinearRegression()
        linear.fit(self.x_train, self.y_train)
        self.y_pre_linear = linear.predict(self.x_test)
        self.linear_score = r2_score(self.y_test, self.y_pre_linear)
        print("在线性模型下，R2 决定系数-拟合优度为", self.linear_score)
        return "在线性模型下，R2 决定系数-拟合优度为: {:.2f}".format(self.linear_score), self.y_pre_linear, self.y_test


    def sk_ridge(self): #岭回归
        ridge = Ridge()
        ridge.fit(self.x_train, self.y_train)
        self.y_pre_ridge = ridge.predict(self.x_test)
        self.ridge_score = r2_score(self.y_test, self.y_pre_ridge)
        print("在岭回归的模型下，R2 决定系数-拟合优度为：", self.ridge_score)
        return "在岭回归模型下，R2 决定系数-拟合优度为: {:.2f}".format(self.ridge_score), self.y_pre_ridge, self.y_test


    def sk_DecisionTreeRegressor(self): #决策树回归
        decision = DecisionTreeRegressor()
        decision.fit(self.x_train, self.y_train)
        self.y_pre_decision = decision.predict(self.x_test)
        self.decision_score = r2_score(self.y_test, self.y_pre_decision)
        print("在决策树回归的模型下，R2 决定系数-拟合优度为", self.decision_score)
        return "在决策树回归模型下，R2 决定系数-拟合优度为: {:.2f}".format(self.decision_score), self.y_pre_decision, self.y_test


    def sk_PolynomialFeatures(self): #多项式回归，默认二项式回归 参数degree=3，即为三项式回归
        lr = LinearRegression()
        quadratic = PolynomialFeatures()
        X_train_quad = quadratic.fit_transform(self.x_train)#X_train_quad为用于训练的x的数据

        lr.fit(X_train_quad, self.y_train)
        #self.y_pre_quad = lr.predict(X_train_quad)
        self.y_pred_quad = lr.predict(quadratic.fit_transform(self.x_test))#x_test 为用于测试的x的数据，需要转变形式后再预测
        self.quadratic_score = r2_score(self.y_test, self.y_pred_quad)
        print("在多项式回归模型下，R2 决定系数-拟合优度为", self.quadratic_score)
        return "在多项式回归模型下，R2 决定系数-拟合优度为: {:.2f}".format(self.quadratic_score), self.y_pred_quad, self.y_test


    def sk_byes_network(self):# 贝叶斯网络
        # 3.使用朴素贝叶斯进行训练
        Gnb = GaussianNB()  # 使用默认配置初始化朴素贝叶斯
        Gnb.fit(self.x_train, self.y_train)  # 利用训练数据对模型参数进行估计
        y_predict = Gnb.predict(self.x_test)  # 对参数进行预测

        # 4.获取结果报告
        print('The Accuracy of Naive Bayes Classifier is:', Gnb.score(self.x_test, self.y_test))
        #print(classification_report(self.y_test, y_predict, target_names="boston"))


    def tf_dnn(self): #tensorflow dnn
        print("tf_dnn运行开始")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        """查看网络结构"""
        #model.summary()

        """编译，配置"""
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae', 'mse']
                      )

        """训练数据"""
        history = model.fit(self.x_train, self.y_train, epochs=100, verbose=0)  # verbose=0禁止打印日志，1为进度条，2为打印日志
        hist = pd.DataFrame(history.history)  # hist为训练过程的参数，每一轮训练的loss，mae，mse，epoch


        hist['epoch'] = history.epoch
        # 将训练轮数的索引改为从1开始
        hist['epoch'] = hist['epoch'] + 1
        self.y_pred_dnn = model.predict(self.x_test)

        self.dnn_score = r2_score(self.y_test, self.y_pred_dnn)
        print("在dnn模型下，R2 决定系数-拟合优度为：", self.dnn_score)

        return "在dnn模型下，R2 决定系数-拟合优度为: {:.2f}".format(self.dnn_score), self.y_pred_dnn, self.y_test


    def markf(self):#马尔科夫模型
        P = np.array([[0.2, 0.7, 0.1],
                      [0.9, 0.0, 0.1],
                      [0.2, 0.8, 0.0]])
        stateChangeHist = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0]])
        state = np.array([[1.0, 0.0, 0.0]])
        currentState = 0
        stateHist = state
        dfStateHist = pd.DataFrame(state)
        distr_hist = [[0, 0, 0]]
        seed(4)

        # Simulate from multinomial distribution
        def simulate_multinomial(vmultinomial):
            r = np.random.uniform(0.0, 1.0)
            CS = np.cumsum(vmultinomial)
            CS = np.insert(CS, 0, 0)
            m = (np.where(CS < r))[0]
            nextState = m[len(m) - 1]
            return nextState

        for x in range(1000):
            currentRow = np.ma.masked_values((P[currentState]), 0.0)
            nextState = simulate_multinomial(currentRow)
            # Keep track of state changes
            stateChangeHist[currentState, nextState] += 1
            # Keep track of the state vector itself
            state = np.array([[0, 0, 0]])
            state[0, nextState] = 1.0
            # Keep track of state history
            stateHist = np.append(stateHist, state, axis=0)
            currentState = nextState
            # calculate the actual distribution over the 3 states so far
            totals = np.sum(stateHist, axis=0)
            gt = np.sum(totals)
            distrib = totals / gt
            distrib = np.reshape(distrib, (1, 3))
            distr_hist = np.append(distr_hist, distrib, axis=0)
        print(distrib)
        P_hat = stateChangeHist / stateChangeHist.sum(axis=1)[:, None]
        # Check estimated state transition probabilities based on history so far:
        print(P_hat)
        dfDistrHist = pd.DataFrame(distr_hist)
        # Plot the distribution as the simulation progresses over time
        dfDistrHist.plot(title="Simulation History")
        plt.show()


    def byes(self):#贝叶斯网络
        pass

    def get_log(self):
        return None


    def get_plot(self):# 作图函数
        #Linear
        plt.plot(self.y_test,label='true')
        plt.plot(self.y_pre_linear,label='linear')
        plt.legend()
        #plt.show()


    # def get_r2_score(self): #打印预测结果的平方差
    #     print("在线性模型的条件下，平方差R^2为：", self.linear_score)
    #     # print("在线性模型的条件下，平方差R^2为：", self.linear_score)
    #     # print("在线性模型的条件下，平方差R^2为：", self.linear_score)