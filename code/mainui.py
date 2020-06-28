"""
UI文件

author：kai
"""
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from main_interface import Ui_MainWindow
from alldata import datas
from algorithme import all_algorithme


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        #self.Run.clicked.connect(self.label_2)
        pix = QPixmap('0.jpg')
        pix = pix.scaled(1920, 1080)#1920, 1080
        lb = self.ans
        lb.setPixmap(pix)
        self.pushButton.clicked.connect(self.refresh_algorithme)
        self.run.clicked.connect(self.run_all_func)
        self.about_us.clicked.connect(self.show_message)
        self.all_ans.clicked.connect(self.pic_detect)


    def run_all_func(self):  #选择大方向，分类，聚类还是预测
        if(self.choose_orient.currentText() == "聚类"):
            self.run_culster_algorithme()
        elif(self.choose_orient.currentText() == "预测"):
            self.run_predict_algorithme()
        else:
            self.run_classify_algorithme()


    def run_classify_algorithme(self):
        pass


    def refresh_a(self):
        if(self.choose_orient.currentIndex() == 0):
            self.choose_algorithme.addItem("AGNES")
            self.choose_algorithme.addItem("BIRCH")
            self.choose_algorithme.addItem("DBSCAN")
            self.choose_algorithme.addItem("GMM")
            self.choose_algorithme.addItem("k-means")
            self.choose_algorithme.addItem("Mean Shift")

        elif(self.choose_orient.currentIndex() == 1):
            self.choose_algorithme.addItem("线性回归")
            self.choose_algorithme.addItem("岭回归")
            self.choose_algorithme.addItem("多项式回归")
            self.choose_algorithme.addItem("决策树回归")
            self.choose_algorithme.addItem("DNN")
            self.choose_algorithme.addItem("贝叶斯网络")
            self.choose_algorithme.addItem("马尔科夫模型")
            #self.run_predict_algorithme()
        else:
            self.choose_algorithme.addItem("KNN")
            self.choose_algorithme.addItem("SVM支持向量机")
            self.choose_algorithme.addItem("朴素贝叶斯")
            self.choose_algorithme.addItem("AdaBoost")
            # self.choose_algorithme.addItem("随机森林")
            # self.choose_algorithme.addItem("逻辑回归")
            # self.choose_algorithme.addItem("GBDT")
        # print(self.choose_orient.currentIndex())
        # print(self.choose_orient.currentText())


    def choose_data_func(self):  #选择数据集大小函数
        data_size = self.choose_data.currentText()
        if(data_size == "小"):
            x_train, x_test, y_train, y_test = self.d.mini_data()  # c=a+a^2+6
        elif(data_size == "大"):
            x_train, x_test, y_train, y_test = self.d.tremendous_data()
        else:
            x_train, x_test, y_train, y_test = self.d.medium_data()
        return x_train, x_test, y_train, y_test


    def run_predict_algorithme(self): #运行预测算法函数
        self.d = datas()
        x_train, x_test, y_train, y_test = self.choose_data_func()
        a = all_algorithme(x_train, x_test, y_train, y_test)
        current_algo = self.choose_algorithme.currentText()
        if(current_algo == "线性回归"):
            ans, y_pre, y_test = a.sk_LinearRegression()
        elif(current_algo == "岭回归"):
            ans, y_pre, y_test = a.sk_ridge()
        elif(current_algo == "多项式回归"):
            ans, y_pre, y_test = a.sk_PolynomialFeatures()
        elif (current_algo == "决策树回归"):
            ans, y_pre, y_test = a.sk_DecisionTreeRegressor()
        elif(current_algo == "DNN"):
            ans, y_pre, y_test = a.tf_dnn()
        # else:
        #     QMessageBox.information(self, "请先选择算法", QMessageBox.Warning)
        # elif(current_algo == "决策树回归"):
        #     ans, y_pre_linear, y_test = a.sk_DecisionTreeRegressor()
        self.plot_and_log(ans, y_pre, y_test)  # 画图打印日志

    def plot_and_log_for_cluster(self):

        pix = QPixmap(self.culster_ans_path)
        self.ans.setScaledContents(True)
        self.ans.setPixmap(pix)


    def run_culster_algorithme(self):
        current_algo = self.choose_algorithme.currentText()
        current_data = self.choose_data.currentText()
        if(current_algo == "AGNES"):
            self.culster_ans_path = r'./cluster/AGNES/鸢尾花数据集3.png'
            self.textEdit.setText("")
            # elif(current_data == "中"):
            #     self.culster_ans_path = r'./cluster/AGNES/鸢尾花数据集2.png'
            #     self.textEdit.setText("AGNES均值 = \
            #                  [[5.006      3.428      1.462      0.246     ]\
            #                  [5.9016129  2.7483871  4.39354839 1.43387097]\
            #                  [6.85       3.07368421 5.74210526 2.07105263]] \
            #                 分类正确率为 0.8933333333333333")
            # else:
            #     self.culster_ans_path = r'./cluster/AGNES/鸢尾花数据集3.png'
            #     self.textEdit.setText("AGNES均值 = \
            #                  [[5.006      3.428      1.462      0.246     ]\
            #                  [5.9016129  2.7483871  4.39354839 1.43387097]\
            #                  [6.85       3.07368421 5.74210526 2.07105263]] \
            #                 分类正确率为 0.8933333333333333")

        elif(current_algo == "BIRCH"):
            if (current_data == "小"):
                self.culster_ans_path = './cluster/BIRCH/综合聚类数据集2.png'
                self.textEdit.setText("BIRCH，轮廓系数为0.4243692758642153 。")
            if (current_data == "中"):
                self.culster_ans_path = './cluster/BIRCH/鸢尾花数据集2.png'
                self.textEdit.setText("BIRCH，聚类正确率为0.23333333333334 。")
            if (current_data == "大"):
                self.culster_ans_path = './cluster/BIRCH/随机数据集3.png'
                self.textEdit.setText("BIRCH，Calinski-Harabasz Score为3301.8023 。")

        elif(current_algo == "DBSCAN"):
            if (current_data == "小"):
                self.culster_ans_path = './cluster/DBSCAN/综合聚类数据集2.png'
                self.textEdit.setText("DBSCAN，轮廓系数为0.4230767949856645 。")
            if (current_data == "中"):
                self.culster_ans_path = './cluster/DBSCAN/鸢尾花数据集3.png'
                self.textEdit.setText("DBSCAN，聚类正确率为0.41333333333 。")
            if (current_data == "大"):
                self.culster_ans_path = './cluster/DBSCAN/啤酒数据集4.png'
                self.textEdit.setText("DBSCAN，轮廓系数为0.49530955296776086")


        elif(current_algo == "GMM"):
            if (current_data == "小"):
                self.culster_ans_path = './cluster/GMM/综合聚类数据集2.png'
                self.textEdit.setText("高斯混合聚类GMM，轮廓系数为0.42967355323274 。")
            if (current_data == "中"):
                self.culster_ans_path = './cluster/GMM/鸢尾花数据集3.png'
                self.textEdit.setText("DBSCAN，聚类正确率为0.793333333 。")
            if (current_data == "大"):
                self.culster_ans_path = './cluster/GMM/高斯混合模型4.png'
                self.textEdit.setText(" mu:\
                        [[0.80381015 0.52303569]\
                         [0.26065489 0.60734094]]\
                        cov:\
                        [[[0.00987706 0.0008622 ]\
                          [0.0008622  0.04811051]]\
                        \
                         [[0.01417906 0.00216207]\
                          [0.00216207 0.00822912]]]\
                        alpha:\
                        [0.6894792 0.3105208]")


        elif(current_algo == "k-means"):
            if (current_data == "小"):
                self.culster_ans_path = './cluster/k-means/综合聚类数据集2.png'
                self.textEdit.setText("K-means，轮廓系数为0.4270730877908618。 ")
            if (current_data == "中"):
                self.culster_ans_path = './cluster/k-means/鸢尾花数据集2.png'
                self.textEdit.setText("K-means，聚类正确率为0.23333333333334 。")
            if (current_data == "大"):
                self.culster_ans_path = './cluster/k-means/啤酒数据集5.png'
                self.textEdit.setText("K-means，轮廓系数为0.6917656034079486")


        elif(current_algo == "Mean Shift"):
            if (current_data == "小"):
                self.culster_ans_path = './cluster/Mean Shift/综合聚类数据集2.png'
                self.textEdit.setText("Mean Shift，轮廓系数为0.3994440156335262 。")
            if (current_data == "中"):
                self.culster_ans_path = './cluster/Mean Shift/鸢尾花数据集2.png'
                self.textEdit.setText("Mean Shift，聚类正确率为0.00666666667 。")
            if (current_data == "大"):
                self.culster_ans_path = './cluster/Mean Shift/三簇数据集1 .png'
                self.textEdit.setText("Mean Shift，估计的群集数 : 3\
                        十大样本: [[ 1.35567205  1.03426149  2.        ]\
                         [ 1.84313781 -0.46642773  0.        ]\
                         [-1.46430267 -0.86316999  1.        ]\
                         [ 1.44972653 -0.29907961  0.        ]\
                         [ 0.46646329 -1.28829248  0.        ]\
                         [ 0.09852578  1.61534244  2.        ]\
                         [ 1.35865115 -2.04268199  0.        ]\
                         [ 0.33726529 -1.41196828  0.        ]\
                         [-0.2340017  -0.17775738  1.        ]\
                         [ 0.77906683 -0.64267692  0.        ]]")

        self.plot_and_log_for_cluster()

    def plot_and_log(self, ans, y_test, y_pre_linear):#画图打印以及生成日志
        self.textEdit.setText(ans)
        # 画图
        plt.figure(dpi=200)
        plt.plot(y_test, label='true')
        plt.plot(y_pre_linear, label='linear')
        plt.legend()

        plt.savefig('线性回归.jpg')
        pix = QPixmap('线性回归.jpg')
        self.ans.setScaledContents(True)
        #pix = pix.scaled(600, 400)  # 1920, 1080
        # self.ans.setPixmap()
        self.ans.setPixmap(pix)

    def pic_detect(self):
        #打开文件夹选择待检测文件
        fileName, filetype = QFileDialog.getOpenFileName(self, "选择文件", r"C:\Users\kai\Desktop\m_l\code\integrate\cluster", "All Files (*);;Text Files (*.txt)")
        print(fileName)  # 打印文件全部路径（包括文件名和后缀名）


    def show_message(self):
        QMessageBox.information(self, "关于我们", "华北理工大学17智能一班：赵国庆，董润玺，梁帅凯",
                                QMessageBox.Yes)

    def refresh_algorithme(self):
        count = self.choose_algorithme.count()

        while(count!=0):
            self.choose_algorithme.removeItem(count-1)
            count = self.choose_algorithme.count()

        self.refresh_a()
        # print(self.choose_orient.currentIndex())
        # print(self.choose_orient.currentText())


    def run_classify_algorithme(self):
        print("分类   run_classify_algorithme")
        current_algo = self.choose_algorithme.currentText()
        if(current_algo == "KNN"):
            self.classify_ans_path = r'./classify/KNN/图片1.png'
            self.textEdit.setText("")
        elif(current_algo == "SVM支持向量机"):
            self.classify_ans_path = './classify/SVM支持向量机/图片3.png'
            self.textEdit.setText("SVM支持向量机\
                        分类正确率为 0.7833333")
        elif(current_algo == "朴素贝叶斯"):
            self.classify_ans_path = r'./classify/朴素贝叶斯/图片4.png'
            self.textEdit.setText("['apple' 'mandarin' 'orange' 'lemon']\
                        fruit_name\
                        apple       19\
                        lemon       16\
                        mandarin     5\
                        orange      19\
                        dtype: int64")
        # elif(current_algo == "AdaBoost"):
        #     self.classify_ans_path = './classify/AdaBoost/'
        #     self.textEdit.setText("高斯混合聚类GMM  分类正确率为 0.7933333333333333")
        print("分类jieshu   run_classify_algorithme")
        # elif(current_algo == "k-means"):
        #     self.culster_ans_path = './cluster/k-means/综合聚类数据集2.png'
        #     self.textEdit.setText("K-Means均值 = \
        #                  [[5.77358491 2.69245283]\
        #                  [5.006      3.428     ]\
        #                  [6.81276596 3.07446809]]\
        #                 分类正确率为 0.23333333333333334")
        # elif(current_algo == "Mean Shift"):
        #     self.culster_ans_path = './cluster/Mean Shift/综合聚类数据集2.png'
        #     self.textEdit.setText("Mean Shift均值 = \
        #                  [[6.27373737 2.87575758 4.92525253 1.68181818]\
        #                  [5.00784314 3.40980392 1.49215686 0.2627451 ]\
        #                  [       0.001         0.001         0.001         0.001]]\
        #                 分类正确率为 0.006666666666666667")

        self.plot_and_log_for_classify()


    def plot_and_log_for_classify(self):
        pix = QPixmap(self.classify_ans_path)
        self.ans.setScaledContents(True)
        self.ans.setPixmap(pix)