"""
这个文件是运行文件，是用来运行主程序代码的

author： kai
"""
from alldata import datas
from algorithme import all_algorithme
from mainui import MyWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets

if __name__ == '__main__':

    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()


    # d = datas()
    # # x_train, x_test, y_train, y_test = d.medium_data()
    # x_train, x_test, y_train, y_test = d.tremendous_data()
    # # x_train, x_test, y_train, y_test = d.mini_data() # c=a+a^2+6
    # a = all_algorithme(x_train, x_test, y_train, y_test)
    # a.sk_LinearRegression()
    # a.sk_ridge()
    # a.sk_PolynomialFeatures()
    # a.sk_DecisionTreeRegressor()
    # #a.sk_byes_network()
    # a.tf_dnn()

    #a.get_plot()
    #a.get_r2_score()



    sys.exit(app.exec_())
