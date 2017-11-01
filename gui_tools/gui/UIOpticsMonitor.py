# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UIOpticsMonitor.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1222, 914)
        MainWindow.setMinimumSize(QtCore.QSize(1150, 0))
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget_2.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tabWidget_2.setFont(font)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_6.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_6.setSpacing(6)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.w_twiss_monitor = OclMonitor(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.w_twiss_monitor.sizePolicy().hasHeightForWidth())
        self.w_twiss_monitor.setSizePolicy(sizePolicy)
        self.w_twiss_monitor.setMinimumSize(QtCore.QSize(0, 0))
        self.w_twiss_monitor.setObjectName("w_twiss_monitor")
        self.gridLayout_6.addWidget(self.w_twiss_monitor, 0, 0, 1, 2)
        self.w_q_table = VOclTable(self.tab_4)
        self.w_q_table.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.w_q_table.sizePolicy().hasHeightForWidth())
        self.w_q_table.setSizePolicy(sizePolicy)
        self.w_q_table.setMinimumSize(QtCore.QSize(0, 200))
        self.w_q_table.setMaximumSize(QtCore.QSize(400, 16777215))
        self.w_q_table.setObjectName("w_q_table")
        self.gridLayout_6.addWidget(self.w_q_table, 0, 3, 1, 1)
        self.tabWidget_2.addTab(self.tab_4, "")
        self.gridLayout.addWidget(self.tabWidget_2, 0, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.sb_delay = QtWidgets.QDoubleSpinBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.sb_delay.setFont(font)
        self.sb_delay.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.sb_delay.setMinimum(0.5)
        self.sb_delay.setProperty("value", 1.0)
        self.sb_delay.setObjectName("sb_delay")
        self.gridLayout_3.addWidget(self.sb_delay, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.cb_design = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.cb_design.setFont(font)
        self.cb_design.setObjectName("cb_design")
        self.horizontalLayout.addWidget(self.cb_design)
        self.cb_otrc55 = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.cb_otrc55.setFont(font)
        self.cb_otrc55.setObjectName("cb_otrc55")
        self.horizontalLayout.addWidget(self.cb_otrc55)
        self.cb_otrb218 = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.cb_otrb218.setFont(font)
        self.cb_otrb218.setObjectName("cb_otrb218")
        self.horizontalLayout.addWidget(self.cb_otrb218)
        self.cb_otrb450 = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.cb_otrb450.setFont(font)
        self.cb_otrb450.setObjectName("cb_otrb450")
        self.horizontalLayout.addWidget(self.cb_otrb450)
        self.gridLayout_3.addLayout(self.horizontalLayout, 1, 0, 1, 4)
        self.pb_start_stop = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pb_start_stop.sizePolicy().hasHeightForWidth())
        self.pb_start_stop.setSizePolicy(sizePolicy)
        self.pb_start_stop.setMinimumSize(QtCore.QSize(0, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pb_start_stop.setFont(font)
        self.pb_start_stop.setStyleSheet("color: rgb(85, 255, 255);")
        self.pb_start_stop.setObjectName("pb_start_stop")
        self.gridLayout_3.addWidget(self.pb_start_stop, 0, 4, 2, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_4.setSpacing(6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pb_hide_show = QtWidgets.QPushButton(self.groupBox_2)
        self.pb_hide_show.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pb_hide_show.sizePolicy().hasHeightForWidth())
        self.pb_hide_show.setSizePolicy(sizePolicy)
        self.pb_hide_show.setMinimumSize(QtCore.QSize(0, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pb_hide_show.setFont(font)
        self.pb_hide_show.setStyleSheet("color: rgb(85, 255, 127);")
        self.pb_hide_show.setObjectName("pb_hide_show")
        self.gridLayout_4.addWidget(self.pb_hide_show, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_2, 0, 2, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1222, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu_File = QtWidgets.QMenu(self.menuBar)
        self.menu_File.setObjectName("menu_File")
        self.menuGolden_Orbit = QtWidgets.QMenu(self.menuBar)
        self.menuGolden_Orbit.setObjectName("menuGolden_Orbit")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.action_Parameters = QtWidgets.QAction(MainWindow)
        self.action_Parameters.setObjectName("action_Parameters")
        self.actionLoad_Golden_Orbit = QtWidgets.QAction(MainWindow)
        self.actionLoad_Golden_Orbit.setObjectName("actionLoad_Golden_Orbit")
        self.actionSave_Golden_Orbit = QtWidgets.QAction(MainWindow)
        self.actionSave_Golden_Orbit.setObjectName("actionSave_Golden_Orbit")
        self.actionRead_BPMs_Corrs = QtWidgets.QAction(MainWindow)
        self.actionRead_BPMs_Corrs.setObjectName("actionRead_BPMs_Corrs")
        self.actionCalculate_RM = QtWidgets.QAction(MainWindow)
        self.actionCalculate_RM.setObjectName("actionCalculate_RM")
        self.actionCalculate_ORM = QtWidgets.QAction(MainWindow)
        self.actionCalculate_ORM.setObjectName("actionCalculate_ORM")
        self.actionAdaptive_Feedback = QtWidgets.QAction(MainWindow)
        self.actionAdaptive_Feedback.setObjectName("actionAdaptive_Feedback")
        self.actionLoad_GO_from_Orbit_Display = QtWidgets.QAction(MainWindow)
        self.actionLoad_GO_from_Orbit_Display.setObjectName("actionLoad_GO_from_Orbit_Display")
        self.actionSave_corrs = QtWidgets.QAction(MainWindow)
        self.actionSave_corrs.setObjectName("actionSave_corrs")
        self.actionLoad_corrs = QtWidgets.QAction(MainWindow)
        self.actionLoad_corrs.setObjectName("actionLoad_corrs")
        self.menu_File.addAction(self.action_Parameters)
        self.menu_File.addAction(self.actionCalculate_ORM)
        self.menu_File.addAction(self.actionCalculate_RM)
        self.menu_File.addAction(self.actionRead_BPMs_Corrs)
        self.menu_File.addAction(self.actionAdaptive_Feedback)
        self.menuGolden_Orbit.addAction(self.actionLoad_Golden_Orbit)
        self.menuGolden_Orbit.addAction(self.actionSave_Golden_Orbit)
        self.menuGolden_Orbit.addAction(self.actionLoad_GO_from_Orbit_Display)
        self.menuFile.addAction(self.actionSave_corrs)
        self.menuFile.addAction(self.actionLoad_corrs)
        self.menuBar.addAction(self.menu_File.menuAction())
        self.menuBar.addAction(self.menuGolden_Orbit.menuAction())
        self.menuBar.addAction(self.menuFile.menuAction())
        self.mainToolBar.addSeparator()

        self.retranslateUi(MainWindow)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "Twiss Monitor"))
        self.groupBox.setTitle(_translate("MainWindow", "Automatic Reading"))
        self.label_2.setText(_translate("MainWindow", "sec   "))
        self.label.setText(_translate("MainWindow", "Read Magnets from DOOCS every       "))
        self.label_3.setText(_translate("MainWindow", "Twiss Monitors       "))
        self.cb_design.setText(_translate("MainWindow", "Design"))
        self.cb_otrc55.setText(_translate("MainWindow", "OTRC 55"))
        self.cb_otrb218.setText(_translate("MainWindow", "OTRB 218"))
        self.cb_otrb450.setText(_translate("MainWindow", "OTRB 450"))
        self.pb_start_stop.setText(_translate("MainWindow", "Start Cyclic Reading"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Options"))
        self.pb_hide_show.setText(_translate("MainWindow", "Hide Table"))
        self.menu_File.setTitle(_translate("MainWindow", "&Expert Panel"))
        self.menuGolden_Orbit.setTitle(_translate("MainWindow", "Golden Orbit"))
        self.menuFile.setTitle(_translate("MainWindow", "Correctors"))
        self.action_Parameters.setText(_translate("MainWindow", "Parameters"))
        self.actionLoad_Golden_Orbit.setText(_translate("MainWindow", "Load Golden Orbit"))
        self.actionSave_Golden_Orbit.setText(_translate("MainWindow", "Save Golden Orbit"))
        self.actionRead_BPMs_Corrs.setText(_translate("MainWindow", "Read BPMs and Corrs"))
        self.actionCalculate_RM.setText(_translate("MainWindow", "Calculate ORM and DRM"))
        self.actionCalculate_ORM.setText(_translate("MainWindow", "Calculate ORM"))
        self.actionAdaptive_Feedback.setText(_translate("MainWindow", "Adaptive Feedback"))
        self.actionLoad_GO_from_Orbit_Display.setText(_translate("MainWindow", "Load GO from Orbit Display"))
        self.actionSave_corrs.setText(_translate("MainWindow", "Save"))
        self.actionLoad_corrs.setText(_translate("MainWindow", "Load"))

from gui.monitor.ocl_monitor import OclMonitor
from gui.table.v_ocl_table import VOclTable
