import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSpinBox
from PyQt5.QtWidgets import QDoubleSpinBox, QLineEdit, QPushButton
from PyQt5.QtWidgets import QGroupBox, QGridLayout
import os


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Collision Detection Program")
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Qroupbox
        self.body1 = QGroupBox("Body 1")
        self.body2 = QGroupBox("Body 2")
        self.simparam = QGroupBox("Simulation Parameters")
        self.layout.addWidget(self.body1, 0, 0)
        self.layout.addWidget(self.body2, 0, 1)
        self.layout.addWidget(self.simparam, 1, 0, 1, 2)

        self.grid1 = QGridLayout()
        self.grid2 = QGridLayout()
        self.grid3 = QGridLayout()

        self.body1.setLayout(self.grid1)
        self.body2.setLayout(self.grid2)
        self.simparam.setLayout(self.grid3)

        # body1

        self.grid1.addWidget(QLabel('CG Location (m):'), 0, 0)
        self.CG1 = QLineEdit('0.00 0.00 0.00')
        self.grid1.addWidget(self.CG1, 0, 1)

        self.grid1.addWidget(QLabel('Initial Velocity (m/s):'), 1, 0)
        self.vel1 = QLineEdit()
        self.vel1.setText('0.00 0.00 0.00')
        self.grid1.addWidget(self.vel1, 1, 1)

        self.grid1.addWidget(QLabel('Quaternion:'), 2, 0)
        self.quat1 = QLineEdit('1.00 0.00 0.00 0.00')
        self.grid1.addWidget(self.quat1, 2, 1)

        self.grid1.addWidget(QLabel('Angular velocity (rad/s):'), 3, 0)
        self.anvel1 = QLineEdit('0.00 0.00 -0.40')
        self.grid1.addWidget(self.anvel1, 3, 1)

        # body 2

        self.grid2.addWidget(QLabel('CG Location (m):'), 0, 0)
        self.CG2 = QLineEdit('-20.00 0.00 0.00')
        self.grid2.addWidget(self.CG2, 0, 1)

        self.grid2.addWidget(QLabel('Initial Velocity (m/s):'), 1, 0)
        self.vel2 = QLineEdit('0.81 0.00 0.00')
        self.grid2.addWidget(self.vel2, 1, 1)

        self.grid2.addWidget(QLabel('Quaternion:'), 2, 0)
        self.quat2 = QLineEdit('1.00 0.00 0.00 0.00')
        self.grid2.addWidget(self.quat2, 2, 1)

        self.grid2.addWidget(QLabel('Angular velocity (rad/s):'), 3, 0)
        self.anvel2 = QLineEdit('0.00 0.50 -0.00')
        self.grid2.addWidget(self.anvel2, 3, 1)

        # body 3
        self.grid3.addWidget(QLabel('Tmax'), 0, 0)
        self.Tmax = QDoubleSpinBox()
        self.Tmax.setValue(30.0)
        self.grid3.addWidget(self.Tmax, 0, 1)
        self.grid3.addWidget(QLabel('No. of Parallel process'), 0, 2)
        self.nprcs = QSpinBox()
        self.nprcs.setValue(4)
        self.grid3.addWidget(self.nprcs, 0, 3)

        # Run
        run = QPushButton('Run')
        self.layout.addWidget(run, 3, 0)
        run.clicked.connect(self.run)

    def run(self):

        # body1.txt
        file1 = open('Body1.txt', 'w')
        lines1 = [self.CG1.text(), self.vel1.text(), self.quat1.text(),
                  self.anvel1.text()]
        for line in lines1:
            file1.write(line)
            file1.write('\n')
        file1.close()

        # body2.txt
        file2 = open('Body2.txt', 'w')
        lines2 = [self.CG2.text(), self.vel2.text(), self.quat2.text(),
                  self.anvel2.text()]
        for line in lines2:
            file2.write(line)
            file2.write('\n')
        file2.close()

        # simparam.txt
        file3 = open('simparam.txt', 'w')
        lines3 = [str(self.Tmax.value()), str(self.nprcs.value())]
        for line in lines3:
            file3.write(line)
            file3.write('\n')
        file3.close()

        # Run the run.py file using the command below

        NPROC = self.nprcs.value()
        os.system(f'mpiexec -n {NPROC} python run.py')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    sys.exit(app.exec_())
