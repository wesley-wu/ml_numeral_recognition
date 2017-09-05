#!/usr/bin/env python


from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QImage, QImageWriter, QPainter, QPen, qRgb, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QPushButton, QHBoxLayout,
        QVBoxLayout, QDialog, QMessageBox, QWidget, QLabel, qApp)
import numpy as np

class ScribbleArea(QWidget):
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 1
        self.myPenColor = Qt.blue
        self.image = QImage()
        self.lastPoint = QPoint()

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    def paintEvent(self, event):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width(), self.image.width())
            newHeight = max(self.height(), self.image.height())
            self.resizeImage(self.image, QSize(newWidth, newHeight))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine,
                Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = self.myPenWidth / 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QImage(newSize, QImage.Format_RGB32)
        newImage.fill(qRgb(255, 255, 255))
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth


# Helper functions.
def ReLU(inputs):
    """
    Compute the ReLU: max(x, 0) nonlinearity.
    """
    return np.where(inputs < 0.0, 0.0, inputs)


def softmax(inputs):
    """
    Compute the softmax nonlinear activation function.
    """
    probs = np.exp(inputs)
    probs /= np.sum(probs, axis=1)[:, np.newaxis]
    return probs


def predict(probs):
    """
    Make predictions based on the model probability.
    """
    return np.argmax(probs, axis=1)


class RecognizeModel:
    def __init__(self):
        self.w1 = np.load("./params/w1.npy")
        self.b1 = np.load("./params/b1.npy")
        self.w2 = np.load("./params/w2.npy")
        self.b2 = np.load("./params/b2.npy")

    def forward(self, inputs):
        """
        Forward evaluation of the model.
        """
        h1 = ReLU(np.dot(inputs, self.w1) + self.b1)
        h2 = softmax(np.dot(h1, self.w2) + self.b2)
        return h1, h2


class MainWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.model = RecognizeModel()

        self.scribbleArea = ScribbleArea()
        self.scribbleArea.setPenWidth(4)
        self.scribbleArea.setPenColor(Qt.black)
        self.scribbleArea.setFixedSize(28 * self.scribbleArea.penWidth(), 28 * self.scribbleArea.penWidth())

        self.resultLabel = QLabel("")

        closeButton = QPushButton("Close")
        clearButton = QPushButton("Clear")
        recognizeButton = QPushButton("Recognize")

        vboxMain = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.scribbleArea)

        vbox = QVBoxLayout()
        vbox.addWidget(closeButton)
        vbox.addWidget(clearButton)
        vbox.addWidget(recognizeButton)
        hbox.addLayout(vbox)

        vboxMain.addLayout(hbox)
        vboxMain.addWidget(self.resultLabel)

        self.setLayout(vboxMain)

        self.setWindowTitle("Numeral Recognition")
        self.setFixedSize(self.sizeHint())

        closeButton.clicked.connect(qApp.exit)
        clearButton.clicked.connect(self.clear)
        recognizeButton.clicked.connect(self.recognize)

    def clear(self):
        self.scribbleArea.clearImage()
        self.resultLabel.clear()

    def recognize(self):
        image = self.scribbleArea.image
        image = image.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image = image.convertToFormat(QImage.Format_Grayscale8)

        buf = image.bits().asstring(image.byteCount())
        digitimg = np.frombuffer(buf, np.uint8).reshape(1, 784)
        digitimg = 255 - digitimg
        digitimg = digitimg.astype(np.float32)
        digitimg = np.multiply(digitimg, 1.0 / 255.0)

        _, probs = self.model.forward(digitimg)

        self.resultLabel.setText("The result of recognition is: <font color=red><b>%d</b></font>" % predict(probs)[0])


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
