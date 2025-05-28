from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui     import QPainter, QColor, QPen, QFont
from PyQt6.QtCore    import Qt, QPoint, QRectF

class ShapesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._square_side = 80
        self.setMinimumSize(300, 300)

    def set_square_side(self, side: int):
        if side > 0 and side != self._square_side:
            self._square_side = side
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#cccccc"))
        p.setBrush(QColor("#666666"))
        p.setPen(QPen(Qt.GlobalColor.black))

        centers = [
            QPoint(80, 260), QPoint(180, 180),
            QPoint(90, 100), QPoint(260, 80),
            QPoint(260, 260),
        ]
        half = self._square_side / 2
        for i, c in enumerate(centers):
            r = QRectF(c.x() - half, c.y() - half,
                       self._square_side, self._square_side)
            p.drawRect(r)
            p.drawPoint(c)
            p.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            p.drawText(c + QPoint(5, -5), f"w{i+1}")
        p.end()
