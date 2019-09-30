import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        elif len(approx) == 7:
            shape = "heptagon"
        elif len(approx) == 8:
            shape = "octagon"
        elif len(approx) == 10:
            shape = "star"
        elif len(approx) == 12:
            shape = "cross"
        else:
            shape = "circle"

        return shape
