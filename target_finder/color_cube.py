from math import sqrt


class ColorCube:
    def __init__(self, color1, color2, using_delta=False):

        if (
            not using_delta
            and color1[0] < 0
            or color1[0] >= 360
            or color1[1] < 0
            or color1[1] > 100
            or color1[2] < 0
            or color1[2] > 100
            or color2[0] < 0
            or color2[0] >= 360
            or color2[1] < 0
            or color2[1] > 100
            or color2[2] < 0
            or color2[2] > 100
        ):
            raise ValueError("Invalid input 0 <= H <= 359 and 0 <= S,V <= 100")

        self.hStart = color1[0]
        self.sStart = color1[1]
        self.vStart = color1[2]
        if using_delta:
            self.hEnd = color1[0] + color2[0]
            self.sEnd = color1[1] + color2[1]
            self.vEnd = color1[2] + color2[2]
        else:
            self.hEnd = color2[0]
            self.sEnd = color2[1]
            self.vEnd = color2[2]

    def displayHSV(self):
        print(self.hStart, self.sStart, self.vStart)

    def displayDeltas(self):
        print(
            self.hEnd - self.hStart, self.sEnd - self.sStart, self.vEnd - self.vStart,
        )

    def contains(self, color):
        h = color[0]
        s = color[1]
        v = color[2]
        if (
            self.hStart <= h <= self.hEnd
            and self.sStart <= s <= self.sEnd
            and self.vStart <= v <= self.vEnd
        ):
            return True
        else:
            return False

    def get_distance(self, point1, point2):
        return sqrt(
            (point1[0] - point2[0]) ** 2
            + (point1[1] - point2[1]) ** 2
            + (point1[2] - point2[2]) ** 2
        )

    def get_closest_distance(self, color):
        closestDistance = 385.851
        closestPoint = (0, 0, 0)
        hS = self.hStart
        hE = self.hEnd
        sS = self.sStart
        sE = self.sEnd
        vS = self.vStart
        vE = self.vEnd
        for h in range(hS, hE + 1):
            s, v = sE, vS
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            s, v = sS, vS
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            s, v = sS, vE
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            s, v = sE, vE
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

        for s in range(sS, sE + 1):

            h, v = hS, vS
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            h, v = hE, vS
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            h, v = hS, vE
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            h, v = hE, vE
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

        for v in range(vS, vE + 1):

            h, s = hS, sS
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            h, s = hS, sE
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            h, s = hE, sS
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

            h, s = hE, sE
            currentDistance = self.get_distance((h, s, v), color)
            if currentDistance < closestDistance:
                closestDistance = currentDistance
                closestPoint = (h, s, v)

        return closestPoint
