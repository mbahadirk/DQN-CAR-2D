import pygame


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, win):
        pygame.draw.line(win, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 5)


def getWalls():
    walls = []

    wall1 = Wall(12, 451, 15, 130)
    wall2 = Wall(15, 130, 61, 58)
    wall3 = Wall(61, 58, 149, 14)
    wall4 = Wall(149, 14, 382, 20)
    wall5 = Wall(382, 20, 549, 31)
    wall6 = Wall(549, 31, 636, 58)
    wall7 = Wall(636, 58, 678, 102)
    wall8 = Wall(678, 102, 669, 167)
    wall9 = Wall(669, 167, 600, 206)
    wall10 = Wall(600, 206, 507, 214)
    wall11 = Wall(507, 214, 422, 232)
    wall12 = Wall(422, 232, 375, 263)
    wall13 = Wall(375, 263, 379, 283)
    wall14 = Wall(379, 283, 454, 299)
    wall15 = Wall(454, 299, 613, 286)
    wall16 = Wall(613, 286, 684, 238)
    wall17 = Wall(684, 238, 752, 180)
    wall18 = Wall(752, 180, 862, 185)
    wall19 = Wall(862, 185, 958, 279)
    wall20 = Wall(958, 279, 953, 410)
    wall21 = Wall(953, 410, 925, 505)
    wall22 = Wall(925, 505, 804, 566)
    wall23 = Wall(804, 566, 150, 570)
    wall24 = Wall(150, 570, 46, 529)
    wall25 = Wall(46, 529, 12, 451)
    wall27 = Wall(104, 436, 96, 161)
    wall28 = Wall(96, 161, 122, 122)
    wall29 = Wall(122, 122, 199, 91)
    wall30 = Wall(199, 91, 376, 94)
    wall31 = Wall(376, 94, 469, 100)
    wall32 = Wall(469, 100, 539, 102)
    wall33 = Wall(539, 102, 585, 121)
    wall34 = Wall(585, 121, 585, 139)
    wall35 = Wall(585, 139, 454, 158)
    wall36 = Wall(454, 158, 352, 183)
    wall37 = Wall(352, 183, 293, 239)
    wall38 = Wall(293, 239, 294, 318)
    wall39 = Wall(294, 318, 361, 357)
    wall40 = Wall(361, 357, 490, 373)
    wall41 = Wall(490, 373, 671, 359)
    wall42 = Wall(671, 359, 752, 300)  #
    wall43 = Wall(752, 300, 812, 310)  #
    wall44 = Wall(812, 310, 854, 369)
    wall45 = Wall(854, 369, 854, 429)
    wall46 = Wall(854, 429, 754, 483)
    wall47 = Wall(754, 483, 192, 489)
    wall48 = Wall(192, 489, 104, 436)

    walls.append(wall1)
    walls.append(wall2)
    walls.append(wall3)
    walls.append(wall4)
    walls.append(wall5)
    walls.append(wall6)
    walls.append(wall7)
    walls.append(wall8)
    walls.append(wall9)
    walls.append(wall10)
    walls.append(wall11)
    walls.append(wall12)
    walls.append(wall13)
    walls.append(wall14)
    walls.append(wall15)
    walls.append(wall16)
    walls.append(wall17)
    walls.append(wall18)
    walls.append(wall19)
    walls.append(wall20)
    walls.append(wall21)
    walls.append(wall22)
    walls.append(wall23)
    walls.append(wall24)
    walls.append(wall25)

    walls.append(wall27)
    walls.append(wall28)
    walls.append(wall29)
    walls.append(wall30)
    walls.append(wall31)
    walls.append(wall32)
    walls.append(wall33)
    walls.append(wall34)
    walls.append(wall35)
    walls.append(wall36)
    walls.append(wall37)
    walls.append(wall38)
    walls.append(wall39)
    walls.append(wall40)
    walls.append(wall41)
    walls.append(wall42)
    walls.append(wall43)
    walls.append(wall44)
    walls.append(wall45)
    walls.append(wall46)
    walls.append(wall47)
    walls.append(wall48)

    return (walls)

if __name__ == "__main__":
    walls = getWalls()
    for wall in walls:
        print(f"Duvar Koordinatları: ({wall.x1}, {wall.y1}) -> ({wall.x2}, {wall.y2})")