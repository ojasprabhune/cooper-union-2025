import random

class Ball:
    def __init__(self, position, circle):
        self.position = position 
        self.circle = circle 
        self.velocity_x = random.uniform(0, 0.05)
        self.velocity_y = random.uniform(0, 0.05)
        self.direction_x = random.choice([-1, 1])
        self.direction_y = random.choice([-1, 1])
        self.colliding = False
