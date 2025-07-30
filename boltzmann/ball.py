import random
import numpy as np

# fix inconsistent printing format
np.set_printoptions(formatter={'float': '{:.8f}'.format}, suppress=True)

class Ball:
    def __init__(self, position, circle):
        vx = random.uniform(0, 0.025)
        vy = random.uniform(0, 0.025)
        dx = random.choice([-1, 1])
        dy = random.choice([-1, 1])

        self.position = np.array(position)
        self.circle = circle 
        self.velocity = np.array([0.01 * dx, 0.01 * dy])
        self.colliding = False
