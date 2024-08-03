import random
import numpy as np


def one_hot_encoding(value):
    binary = bin(value)
    encode = [int(v) for v in binary[2:]]
    while len(encode) < 4:
        encode.insert(0, 0)
    return encode


class Game:
    def __init__(self, size):
        """Initialize the game"""
        self.size = size
        self.score = 0
        self.player = None
        self.food = [0, 0]
        self.direction = 0
        self.health = 0

        self.reset()

    def reset(self):
        """Reset game world"""
        self.score = 0
        self.player = [[self.size // 2, self.size // 2] for _ in range(3)]
        self.health = len(self.player) * self.size * 2
        self.direction = random.randint(0, 3)
        self._place_food()

    def check_collision_body(self, point=None):
        """Check collision with body"""
        if point is None:
            point = self.player[0]
        if point in self.player[1:]:
            return True
        return False

    def check_collision_wall(self, point=None):
        """Check collision with wall"""
        if point is None:
            point = self.player[0]
        if point[0] <= 0 or point[0] >= self.size - 1 or point[1] <= 0 or point[1] >= self.size - 1:
            return True
        return False

    def check_collision(self, point=None):
        """Check if position is not collision"""
        if point is None:
            point = self.player[0]
        if self.check_collision_body(point) or self.check_collision_wall(point):
            return True
        return False

    def check_food(self):
        """Check if in position there is food"""
        return True if self.player[0] == self.food else False

    def _eat(self):
        """Snake eat"""
        self.player.append(self.player[-1])
        self.health = len(self.player) * self.size * 2
        self.score += 1

    def _place_food(self):
        """Place food in random place out of collision"""
        self.food = [random.randint(1, self.size - 2) for _ in range(2)]
        while self.food in self.player:
            self.food = [random.randint(1, self.size - 2) for _ in range(2)]

    def do_action(self, action):
        """Do actions in the game"""
        if action == 0 and self.direction != 2:
            self.direction = 0
        elif action == 1 and self.direction != 3:
            self.direction = 1
        elif action == 2 and self.direction != 0:
            self.direction = 2
        elif action == 3 and self.direction != 1:
            self.direction = 3

        # Move body
        for i in range(len(self.player) - 1, 0, -1):
            self.player[i] = self.player[i - 1].copy()

        # Move head
        if self.direction == 0:
            self.player[0][0] += 1
        elif self.direction == 1:
            self.player[0][1] -= 1
        elif self.direction == 2:
            self.player[0][0] -= 1
        elif self.direction == 3:
            self.player[0][1] += 1

        # Reduce snake health
        self.health -= 1

        if self.check_collision() or self.health <= 0:
            # Dead if collision
            return True, -50
        else:
            # Check if food
            if self.check_food():
                self._eat()
                self._place_food()
                return False, 1

        # Nothing
        return False, -0.05

    def get_state(self):
        """Return actual game state"""
        state = [
            # Snake direction
            1 if self.direction == 1 else 0,
            1 if self.direction == 2 else 0,
            1 if self.direction == 3 else 0,
            1 if self.direction == 4 else 0,

            # Food direction
            1 if self.food[0] < self.player[0][0] else 0,
            1 if self.food[0] > self.player[0][0] else 0,
            1 if self.food[0] == self.player[0][0] else 0,
            1 if self.food[1] < self.player[0][1] else 0,
            1 if self.food[1] > self.player[0][1] else 0,
            1 if self.food[1] == self.player[0][1] else 0,

            # # Collision Wall
            # 1 if self.check_collision_wall([self.player[0][0] + 1, self.player[0][1]]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0] - 1, self.player[0][1]]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0], self.player[0][1] + 1]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0], self.player[0][1] - 1]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0] + 1, self.player[0][1] + 1]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0] - 1, self.player[0][1] - 1]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0] - 1, self.player[0][1] + 1]) else 0,
            # 1 if self.check_collision_wall([self.player[0][0] + 1, self.player[0][1] - 1]) else 0,
            #
            # # Collision body
            #  if self.check_collision_body([self.player[0][0] + 1, self.player[0][1]]) else 0,
            # 1 if self.check_collision_body([self.player[0][0] - 1, self.player[0][1]]) else 0,
            # 1 if self.check_collision_body([self.player[0][0], self.player[0][1] + 1]) else 0,
            # 1 if self.check_collision_body([self.player[0][0], self.player[0][1] - 1]) else 0,
            # 1 if self.check_collision_body([self.player[0][0] + 1, self.player[0][1] + 1]) else 0,
            # 1 if self.check_collision_body([self.player[0][0] - 1, self.player[0][1] - 1]) else 0,
            # 1 if self.check_collision_body([self.player[0][0] - 1, self.player[0][1] + 1]) else 0,
            # 1 if self.check_collision_body([self.player[0][0] + 1, self.player[0][1] - 1]) else 0,
        ]

        # Elements around the head including the head
        for c in range(self.player[0][0] - 1, self.player[0][0] + 2):
            for r in range(self.player[0][1] - 1, self.player[0][1] + 2):
                if [r, c] == self.player[0]:
                    state.extend([1, 0, 0, 0, 0])
                elif [r, c] in self.player[1:]:
                    state.extend([0, 1, 0, 0, 0])
                elif [r, c] == self.food:
                    state.extend([0, 0, 1, 0, 0])
                elif self.check_collision([r, c]):
                    state.extend([0, 0, 0, 1, 0])
                else:
                    state.extend([0, 0, 0, 0, 1])

        return np.array(state)

    def get_world(self):
        """Return world states"""
        world = []
        for r in range(self.size):
            world.append([])
            for c in range(self.size):
                if [r, c] == self.player[0]:
                    world[r].append([1, 0, 0, 0, 0])
                elif [r, c] in self.player[1:]:
                    world[r].append([0, self.player.index([r, c]), 0, 0, 0])
                elif [r, c] == self.food:
                    world[r].append([0, 0, 1, 0, 0])
                elif self.check_collision([r, c]):
                    world[r].append([0, 0, 0, 1, 0])
                else:
                    world[r].append([0, 0, 0, 0, 1])
        world = np.array(world)
        return np.transpose(world, (2, 0, 1))

    def print_world(self):
        """Print game in console"""
        for r in range(self.size):
            for c in range(self.size):
                e = '·'
                if [c, r] in self.player:
                    e = 'O'
                elif [c, r] == self.food:
                    e = 'X'
                print(e, end=' ')
            print()
        print()

    def get_score(self):
        """Return score did"""
        return self.score
