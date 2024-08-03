import pygame
import numpy as np
from game import Game
from agent import Agent


# Initialize the game
game = Game(8)

# Initialize Pygame
pygame.init()
size = (game.size * 20, game.size * 20)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Snake Game")

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
yellow = (255, 255, 0)

# Define fonts
font_small = pygame.font.Font(None, 20)
font_large = pygame.font.Font(None, 50)

# Define clock
clock = pygame.time.Clock()


def draw_text(text, font, color, x, y):
    """Draw text on screen"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def draw_game_state():
    """Draw game states on screen"""
    screen.fill(black)
    world = game.get_world()
    for r in range(game.size):
        for c in range(game.size):
            if world[0][r][c] == 1:
                pygame.draw.rect(screen, yellow, (c * 20, r * 20, 20, 20))
            elif world[2][r][c] == 1:
                pygame.draw.rect(screen, red, (c * 20, r * 20, 20, 20))
            elif world[1][r][c] == 1:
                pygame.draw.rect(screen, green, (c * 20, r * 20, 20, 20))
            elif world[3][r][c] == 1:
                pygame.draw.rect(screen, blue, (c * 20, r * 20, 20, 20))
    draw_text("Score: {}".format(game.get_score()), font_small, white, 10, 10)
    pygame.display.update()


# Main game loop
agent = Agent(5, 4)
agent.load_model('model.pth')
agent.epsilon = 0


for i in range(100):
    game.reset()

    # Init last actions array
    # _last_states = []

    done = False
    while not done:
        # Get current _state of environment
        _state = game.get_world()
        # _last_states.append(_state)
        state = _state

        # Select and do an actions
        action = agent.select_action(state)
        done, reward = game.do_action(action)

        # Draw game states
        draw_game_state()

        # Set game speed
        clock.tick(10)

    print('Epoch {} score: {}'.format(i+1, game.get_score()))

# Quit Pygame
pygame.quit()
