import pygame
import random

# Constants
GRID_SIZE = 20
CELL_SIZE = 20
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

class Snake:
    def __init__(self, start_pos, color):
        self.body = [start_pos]
        self.direction = random.choice(DIRECTIONS)
        self.color = color

    def move(self):
        new_head = ((self.body[0][0] + self.direction[0]) % GRID_SIZE,
                    (self.body[0][1] + self.direction[1]) % GRID_SIZE)
        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        self.body.append(self.body[-1])

    def change_direction(self, new_direction):
        if (self.direction[0] + new_direction[0], self.direction[1] + new_direction[1]) != (0, 0):
            self.direction = new_direction

    def get_head(self):
        return self.body[0]

    def get_body(self):
        return self.body

def generate_food(snake1, snake2):
    while True:
        food_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if food_pos not in snake1.get_body() and food_pos not in snake2.get_body():
            return food_pos

def is_collision(snake1, snake2):
    return snake1.get_head() in snake2.get_body() or snake2.get_head() in snake1.get_body()

def draw_snake(screen, snake):
    for segment in snake.get_body():
        pygame.draw.rect(screen, snake.color, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_food(screen, food_pos):
    pygame.draw.rect(screen, (255, 0, 0), (food_pos[0] * CELL_SIZE, food_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def avoid_collision(snake, other_snake):
    potential_head = (snake.get_head()[0] + snake.direction[0], snake.get_head()[1] + snake.direction[1])
    if (potential_head in other_snake.get_body() or
        potential_head in snake.get_body()):
        for direction in DIRECTIONS:
            new_head = (snake.get_head()[0] + direction[0], snake.get_head()[1] + direction[1])
            if (new_head not in snake.get_body() and
                new_head not in other_snake.get_body()):
                snake.change_direction(direction)
                break

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()

    player_snake = Snake((5, 5), (0, 255, 0))
    ai_snake = Snake((15, 15), (0, 0, 255))
    food = generate_food(player_snake, ai_snake)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player_snake.change_direction(UP)
                elif event.key == pygame.K_DOWN:
                    player_snake.change_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    player_snake.change_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    player_snake.change_direction(RIGHT)

        avoid_collision(ai_snake, player_snake)

        player_snake.move()
        ai_snake.move()

        if is_collision(player_snake, ai_snake):
            print("Collision detected!")
            running = False

        if player_snake.get_head() == food:
            player_snake.grow()
            food = generate_food(player_snake, ai_snake)
        if ai_snake.get_head() == food:
            ai_snake.grow()
            food = generate_food(player_snake, ai_snake)

        head = ai_snake.get_head()
        if head[0] < food[0]:
            ai_snake.change_direction(RIGHT)
        elif head[0] > food[0]:
            ai_snake.change_direction(LEFT)
        elif head[1] < food[1]:
            ai_snake.change_direction(DOWN)
        elif head[1] > food[1]:
            ai_snake.change_direction(UP)

        screen.fill((0, 0, 0))
        draw_snake(screen, player_snake)
        draw_snake(screen, ai_snake)
        draw_food(screen, food)
        pygame.display.flip()

        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
