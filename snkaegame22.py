import pygame
import random
import os  # Import os to check the current working directory

# Constants
GRID_SIZE = 30  # Increase the number of cells in the grid
CELL_SIZE = 20  # Size of each cell in pixels
SCREEN_SIZE = GRID_SIZE * CELL_SIZE  # Calculate the screen size based on the grid size

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Colors
BACKGROUND_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)

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
        self.body.append(self.body[-1])  # Add a new segment at the end

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

def draw_grid(screen):
    for x in range(0, SCREEN_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_SIZE))
    for y in range(0, SCREEN_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (SCREEN_SIZE, y))

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

def display_message(screen, message, size, color, position):
    font = pygame.font.Font(None, size)
    text = font.render(message, True, color)
    text_rect = text.get_rect(center=position)
    screen.blit(text, text_rect)

def start_menu(screen):
    screen.fill(BACKGROUND_COLOR)
    display_message(screen, "Snake Game", 72, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 - 100))
    display_message(screen, "Press 1 for Easy", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2))
    display_message(screen, "Press 2 for Medium", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 50))
    display_message(screen, "Press 3 for Hard", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 100))
    display_message(screen, "Press I for Instructions", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 150))
    pygame.display.flip()

    difficulty = None
    while difficulty is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    difficulty = 5
                elif event.key == pygame.K_2:
                    difficulty = 10
                elif event.key == pygame.K_3:
                    difficulty = 15
                elif event.key == pygame.K_i:  # Show instructions
                    show_instructions(screen)
    return difficulty

def show_instructions(screen):
    screen.fill(BACKGROUND_COLOR)
    display_message(screen, "Instructions", 72, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 - 100))
    display_message(screen, "Use arrow keys to control the snake.", 36, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2))
    display_message(screen, "Eat the red food to grow.", 36, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 50))
    display_message(screen, "Avoid colliding with yourself and the AI.", 36, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 100))
    display_message(screen, "Press any key to return to the menu.", 36, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 150))
    pygame.display.flip()

    waiting_for_key = True
    while waiting_for_key:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                waiting_for_key = False

def load_high_score():
    try:
        with open('high_score.txt', 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def save_high_score(score):
    with open('high_score.txt', 'w') as f:
        f.write(str(score))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()

    # Print the current working directory
    print("Current working directory:", os.getcwd())

    # Load sounds (make sure these files are in the same directory as your script)
    try:
        eat_sound = pygame.mixer.Sound('eat.wav')  # Add your eat sound file
        game_over_sound = pygame.mixer.Sound('game_over.wav')  # Add your game over sound file
    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'eat.wav' and 'game_over.wav' are in the same directory as this script.")
        pygame.quit()
        return

    max_score = load_high_score()  # Load the high score

    while True:
        difficulty = start_menu(screen)
        if difficulty is None:
            break

        player_snake = Snake((5, 5), (0, 255, 0))
        ai_snake = Snake((15, 15), (0, 0, 255))
        food = generate_food(player_snake, ai_snake)

        player_score = 0  # Reset player score for the new game
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        player_snake.change_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        player_snake.change_direction(DOWN)
                    elif event.key == pygame.K_LEFT:
                        player_snake.change_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        player_snake.change_direction(RIGHT)
                    elif event.key == pygame.K_ESCAPE:  # Check for the Esc key
                        running = False  # Exit the game loop

            avoid_collision(ai_snake, player_snake)

            player_snake.move()
            ai_snake.move()

            # AI Logic to eat food
            if ai_snake.get_head() == food:
                ai_snake.grow()  # Grow the AI snake if it eats food
                food = generate_food(player_snake, ai_snake)  # Generate new food
                eat_sound.play()  # Play eating sound

            # AI movement towards food
            head = ai_snake.get_head()
            if head[0] < food[0]:
                ai_snake.change_direction(RIGHT)
            elif head[0] > food[0]:
                ai_snake.change_direction(LEFT)
            elif head[1] < food[1]:
                ai_snake.change_direction(DOWN)
            elif head[1] > food[1]:
                ai_snake.change_direction(UP)

            if is_collision(player_snake, ai_snake):
                game_over_sound.play()
                running = False

            if player_snake.get_head() == food:
                player_snake.grow()
                food = generate_food(player_snake, ai_snake)
                eat_sound.play()
                player_score += 1  # Increment player score

            screen.fill(BACKGROUND_COLOR)
            draw_grid(screen)
            draw_snake(screen, player_snake)
            draw_snake(screen, ai_snake)
            draw_food(screen, food)

            # Display scores
            player_score = len(player_snake.get_body())
            ai_score = len(ai_snake.get_body())
            max_score = max(max_score, player_score, ai_score)
            display_message(screen, f"Player Score: {player_score}", 36, (255, 255, 255), (100, 20))
            display_message(screen, f"AI Score: {ai_score}", 36, (255, 255, 255), (300, 20))
            display_message(screen, f"Max Score: {max_score}", 36, (255, 255, 255), (500, 20))

            pygame.display.flip()
            clock.tick(difficulty)

        # Game over screen
        screen.fill(BACKGROUND_COLOR)
        display_message(screen, "Game Over!", 72, (255, 0, 0), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 - 50))
        display_message(screen, f"Your Score: {player_score}", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2))
        display_message(screen, f"Max Score: {max_score}", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 50))

        # Update high score if necessary
        if player_score > max_score:
            max_score = player_score
            save_high_score(max_score)  # Save new high score

        display_message(screen, "Press R to Restart", 48, (255, 255, 255), (SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 100))
        pygame.display.flip()

        # Wait for restart
        waiting_for_restart = True
        while waiting_for_restart:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting_for_restart = False

if __name__ == "__main__":
    main()
