import pygame
import numpy as np

pygame.init()

WIDTH = 800
HEIGHT = 600

drone_x = 400
drone_y = 300

desired_x = 600
desired_y = 500

desired_x = drone_x
desired_y = drone_y

# path = []

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Simulation")

clock = pygame.time.Clock()

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            desired_x, desired_y = pygame.mouse.get_pos()

    screen.fill((30,30,30))

    pygame.draw.circle(screen,(255,0,0),(int(desired_x),int(desired_y)),8)

    pygame.draw.circle(screen,(0,255,0),(int(drone_x),int(drone_y)),10)

    error_x = desired_x - drone_x
    error_y = desired_y - drone_y

    vx = 0.05 * error_x
    vy = 0.05 * error_y

    drone_x += vx
    drone_y += vy

    # path.append((int(drone_x), int(drone_y)))

    # for p in path:
    #     pygame.draw.circle(screen,(100,100,255),p,2)

    pygame.display.update()

    clock.tick(60)

pygame.quit()