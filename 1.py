import pygame,sys,pymunk

def create_apple(space):
    body = pymunk.Body(1,100,body_type = pymunk.Body.DYNAMIC)
    body.position = (400,0)
    shape = pymunk.Circle(body,80)
    space.add(body,shape)
    return shape

def draw_apples(apples):
    for apple in apples:
        pygame.draw.Circle(screen,(0,0,0),apple.body.position)

pygame.init()
screen = pygame.display.set_mode((800,800))
clock = pygame.time.Clock()
space = pymunk.Space()
space.gravity = (0,500)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((217,217,217))
    space.step(1/50)
    pygame.display.update()
    clock.tick(120)
