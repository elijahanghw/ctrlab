import pygame

class Joystick:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("No joystick detected.")
            return
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Joystick detected: {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        print(f"Number of hats: {self.joystick.get_numhats()}")
        
        self.roll = self.joystick.get_axis(0)
        self.pitch = self.joystick.get_axis(1)  
        self.throttle = self.joystick.get_axis(2)
        self.yaw = self.joystick.get_axis(3)
        
        # Initialize Pygame window
        self.screen_width, self.screen_height = 640, 480
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Joystick Control")
        
    def map_axis_to_screen_x(self, value, min_screen, max_screen):
        """Maps joystick X-axis (-1 to 1) to screen coordinates (left to right)"""
        return int((value + 1) / 2 * (max_screen - min_screen) + min_screen)

    def map_axis_to_screen_y(self, value, min_screen, max_screen):
        """Maps joystick Y-axis (-1 to 1) to screen coordinates (bottom to top)"""
        return int((1 - value) / 2 * (max_screen - min_screen) + min_screen)

    
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print("Quit key pressed. Exiting...")
                return False
            elif event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    self.roll = event.value
                if event.axis == 1:
                    self.pitch = event.value
                if event.axis == 2:
                    self.throttle = event.value
                if event.axis == 3:
                    self.yaw = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"Button {event.button} pressed")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"Button {event.button} released")
            elif event.type == pygame.JOYHATMOTION:
                print(f"Hat {event.hat} moved to {event.value}")
                
        # print(f"Throttle: {self.throttle}, Roll: {self.roll}, Pitch: {self.pitch}, Yaw: {self.yaw}")
        
        return True
    
    def draw(self):
        self.screen.fill((0, 0, 0))  # Clear screen

        # Convert joystick values (-1 to 1) to screen coordinates
        yaw_x = self.map_axis_to_screen_x(self.yaw, 0, self.screen_width // 2)
        throttle_y = self.map_axis_to_screen_y(self.throttle, 0, self.screen_height)

        roll_x = self.map_axis_to_screen_x(self.roll, self.screen_width // 2, self.screen_width)
        pitch_y = self.map_axis_to_screen_y(self.pitch, 0, self.screen_height)

        # Draw two dots representing joystick positions
        pygame.draw.circle(self.screen, (255, 0, 0), (yaw_x, throttle_y), 10)  # Left dot (Yaw, Throttle)
        pygame.draw.circle(self.screen, (0, 0, 255), (roll_x, pitch_y), 10)    # Right dot (Roll, Pitch)

        pygame.display.update()  # Refresh screen
    
    def cleanup(self):
        pygame.quit()

if __name__ == "__main__":
    joystick = Joystick()
    running = True
    
    while running:
        running = joystick.process_events()
        joystick.draw()  # Update the display with new dot positions
        
    joystick.cleanup()