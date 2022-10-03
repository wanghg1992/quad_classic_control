import numpy as np

import pygame


class Gamepad:
    def __init__(self):
        pygame.init()

        # Loop until the user clicks the close button.
        done = False

        # Used to manage how fast the screen updates
        clock = pygame.time.Clock()

        # Initialize the joysticks
        pygame.joystick.init()

        self.gamepad_count = pygame.joystick.get_count()
        if self.gamepad_count > 0:
            self.joystick0 = pygame.joystick.Joystick(0)
            self.joystick0.init()
        else:
            print("Not found gamepad!")


class Receiver:
    def __init__(self):
        self.gamepad = Gamepad()
        self.body_pos_offset = np.matrix([0.] * 6).T
        self.body_vel = np.matrix([0.] * 6).T

    def step(self):
        if self.gamepad.gamepad_count > 0:
            for event in pygame.event.get():  # User did something
                # if event.type == pygame.JOYBUTTONDOWN:
                if self.gamepad.joystick0.get_hat(0)[1] > 0:
                    self.body_pos_offset[2] = self.body_pos_offset[2] + 0.005
                elif self.gamepad.joystick0.get_hat(0)[1] < 0:
                    self.body_pos_offset[2] = self.body_pos_offset[2] - 0.005
                self.body_vel[0] = -self.gamepad.joystick0.get_axis(4)
                self.body_vel[1] = -self.gamepad.joystick0.get_axis(3)
                self.body_vel[5] = -self.gamepad.joystick0.get_axis(0)
                # print("body pos offset: ", self.body_pos_offset.T)
                # print("body vel: ", receiver.body_vel.T)


if __name__ == '__main__':
    receiver = Receiver()
    for i in range(10000000):
        receiver.step()
