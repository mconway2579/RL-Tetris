import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import torch.nn.init as init
INPUT_WIDTH = 40
INPUT_HEIGHT = 176


class Tetris_ConvNet(nn.Module):
    def __init__(self, init_zero=False):
        super(Tetris_ConvNet, self).__init__()
        input_channels = 3
        img_height = 22
        img_width = 10
        num_classes = 4


        # Flatten the image into a single vector
        self.input_features = input_channels * img_height * img_width

        # Define three fully connected layers
        self.fc1 = nn.Linear(self.input_features, 128)  # Adjust size as needed
        self.fc2 = nn.Linear(128, 64)                   # Adjust size as needed
        self.fc3 = nn.Linear(64, num_classes)           # Ends with num_classes for classification

        if init_zero:
            # Initialize weights and biases to zero
            init.constant_(self.fc1.weight, 0)
            init.constant_(self.fc1.bias, 0)
            init.constant_(self.fc2.weight, 0)
            init.constant_(self.fc2.bias, 0)
            init.constant_(self.fc3.weight, 0)
            init.constant_(self.fc3.bias, 0)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input image
        #print(x.shape)
        x = x.reshape(x.size(0), -1)
        #print(x.shape)

        # Apply the fully connected layers with ReLU activations
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No ReLU after last layer, assuming you're classifying

        return x



if __name__ == "__main__":
    import gymnasium as gym
    from displays import showboard_and_weight, crop_board, stack_boards

    #env = gym.make('ALE/Tetris-v5', render_mode='rgb_array')
    env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', repeat_action_probability=0, frameskip = 16)

    env.reset()

    # Create an instance of the model
    model = Tetris_ConvNet()
    #model = DeepQNetwork()
    epochs = 10
    current_epoch = 0
    observation3, reward, done, info, *extra_info = env.step(0)
    observation2, reward, done, info, *extra_info = env.step(0)
    observation1, reward, done, info, *extra_info = env.step(0)


    past_board_3 = crop_board(observation3, display = False)
    past_board_2 = crop_board(observation2, display = False)
    current_board = crop_board(observation1, display = False)


    stacked_boards = stack_boards(past_board_3, past_board_2, current_board, display = True)

    while current_epoch < epochs:  # Replace this with your condition for continuing the simulation

        observation, reward, done, info, *extra_info = env.step(0)
        board = crop_board(observation, display = True)

        past_board_3 = past_board_2
        past_board_2 = current_board
        current_board = board
        stacked_boards = stack_boards(past_board_3, past_board_2, current_board, display = True)

        output = model(stacked_boards)

        showboard_and_weight(observation, output)
        if done:
            env.reset()
            current_epoch+=1

    # Don't forget to close the environment
    env.close()
