import cv2

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import itertools


def put_text_on_image(image, text, position=None, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 0, 0), thickness=1, line_type=cv2.LINE_AA):
    """
    Writes text on an image and returns the modified image.

    Parameters:
    - image: The image on which to write text, as a NumPy array.
    - text: The text string to write on the image.
    - position: The bottom-left corner of the text string in the image (tuple).
    - font_face: The font type (default=cv2.FONT_HERSHEY_SIMPLEX).
    - font_scale: The font scale factor that is multiplied by the font-specific base size (default=1).
    - color: The color of the text in BGR format (default=(255, 0, 0)).
    - thickness: The thickness of the lines used to draw the text (default=2).
    - line_type: The type of the line, cv2.LINE_AA for anti-aliased (default).

    Returns:
    The image with the text written on it.
    """
    # Use cv2.putText() to add text to the image
    cv2.putText(image, text, position, font_face, font_scale, color, thickness, line_type)

    return image

def find_first_white_pixel(image):
    """
    Find the first white pixel in the image.

    :param image: NumPy array of the image, expected to be in RGB format.
    :return: y-coordinate of the first white pixel, or None if no white pixel is found.
    """
    for y in range(image.shape[0]):  # Iterate over rows
        for x in range(image.shape[1]):  # Iterate over columns
            if np.array_equal(image[y, x], [255, 255, 255]):  # Check for white pixel
                #print(y)
                return y  # Return the y-coordinate of the first white pixel

    return image.shape[0]-1  # Return None if no white pixel is found

def crop_board(observation, display = False):
    x1, y1 = 24, 27  # Example coordinates
    x2, y2 = 64, 203
    image_width = x2-x1
    image_height = y2-y1
    block_width = 3
    block_height = 7
    spacing = 1

    # Calculate the number of blocks and the new dimensions
    num_blocks_horizontal = image_width // (block_width + spacing)
    num_blocks_vertical = image_height // (block_height + spacing)



    # Crop the image
    # Note: NumPy indexing is row-major, so the y-coordinates come first
    board = observation[y1:y2, x1:x2]
    masked_board = board.copy()
    mask = np.all(masked_board == [111, 111, 111], axis=-1)

    # Set those pixels to black ([0, 0, 0])
    masked_board[mask] = [0, 0, 0]
    inverse_mask = ~mask

    # Set those pixels to white ([255, 255, 255])
    masked_board[inverse_mask] = [255, 255, 255]

    grey_masked_board = cv2.cvtColor(masked_board, cv2.COLOR_BGR2GRAY)


    # Calculate the new image dimensions
    reduced_image_width = num_blocks_horizontal
    reduced_image_height = num_blocks_vertical
    reduced_image = np.zeros((reduced_image_height, reduced_image_width))

    for i in range(int(0.5*block_width), image_width, block_width+spacing):
        for j in range(int(0.5*block_height), image_height, block_height+spacing):
            x = i //  (block_width + spacing)
            y = j //  (block_height + spacing)
            #print(f"y = {j} -> {y}")
            #print(f"x = {i} -> {x}")
            if grey_masked_board[j, i] == 255:
                reduced_image[y,x] = 255
            grey_masked_board[j, i] = 128

    #cv2.imshow(f"reduced_image", reduced_image)
    #cv2.imshow(f"grey_masked_board", grey_masked_board)
    #print(grey_masked_board.shape)
    #cv2.waitKey(0)
    reduced_image = np.expand_dims(reduced_image, axis=-1)


    reduced_image_transpose = reduced_image.transpose(2, 0, 1)
    reduced_image_tensor = torch.from_numpy(reduced_image_transpose).float()

    if display:
        img = reduced_image_tensor.squeeze(0).numpy()
        cv2.imshow('board', img)
        #print(output_list)
        cv2.waitKey(1)  # Wait for the specified time
    return reduced_image_tensor

def showboard_and_weight(observation, model_outputs = None, extra_text = []):
    output_str = ""
    if model_outputs is not None:
        output_list = model_outputs.tolist()
        output_list = [f"{x:.1f}" for x in output_list[0]]
        output_str = ", ".join(output_list)

    new_dimensions = (1080, 720)

    # Resize the image
    resized_image = cv2.resize(observation, new_dimensions, interpolation=cv2.INTER_LINEAR)
    labeled_image = put_text_on_image(resized_image, output_str, (450,300))
    cv2.imshow('observation', observation)
    #print(output_list)
    cv2.waitKey(1)  # Wait for the specified time

def stack_boards(past_past_board, past_board, current_board, display = True):
    #print(f"past_past_board:{past_past_board.shape}")
    #print(f"past_board:{past_board.shape}")
    #print(f"current_board:{current_board.shape}")

    stacked_tensor = torch.cat((past_past_board.unsqueeze(0), past_board.unsqueeze(0), current_board.unsqueeze(0)), dim=1)
    #print(f"stacked_tensor: {stacked_tensor.shape}")

    image_np = stacked_tensor.squeeze().numpy().transpose(1, 2, 0)


    image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
    """
    x_grid_interval = 4
    y_grid_interval = 8
    # Draw the vertical lines of the grid
    for x in range(3, image_np.shape[1], x_grid_interval):
        cv2.line(image_np, (x, 0), (x, image_np.shape[0]), color=(100, 100, 100), thickness=1)

    # Draw the horizontal lines of the grid
    for y in range(0, image_np.shape[0], y_grid_interval):
        cv2.line(image_np, (0, y), (image_np.shape[1], y), color=(100, 100, 100), thickness=1)

    #draw height line
    y = find_first_white_pixel(image_np)
    pen = image_np.shape[0] - y
    pen = pen // 16
    color = [0, 255*(1-(pen/9)), 255*(pen/9)]
    #print(f"{y} : {color}")
    image_np[y, :] = color

    """
    image_np = image_np.astype(np.float32) / 255.0
    # Convert the numpy array back to a tensor
    image_tensor = torch.from_numpy(image_np)

    # Permute the dimensions from HWC to CHW format expected by PyTorch
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    #print(f"image_tensor: {image_tensor.shape}")

    #print(f"added height line")

    # Display the image using OpenCV
    #print(image_np.shape)
    if display:
        cv2.imshow('stacked image', image_np)
        cv2.waitKey(1)  # Wait for a key press to close the window
    #print(f"stacked_tensor shape: {stacked_tensor.shape}")
    return image_tensor

class DataAnimator:
    def __init__(self, title="", y_label=""):
        self.title = title
        self.y_label = y_label
        self.fig, self.ax = plt.subplots()
        self.y = []
        self.line, = self.ax.plot([], [], 'r-')  # Initialize a line plot

        self.ani = FuncAnimation(self.fig, self.update, frames=100000, init_func=self.init_graph, interval=1)
        plt.show(block = False)
    def init_graph(self):
        self.ax.clear()
        self.ax.set_title(self.title)
        self.ax.set_xlabel("episode")
        self.ax.set_ylabel(self.y_label)
        # Set initial plot parameters (e.g., axis limits, titles)
        # Typically no drawing, just setting the stage.
    def update(self, frame):
        if len(self.y) == 0:  # Avoids errors if y is empty
            return self.line,
        self.ax.clear()
        x_values = range(len(self.y))
        self.ax.set_xlim(0, len(self.y) + 1)  # Update x limits to fit the data
        self.ax.set_ylim(min(self.y), max(self.y)+1)  # Update y limits to fit the data
        self.line, = self.ax.plot(x_values, self.y, 'r-')  # Redraw the line
        return self.line,

    def add_data(self, new_y):
        self.y.append(new_y)
        plt.pause(0.05)


if __name__ == "__main__":
    import gymnasium as gym
    import torch
    import random

    import time
    import matplotlib
    matplotlib.use('TkAgg')

    #env = gym.make('ALE/Tetris-v5', render_mode='rgb_array')
    env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', repeat_action_probability=0, frameskip = 8)
    env.reset()
    total_epochs = 100
    #cr_animator = DataAnimator(title="cumulative_reward", y_label = "cr")
    #ns_animator = DataAnimator(title="num_steps", y_label="ns")


    steps_between_tupdate = 4

    for current_epoch in range(total_epochs):
        steps = 0

        observation3, reward, done, info, *extra_info = env.step(0)
        board3 = crop_board(observation3, display = False)
        steps +=1

        for step in range(steps_between_tupdate-1):
            env.step(0)
            steps += 1

        observation2, reward, done, info, *extra_info = env.step(0)
        board2 = crop_board(observation2, display = False)
        steps += 1

        while not done:
            a = random.randint(0,4)
            observation, reward, done, info, *extra_info = env.step(a)
            steps +=1

            board = crop_board(observation, display = True)
            showboard_and_weight(observation)
            stacked_boards = stack_boards(board3, board2, board, display = True)

            print
            if steps % steps_between_tupdate == 0:
                board3 = board2
                board2 = board
            #print(board.shape)
            if done:
                env.reset()

        #cr_animator.add_data(random.randint(0,100))
        #ns_animator.add_data(random.randint(0,100))
