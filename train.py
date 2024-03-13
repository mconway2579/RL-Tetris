import gymnasium as gym
import cv2
import time
from Model import Tetris_ConvNet
from displays import showboard_and_weight, crop_board, find_first_white_pixel, stack_boards, DataAnimator

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import random


#https://gymnasium.farama.org/environments/atari/tetris/
#Observations are (210,160,3)
#http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
class Model_Trainer:
    def __init__(self, load_path = None):
        print(f"cuda? {torch.cuda.is_available()}")

        self.player_controlled = False
        self.replay_memory = []

        self.device = torch.device("cuda:0")

        self.model = Tetris_ConvNet()
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
            print(f"loaded model: {load_path}")
        self.model.to(self.device)
        self.model.train()

        self.cached_model = Tetris_ConvNet(init_zero = True)
        self.cached_model.to(self.device)

        #self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()

        learning_rate = 0.001
        betas = (0.9, 0.999)  # These are the default values in PyTorch
        eps = 1e-08  # This is the default value in PyTorch
        weight_decay = 0  # Default is no weight decay

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

        self.learning_rate = 0.8
        self.gamma = 0.99
        self.batch_size = 512

    def update_cached_model(self):
        self.cached_model.load_state_dict(self.model.state_dict())

    def predict(self, board, use_cached = False):
        board = board.to(self.device)
        output = None
        if use_cached:
            output = self.cached_model(board)
        else:
            output = self.model(board)
        chosen_action = torch.argmax(output)
        return output, chosen_action

    def batch_update_model(self, old_board_batch, new_board_batch, actions_batch, rewards_batch, do_print=False):

        # Predict the Q-values for the old states
        old_state_q_values = self.predict(old_board_batch)[0]

        # Predict the future Q-values from the next states using the target network
        next_state_q_values = self.predict(new_board_batch, use_cached=True)[0]

        # Clone the old Q-values to use as targets for loss calculation
        target_q_values = old_state_q_values.clone()

        # Ensure that actions and rewards are tensors
        actions_batch = actions_batch.long()
        rewards_batch = rewards_batch.float()

        # Update the Q-value for each action taken
        batch_index = torch.arange(old_state_q_values.size(0), device=self.device)  # Ensuring device consistency
        max_future_q_values = next_state_q_values.max(1)[0]
        target_values = rewards_batch + self.gamma * max_future_q_values
        target_q_values[batch_index, actions_batch] = target_values

        # Calculate the loss
        loss = self.loss_fn(old_state_q_values, target_q_values)

        # Logging for debugging
        if do_print:
            print(f"\n")
            print(f"   action: {actions_batch[0]}")
            print(f"   reward: {rewards_batch[0]}")
            print(f"   old_board_batch.shape: {old_board_batch.shape}")
            print(f"   new_board_batch.shape: {new_board_batch.shape}")
            print(f"   old_state_q_values: {old_state_q_values[0]}")
            print(f"   next_state_q_values: {next_state_q_values[0]}")
            print(f"   target_q_values: {target_q_values[0]}")
            print(f"   loss: {loss}\n")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss



    def train_replay_memory(self):
        cv2.destroyAllWindows()

        print(f"training replay memory")
        print()

        n_epochs = 10
        old_boards = []
        new_boards = []
        rewards = []
        actions = []

        for e in range(n_epochs):
            random.shuffle(self.replay_memory)
            for memory in self.replay_memory:
                state, new_state, chosen_action, reward = memory
                old_boards.append(state)
                new_boards.append(new_state)
                actions.append(chosen_action)
                rewards.append(reward)

            old_boards_all = torch.cat(old_boards, dim=0)
            old_boards_batches = torch.split(old_boards_all, self.batch_size, dim=0)

            new_boards_all = torch.cat(new_boards, dim=0)
            new_board_batches = torch.split(new_boards_all, self.batch_size, dim=0)



            rewards_all = torch.tensor(rewards, device=self.device)
            rewards_batches = torch.split(rewards_all, self.batch_size, dim=0)

            actions_all = torch.tensor(actions, device=self.device)
            actions_batches = torch.split(actions_all, self.batch_size, dim=0)

            if e == 0:
                print(f"old_boards_all:{old_boards_all.shape}, n_splits: {len(old_boards_batches)}, split_dim: {old_boards_batches[0].shape}")
                print(f"new_boards_all:{new_boards_all.shape}, n_splits: {len(new_board_batches)}, split_dim: {new_board_batches[0].shape}")
                print(f"rewards_all:{rewards_all.shape}, n_splits: {len(rewards_batches)}, split_dim: {rewards_batches[0].shape}")
                print(f"actions_all:{actions_all.shape}, n_splits: {len(actions_batches)}, split_dim: {actions_batches[0].shape}")

            for b in range(len(old_boards_batches)):
                p = b== 0 and (e == 0 or e == n_epochs-1)
                if p:
                    print(f"epoch: {e}")
                self.batch_update_model(old_boards_batches[b], new_board_batches[b], actions_batches[b], rewards_batches[b], do_print = p)


        self.replay_memory = []

def reward_func(t_board):
    image_np = t_board.squeeze().numpy().transpose(1, 2, 0)

    y = find_first_white_pixel(image_np)
    #print(f"reward func called height = {y}")
    #print(board.shape)
    #print(f"pen = {board.shape[0]} - {y}")

    if y is None:
        return 0
    pen = image_np.shape[0] - y
    pen *= -1
    pen = pen // 16
    #print(f"y = {y}, pen = {pen}")
    return pen

def episode(env, model_trainer, wait_time, epsilon, train = True):
    random_seed = np.random.randint(0, 10000)
    #print(f"starting episode with seed: {random_seed}")
    env.reset(seed = random_seed)
    #env.seed(random_seed)
    rs = np.random.RandomState(random_seed)
    #print(f"   random_state: {rs}")
    env.np_random =  rs
    #print(f"   env_seed: {env.np_random}")
    cumulative_reward = 0
    lines_cleared = 0
    steps = 0
    epoch_start_time = time.time()

    observation3, reward, done, info, *extra_info = env.step(0)
    observation2, reward, done, info, *extra_info = env.step(0)


    past_board_3 = crop_board(observation3, display = False)

    past_board_2 = crop_board(observation2, display = False)

    cur_observation, reward, done, info, *extra_info = env.step(0)

    while not done:
        cur_board = crop_board(cur_observation, display = False)
        output, chosen_action = None, None

        cur_temporal_img = None
        new_temporal_img = None

        cur_temporal_img = stack_boards(past_board_3, past_board_2, cur_board, display = True)


        random_float = np.random.rand()
        if model_trainer.player_controlled:
            key = cv2.waitKey(0) & 0xFF  # Wait for the specified time
            chosen_action = 0 #NOOP
            if key == ord(' '):
                chosen_action = 1 #FIRE
            if key == ord('d'):
                chosen_action = 2 #RIGHT
            if key == ord('a'):
                chosen_action = 3 #LEFT
        elif random_float < epsilon:
            chosen_action = np.random.randint(0, 4)
        else:
            output, chosen_action = model_trainer.predict(cur_temporal_img)

        #showboard_and_weight(cur_observation, output)

        new_observation, reward, done, info, *extra_info = env.step(chosen_action)

        steps += 1
        new_board = crop_board(new_observation, display = False)
        new_temporal_img = stack_boards(past_board_2, cur_board, new_board, display = False)

        lines_cleared += reward
        reward += (reward*100)
        #reward += steps/100
        #reward += reward_func(new_temporal_img)

        if done:
            reward += -10
            #pass

        cumulative_reward += reward

        if epsilon != 0 and (reward != 0 or random.random() > 0.66):
            #model_trainer.update_model(cur_temporal_img, new_temporal_img, chosen_action, reward)
            memory = (cur_temporal_img, new_temporal_img, chosen_action, reward)
            model_trainer.replay_memory.append(memory)



        past_board_3 = past_board_2
        past_board_2 = cur_board
        cur_observation = new_observation


    epoch_end_time = time.time()
    duration = round(epoch_end_time - epoch_start_time, 2)
    print(f"{duration} epoch: {current_epoch}/{total_epochs}\n   cumulative_reward:{cumulative_reward}\n   steps:{steps}\n   lines_cleared:{lines_cleared}\n   epsilon:{epsilon:.6f}")

    return cumulative_reward/steps, steps


if __name__ == "__main__":
    # Create your environment
    random_seed = np.random.randint(0, 10000)
    env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', repeat_action_probability=0, frameskip = 8)
    print(dir(env))

    wait_time = 1 #33
    total_epochs = 10
    nine_tenths_total_epochs = total_epochs*0.9
    current_epoch = 0

    best_model_path = 'best_model_state_dict.pth'
    latest_model_path = 'latest_model_state_dict.pth'

    model_trainer = Model_Trainer(load_path = None)

    epislon_func = lambda e: max((-0.8*(e/nine_tenths_total_epochs))+0.9, 0.1)

    avgR_animator = DataAnimator(title="reward over time", y_label="average reward")




    best_model_avgR = -np.inf
    while current_epoch < total_epochs:

        if current_epoch % 30 == 0 and current_epoch != 0:
            model_trainer.train_replay_memory()
            model_trainer.update_cached_model()
            torch.save(model_trainer.model.state_dict(), latest_model_path)

        avg_reward, steps = episode(env, model_trainer, wait_time, epislon_func(current_epoch))
        if avg_reward > best_model_avgR:
            torch.save(model_trainer.model.state_dict(), best_model_path)
            best_model_avgR = avg_reward
        avgR_animator.add_data(avg_reward)

        current_epoch+=1

    print(f"done training")

    model_trainer = Model_Trainer(load_path = best_model_path)
    print(f"loaded model: {best_model_path}")
    while True:
        episode(env, model_trainer, wait_time, 0)



    # Don't forget to close the environment
    env.close()
