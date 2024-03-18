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
import sys

#https://gymnasium.farama.org/environments/atari/tetris/
#Observations are (210,160,3)
#http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
class Model_Trainer:
    def __init__(self, load_path = None):
        print(f"cuda? {torch.cuda.is_available()}")

        self.player_controlled = False
        self.replay_memory = []

        self.device = torch.device("cuda:0")

        self.model = Tetris_ConvNet(init_zero = False)
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
            print(f"loaded model: {load_path}")
        self.model.to(self.device)
        self.model.train()

        self.cached_model = Tetris_ConvNet(init_zero = True)
        self.cached_model.to(self.device)

        #self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = nn.SmoothL1Loss()

        learning_rate = 0.001
        betas = (0.9, 0.999)  # These are the default values in PyTorch
        eps = 1e-08  # This is the default value in PyTorch
        weight_decay = 0  # Default is no weight decay

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)


        self.gamma = 0.8
        self.batch_size = 4096

        self.max_memories = 100000
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

    def batch_update_model(self, old_board_batch, new_board_batch, actions_batch, rewards_batch, done_batch, do_print=False, print_n = 5):
        # Ensure that actions and rewards are tensors
        actions_batch = actions_batch.long()
        #print(f"\n\n\nactions: {actions_batch}")

        rewards_batch = rewards_batch.float()
        min_reward = rewards_batch.min()
        max_reward = rewards_batch.max()
        #print(f"rewards: {rewards_batch} min: {min_reward}, max_reward: {max_reward}")

        #print(f"Done: {done_batch}")




        # Predict the Q-values for the old states
        old_state_q_values = self.predict(old_board_batch)[0]
        #print(f"old_state_q_values: {old_state_q_values}")
        # Clone the old Q-values to use as targets for loss calculation
        target_q_values = old_state_q_values.clone()

        # Predict the future Q-values from the next states using the target network
        next_state_q_values = self.predict(new_board_batch, use_cached=True)[0]
        #print(f"next_state_q_values: {next_state_q_values}")

        max_future_q_values = next_state_q_values.max(1).values
        #print(f"max_future_q_values: {max_future_q_values}")




        # Update the Q-value for each action taken
        batch_index = torch.arange(old_state_q_values.size(0), device=self.device)  # Ensuring device consistency
        #print(f"batch_index: {batch_index}")



        #target_values = rewards_batch + self.gamma * max_future_q_values
        target_values = rewards_batch + self.gamma * max_future_q_values * (~done_batch)
        target_values[done_batch] = rewards_batch[done_batch]
        #print(f"target_values: {target_values}")



        target_q_values[batch_index, actions_batch] = target_values
        #print(f"target_q_values: {target_q_values}")


        # Calculate the loss
        loss = self.loss_fn(old_state_q_values, target_q_values)

        # Logging for debugging
        if do_print:
            print(f"\n")
            #print(f"   {rewards_batch[i]} + ({self.gamma}*{max_future_q_values[i]}) = {target_values[i]} ")
            print(f"   action: {actions_batch[0]}")
            print(f"   reward: {rewards_batch[0]}")
            print(f"   done: {done_batch[0]}")
            print(f"   old_board_batch.shape: {old_board_batch.shape}")
            print(f"   new_board_batch.shape: {new_board_batch.shape}")
            print(f"   old_state_q_values: {old_state_q_values[0]}")
            print(f"   next_state_q_values: {next_state_q_values[0]}")
            print(f"   target_q_values: {target_q_values[0]}")
            print(f"   target_q_values[0, actions_batch]: {target_q_values[batch_index, actions_batch][0]}")
            print(f"   loss: {loss}")
            print(f"\n")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss



    def train_replay_memory(self):
        cv2.destroyAllWindows()

        print(f"training replay memory")
        print()

        n_epochs = 5
        old_boards = []
        new_boards = []
        rewards = []
        actions = []
        dones = []
        total_loss = 0
        initial_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        replay_memories = self.replay_memory.copy()
        for e in range(n_epochs):
            random.shuffle(replay_memories)
            for memory in replay_memories:
                state, new_state, chosen_action, reward, done = memory
                old_boards.append(state)
                new_boards.append(new_state)
                actions.append(chosen_action)
                rewards.append(reward)
                dones.append(done)

            old_boards_all = torch.cat(old_boards, dim=0)
            old_boards_batches = torch.split(old_boards_all, self.batch_size, dim=0)

            new_boards_all = torch.cat(new_boards, dim=0)
            new_board_batches = torch.split(new_boards_all, self.batch_size, dim=0)



            rewards_all = torch.tensor(rewards, device=self.device)
            rewards_batches = torch.split(rewards_all, self.batch_size, dim=0)

            actions_all = torch.tensor(actions, device=self.device)
            actions_batches = torch.split(actions_all, self.batch_size, dim=0)

            dones_all = torch.tensor(dones, device=self.device)
            dones_batches = torch.split(dones_all, self.batch_size, dim=0)

            if e == 0:
                print(f"old_boards_all:{old_boards_all.shape}, n_splits: {len(old_boards_batches)}, split_dim: {old_boards_batches[0].shape}")
                print(f"new_boards_all:{new_boards_all.shape}, n_splits: {len(new_board_batches)}, split_dim: {new_board_batches[0].shape}")
                print(f"rewards_all:{rewards_all.shape}, n_splits: {len(rewards_batches)}, split_dim: {rewards_batches[0].shape}")
                print(f"actions_all:{actions_all.shape}, n_splits: {len(actions_batches)}, split_dim: {actions_batches[0].shape}")
                print(f"dones_all:{dones_all.shape}, n_splits: {len(actions_batches)}, split_dim: {dones_all[0].shape}")


            for b in range(len(old_boards_batches)):
                p = b== 0 and (e == 0 or e == n_epochs-1)
                if p:
                    print(f"epoch: {e}")
                total_loss += self.batch_update_model(old_boards_batches[b], new_board_batches[b], actions_batches[b], rewards_batches[b], dones_batches[b], do_print = p).cpu().detach().numpy()
                #self.scheduler.step()

        post_training_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        print_lr = False
        for name, initial_param in initial_weights.items():
            if torch.equal(initial_param, post_training_weights[name]):
                print(f"Weights of '{name}' did not change.")
                print_lr = True
        if print_lr:
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
            assert print_lr, "weights did not change"

        return total_loss#/len(old_boards_batches)


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
        #print(chosen_action)
        new_observation, reward, done, info, *extra_info = env.step(chosen_action)

        steps += 1
        new_board = crop_board(new_observation, display = False)
        new_temporal_img = stack_boards(past_board_2, cur_board, new_board, display = False)

        lines_cleared += reward
        reward = (reward*100)
        #reward += steps/100
        reward += reward_func(new_temporal_img)

        #if done:
        #    reward += -10
            #pass

        cumulative_reward += reward

        if epsilon != 0:
            #model_trainer.update_model(cur_temporal_img, new_temporal_img, chosen_action, reward)
            memory = (cur_temporal_img, new_temporal_img, chosen_action, reward, done)
            if len(model_trainer.replay_memory) >= model_trainer.max_memories:
                model_trainer.replay_memory.pop(0)
            model_trainer.replay_memory.append(memory)



        past_board_3 = past_board_2
        past_board_2 = cur_board
        cur_observation = new_observation


    epoch_end_time = time.time()
    duration = round(epoch_end_time - epoch_start_time, 2)
    print(f"{duration} epoch: {current_epoch}/{total_epochs}\n   cumulative_reward:{cumulative_reward}\n   steps:{steps}\n   lines_cleared:{lines_cleared}\n   epsilon:{epsilon:.6f}")

    return cumulative_reward/steps, steps, cumulative_reward


def is_top(img, x ,y):
    if (img[y,x,:] != np.array([1,1,1])).all():
        return False
    for i in range(y-1, -1):
        if (img[i, x, :] == np.array([1,1,1])).all():
            return False
    return True

def is_hole(img, x, y):
    # Check if the current pixel is filled.
    if (img[y, x, :] == np.array([1, 1, 1])).all():
        #print(f"not hole: {x}, {y} filled")
        return False

    # Check pixels above the current one.
    for i in range(y - 1, -1, -1):  # Corrected range for downward iteration.
        #print(f"checking hole: {x}, {y} with ontop {x}, {i}")
        if (img[i, x, :] == np.array([1, 1, 1])).all():
            return True

    # If it reaches this point without finding a filled pixel above, it's not a hole.
    #print(f"not hole: {x}, {y} nothing ontop")
    return False

def reward_func(board):
    """
    This reward function uses an explicitly featurized fitness function from [10],
    which takes into account the aggregate height of the tetris grid (i.e. the sum of the heights of every column),
    the number of complete lines, the number of holes in the grid, and the “bumpiness” of the grid (i.e. the sum of the absolute differences in height between adjacent columns).
    The actual formula for this fitness function is
    -0.51x*Height+0.76*lines-0.36*Holes-0.18*bumpiness
    """
    image = board.squeeze(0).permute(1, 2, 0).numpy()
    first_all_black_row = None
    y = image.shape[0] -1
    while first_all_black_row is None and y>0:
        row = image[y, :, :]
        if np.all(row == 0):
            first_all_black_row = y
        y -= 1
    pile = image[first_all_black_row:, :, :]


    #print(image.shape)
    heights = [0 for _ in range(pile.shape[1])]
    holes = []
    bumpiness = 0
    filled_cell = np.array([1,1,1])
    empty_cell = np.array([0,0,0])


    for x in range(pile.shape[1]):
        for y in range(pile.shape[0]-1, 0, -1):
            if is_top(pile, x, y):
                heights[x] = pile.shape[0]-y
            if is_hole(pile, x, y):
                holes.append((x,y))

    #print(f"holes: {holes}")
    for x,y in holes:
        pile[y,x] = np.array([0,0,1])
    for x in range(image.shape[1]):
        if heights[x] != 0:
            pile[pile.shape[0] - heights[x], x, :] = np.array([1,0,0])

    for i in range(len(heights)):
        if i == 0:
            bumpiness += abs(heights[i] - heights[i+1])
        elif i == len(heights)-1:
            bumpiness += abs(heights[i] - heights[i-1])
        else:
            bumpiness += abs(heights[i] - heights[i-1]) + abs(heights[i] - heights[i+1])

    cv2.imshow(f"pile", pile)
    #cv2.waitKey(0)

    heights_weight = -1
    holes_weight = -1
    bumpiness_weight = -1
    return (heights_weight*sum(heights))+(holes_weight*len(holes))+(bumpiness_weight*bumpiness)
if __name__ == "__main__":
    # Create your environment
    random_seed = np.random.randint(0, 10000)
    env = gym.make('ALE/Tetris-v5', render_mode='rgb_array', repeat_action_probability=0, frameskip = 4)
    print(dir(env))

    wait_time = 1 #33
    total_epochs = 3000
    nine_tenths_total_epochs = total_epochs*0.9
    current_epoch = 0

    best_model_path = 'best_model_state_dict.pth'
    latest_model_path = 'latest_model_state_dict.pth'
    fit_model_path = "fit_model_path.pth"

    model_trainer = Model_Trainer(load_path = None)

    epislon_func = lambda e: max((-0.8*(e/nine_tenths_total_epochs))+0.9, 0.1)

    animator = DataAnimator(titles=["total_loss", "fitness"])

    test = True
    if test:
        print("testing:\n")
        model_trainer = Model_Trainer(load_path = latest_model_path)
        while True:
            episode(env, model_trainer, wait_time, 0)

    best_model_avgR = -np.inf
    best_model_fitness = -np.inf
    while current_epoch < total_epochs:

        if current_epoch % 10 == 0 and current_epoch != 0:
            total_loss = model_trainer.train_replay_memory()
            model_trainer.update_cached_model()
            torch.save(model_trainer.model.state_dict(), latest_model_path)
            _, fitness, _ = episode(env, model_trainer, wait_time, 0)
            animator.add_data([total_loss, fitness])

            if fitness > best_model_fitness:
                torch.save(model_trainer.model.state_dict(), fit_model_path)

        avg_reward, steps, cumulative_reward = episode(env, model_trainer, wait_time, epislon_func(current_epoch))
        if avg_reward > best_model_avgR:
            torch.save(model_trainer.model.state_dict(), best_model_path)
            best_model_avgR = avg_reward

        current_epoch+=1

    print(f"done training")

    model_trainer = Model_Trainer(load_path = best_model_path)
    print(f"loaded model: {best_model_path}")
    while True:
        episode(env, model_trainer, wait_time, 0)



    # Don't forget to close the environment
    env.close()
