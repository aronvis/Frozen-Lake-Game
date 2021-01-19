"""
Play ground for Frozen Lake env
"""

from tkinter import Tk, Canvas, Frame, X, Text, END, DISABLED, Button
import numpy as np
import random
import json

UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3


def ajacent_action(action1, action2):
    mod = (action1 - action2) % 4
    return mod == 1 or mod == 3


class FrozenLakeEnv(object):
    def __init__(self, json_file, fully_visible):
        self.__parse_json(json_file)
        self.current_state = None
        self.fully_visible = fully_visible
        self.episode_rewards = None
        self.done = True
        self.__draw_board()

    def __get_reward(self, state):
        x, y = state
        symbol = self.two_d_map[x][y]
        return self.reward_dict[symbol]

    def __get_next_state_prob_reward(self, current_state, intend_action):
        result_dict = {}
        all_action_next_state = []
        all_next_state = []
        if current_state[0] != 0:
            all_action_next_state.append((UP, (current_state[0] - 1, current_state[1])))
            all_next_state.append((current_state[0] - 1, current_state[1]))
        if current_state[0] != self.num_rows - 1:
            all_action_next_state.append((DOWN, (current_state[0] + 1, current_state[1])))
            all_next_state.append((current_state[0] + 1, current_state[1]))
        if current_state[1] != 0:
            all_action_next_state.append((LEFT, (current_state[0], current_state[1] - 1)))
            all_next_state.append((current_state[0], current_state[1] - 1))
        if current_state[1] != self.num_columns - 1:
            all_action_next_state.append((RIGHT, (current_state[0], current_state[1] + 1)))
            all_next_state.append((current_state[0], current_state[1] + 1))

        for i, (action, next_state) in enumerate(all_action_next_state):
            result_dict[action] = [(next_state, 1 - self.noise, self.__get_reward(next_state))]
            total_num_possible_other_state = 0
            for j in range(len(all_action_next_state)):
                if i != j and ajacent_action(action, all_action_next_state[j][0]):
                    total_num_possible_other_state += 1
            if self.noise != 0 and len(all_next_state) > 1:
                each_prob = self.noise / total_num_possible_other_state
                for j, other_state in enumerate(all_next_state):
                    if i != j and ajacent_action(action, all_action_next_state[j][0]):
                        result_dict[action].append((other_state, each_prob, self.__get_reward(other_state)))
        return result_dict[intend_action]

    def __validate_action(self, action):
        if self.current_state[0] == 0 and action == UP:
            return False
        if self.current_state[0] == self.num_rows - 1 and action == DOWN:
            return False
        if self.current_state[1] == 0 and action == LEFT:
            return False
        if self.current_state[1] == self.num_columns - 1 and action == RIGHT:
            return False
        return True

    def step(self, action):
        if self.done:
            print('Must reset before next episode.')
            return
        # construct a map from [(next_state_1, prob_1, reward_1), (next_state_2, prob_2, reward_2)]
        if not self.__validate_action(action):
            print('Action {} is forbidden at state {}'.format(action, self.current_state))
            return
        transitions = self.__get_next_state_prob_reward(self.current_state, action)
        state, prob, reward = list(zip(*transitions))
        index = np.random.choice(len(state), p=prob)
        self.current_state = state[index]
        # update locations in
        self.app.update_position(self.current_state)
        if self.current_state in self.terminal_state_dict:
            self.done = True
        else:
            self.done = False
        self.episode_rewards += reward[index]
        print('Current State: {}. Action: {}. Reward: {}. Done: {}'.format(self.current_state, action,
                                                                           reward[index], self.done))
        if self.done:
            print('Finish episode. Total rewards: {}'.format(self.episode_rewards))
        return self.current_state, reward[index], self.done

    def reset(self):
        self.episode_rewards = 0
        self.done = False
        self.current_state = random.choice(self.start_state_dict)
        self.app.update_position(self.current_state)
        print('Reset! Current State: {}'.format(self.current_state))
        return self.current_state

    def render(self, mode='human'):
        """ Draw the current state (pos) on the board

        :param mode:
        :return:
        """
        pass

    def __draw_board(self):
        """ draw a board, use color to indicate

        :return:
        """
        self.root = Tk()
        self.app = Board(self.root, env=self, fully_visible=self.fully_visible)
        self.root.mainloop()

    def __parse_json(self, json_file):
        with open(json_file, 'r') as f:
            json_object = json.load(f)
        self.name = json_object['name']
        self.reward_dict = json_object['reward']
        self.noise = float(json_object['noise'])
        # parse grid into 2D object
        rows = json_object['grid'].split()  # a list of string for each row.
        self.two_d_map = list(map(lambda row: row.split(','), rows))
        self.num_rows = len(self.two_d_map)
        self.num_columns = len(self.two_d_map[0])
        self.start_state_dict = []
        self.terminal_state_dict = set()
        for i, row in enumerate(self.two_d_map):
            for j, state in enumerate(row):
                if state[0] == 'S':
                    self.start_state_dict.append((i, j))
                elif state[0] == 'G' or state[0] == 'H':
                    self.terminal_state_dict.add((i, j))


class Board(Frame):
    def __init__(self, master, env, fully_visible=False):
        super(Board, self).__init__(master=master)
        self.env = env
        self.left_top_pos = (50, 50)
        self.width = 50
        self.fully_visible = fully_visible
        self.old_position = (0, 0)
        self.init_ui(self.env.two_d_map, self.left_top_pos, self.width, self.env.noise,
                     self.env.reward_dict, self.env.name)

    def update_position(self, new_position):
        old_x, old_y = self.old_position
        new_x, new_y = new_position
        old_rect, old_text = self.rectangles[old_x][old_y]
        self.canvas.itemconfig(old_rect, fill='white')
        rect, text = self.rectangles[new_x][new_y]
        self.canvas.itemconfig(rect, fill='yellow')
        if not self.fully_visible:
            self.canvas.itemconfig(text, text=self.env.two_d_map[new_x][new_y])
        self.old_position = new_position

    def init_ui(self, two_d_map, left_top_pos, width, noise, reward_dict, name):
        self.master.title(name)

        if self.fully_visible:
            self.text = Text(master=self.master, height=3)
            self.text.insert(END, 'Noise: {}\nReward: {}'.format(noise, reward_dict))
            self.text.pack(fill=X, pady=1)
            self.text.config(state=DISABLED)

        self.canvas = Canvas(master=self.master, width=100 + self.width * len(two_d_map[0]),
                             height=100 + self.width * len(two_d_map))
        self.canvas.pack(fill=X)
        global_x, global_y = left_top_pos
        self.rectangles = []
        for row in range(len(two_d_map)):
            current = []
            for column in range(len(two_d_map[0])):
                x1 = global_x + width * column
                y1 = global_y + width * row
                x2 = global_x + width * (column + 1)
                y2 = global_y + width * (row + 1)
                if self.fully_visible:
                    rectangle = self.canvas.create_rectangle(x1, y1, x2, y2, fill='white')
                    text = self.canvas.create_text(x1 + width / 2, y1 + width / 2, text=two_d_map[row][column])
                else:
                    rectangle = self.canvas.create_rectangle(x1, y1, x2, y2, fill='white')
                    text = self.canvas.create_text(x1 + width / 2, y1 + width / 2, text='Unk')
                current.append((rectangle, text))
            self.rectangles.append(current)

        # botton
        self.reset_button = Button(self.master, text='RESET', command=self.env.reset)
        self.reset_button.pack()
        self.up_button = Button(self.master, text='UP', command=lambda: self.env.step(UP))
        self.up_button.pack()
        self.down_button = Button(self.master, text='DOWN', command=lambda: self.env.step(DOWN))
        self.down_button.pack()
        self.left_button = Button(self.master, text='LEFT', command=lambda: self.env.step(LEFT))
        self.left_button.pack()
        self.right_button = Button(self.master, text='RIGHT', command=lambda: self.env.step(RIGHT))
        self.right_button.pack()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--not_fully_visible', action='store_true')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    maps = args['map']
    not_fully_visible = args['not_fully_visible']
    env = FrozenLakeEnv(json_file=maps, fully_visible=not not_fully_visible)
