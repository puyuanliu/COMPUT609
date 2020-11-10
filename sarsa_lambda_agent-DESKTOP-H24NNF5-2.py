import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from mpl_toolkits import mplot3d

from rl_glue import RLGlue
from pendulum_env import PendulumEnvironment
from agent import BaseAgent
import plot_sarsa_lambda
import tiles3 as tc

from multiprocessing import Process


class PendulumTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same

        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """

        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tc.IHT(iht_size)

    def get_tiles(self, angle, ang_vel):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.

        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi

        returns:
        tiles -- np.array, active tiles

        """

        ANGLE_MIN = -np.pi
        ANGLE_MAX = np.pi
        ANG_VEL_MIN = -2 * np.pi
        ANG_VEL_MAX = 2 * np.pi

        angle_scale = self.num_tiles / (ANGLE_MAX - ANGLE_MIN)
        ang_vel_scale = self.num_tiles / (ANG_VEL_MAX - ANG_VEL_MIN)

        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, [angle * angle_scale, ang_vel * ang_vel_scale],
                             wrapwidths=[self.num_tiles, False])

        return np.array(tiles)


class Sarsa_lambda_Agent(BaseAgent):
    """
    Initialization of Sarsa_lambda_Agent. All values are set to None so they can
    be initialized in the agent_init method.
    """

    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = None
        self.avg_reward_step_size = None
        self.avg_reward = None
        self.iht_size = None
        self.w = None
        self.alpha = None
        self.num_tilings = None
        self.num_tiles = None
        self.initial_weights = None
        self.num_actions = None
        self.previous_tiles = None
        self.tc = None
        self.z = None
        self.lam = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.num_tilings = agent_info.get("num_tilings", 32)
        self.num_tiles = agent_info.get("num_tiles", 8)
        self.iht_size = agent_info.get("iht_size", 4096)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")
        self.alpha = agent_info.get("alpha", 0.5) / self.num_tilings
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        self.num_actions = agent_info.get("num_actions", 3)
        self.lam = agent_info.get("lambda", 0.8)
        
        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights
        self.avg_reward = 0.0
        self.z = np.zeros([self.num_actions, self.iht_size])
        # We initialize self.mctc to the mountaincar verions of the
        # tile coder that we created
        self.tc = PendulumTileCoder(iht_size=self.iht_size, num_tilings=self.num_tilings, num_tiles=self.num_tiles)
        
    def select_action(self, tiles):
        """
        Selects an action using epsilon greedy
        Args:
        tiles - np.array, an array of active tiles
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """
        action_values = []
        chosen_action = None

        # First loop through the weights of each action and populate action_values
        # with the action value for each action and tiles instance

        # Use np.random.random to decide if an exploritory action should be taken
        # and set chosen_action to a random action if it is
        # Otherwise choose the greedy action using the given argmax
        # function and the action values (don't use numpy's armax)
        for i in range(0, self.num_actions):
            action_values.append(sum(self.w[i][tiles]))
        if np.random.random() < 1 - self.epsilon:
            chosen_action = argmax(action_values)
        else:
            chosen_action = np.random.choice(range(0, self.num_actions))

        return chosen_action, action_values[chosen_action]

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        angle, ang_vel = state

        # Use self.tc to set active_tiles using position and velocity
        # set current_action to the epsilon greedy chosen action using
        # the select_action function above with the active tiles

        active_tiles = self.tc.get_tiles(angle, ang_vel)
        temp = self.select_action(active_tiles)
        current_action = temp[0]

        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # choose the action here
        angle, ang_vel = state
        self.z[self.last_action][self.previous_tiles] = 1
        
        
        active_tiles = self.tc.get_tiles(angle, ang_vel)
        temp = self.select_action(active_tiles)
        current_action = temp[0]
        current_w = self.w[self.last_action][self.previous_tiles]
        next_w = self.w[current_action][active_tiles]
        next_action_value = sum(next_w)
        current_action_value = sum(current_w)
        delta = reward - self.avg_reward + next_action_value - current_action_value

        self.avg_reward += self.avg_reward_step_size * delta
        self.w[self.last_action][self.previous_tiles] = np.add(self.w[self.last_action][self.previous_tiles], self.alpha * delta* self.z[self.last_action][self.previous_tiles])
        self.z = self.lam*self.z
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)

        return self.last_action


    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward


def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    rl_glue = RLGlue(environment, agent)

    # sweep agent parameters
    for num_tilings in agent_parameters['num_tilings']:
        for num_tiles in agent_parameters["num_tiles"]:
            for update_ss in agent_parameters["update_step_size"]:
                for lam in agent_parameters["lambda"]:
                    for avg_reward_ss in agent_parameters["avg_reward_step_size"]:
                        for epsilon in agent_parameters["epsilon"]:
                            env_info = {}
                            agent_info = {"num_tilings": num_tilings,
                                          "num_tiles": num_tiles,
                                          "alpha": update_ss,
                                          "avg_reward_step_size": avg_reward_ss,
                                          "epsilon":epsilon,
                                          "num_actions": agent_parameters["num_actions"],
                                          "iht_size": agent_parameters["iht_size"],
                                          "lambda":lam }
                            # results to save
                            return_per_step = np.zeros(
                                (experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
                            exp_avg_reward_per_step = np.zeros(
                                    (experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
                            # using tqdm we visualize progress bars
                            avg_reward_list = []
                            avg_reward = -10000
                            for run in tqdm(range(1, experiment_parameters["num_runs"] + 1)):
                                env_info["seed"] = run
                                agent_info["seed"] = run
                                rl_glue.rl_init(agent_info, env_info)
                                rl_glue.rl_start()
                                num_steps = 0
                                total_return = 0.
                                #return_arr = []
                                # exponential average reward without initial bias
                                exp_avg_reward = 0.0
                                exp_avg_reward_ss = 0.01
                                exp_avg_reward_normalizer = 0
                                while num_steps < experiment_parameters['max_steps']:
                                    num_steps += 1
                                    rl_step_result = rl_glue.rl_step()
                                    reward = rl_step_result[0]
                                    total_return += reward
                                    #return_arr.append(reward)
                                    avg_reward = rl_glue.rl_agent_message("get avg reward")
                                    exp_avg_reward_normalizer = exp_avg_reward_normalizer + exp_avg_reward_ss * (
                                                    1 - exp_avg_reward_normalizer)
                                    ss = exp_avg_reward_ss / exp_avg_reward_normalizer
                                    exp_avg_reward += ss * (reward - exp_avg_reward)
    
                                    return_per_step[run - 1][num_steps - 1] = total_return
                                    exp_avg_reward_per_step[run - 1][num_steps - 1] = exp_avg_reward
                                avg_reward_list.append(avg_reward)
                            print(np.average(avg_reward_list))
                            if not os.path.exists('results_sarsa_lambda'):
                                os.makedirs('results_sarsa_lambda')
    
                            save_name = "semi-gradient_sarsa_lambda_tilings_{}_tiledim_{}_update_ss_{}_epsilon_ss_{}_avg_reward_ss_{}_lambda_{}_max_steps_{}".format(
                                num_tilings, num_tiles, update_ss, epsilon, avg_reward_ss, lam, experiment_parameters["max_steps"])
                            total_return_filename = "results_sarsa_lambda/{}_total_return.npy".format(save_name)
                            exp_avg_reward_filename = "results_sarsa_lambda/{}_exp_avg_reward.npy".format(save_name)
    
                            np.save(total_return_filename, return_per_step)
                            np.save(exp_avg_reward_filename, exp_avg_reward_per_step)


def get_data(agent_parameters, max_steps, directory, file_type):
    # input is the directory of the data result folder
    # file_type is the type of the data we want, either exp_avg_reward or...
    # the function returns the required data

    num_tilings = agent_parameters["num_tilings"][0]
    num_tiles = agent_parameters["num_tiles"][0]
    update_ss = agent_parameters["update_step_size"][0]
    epsilon = agent_parameters["epsilon"][0]
    avg_reward_ss = agent_parameters["avg_reward_step_size"][0]
    lam = agent_parameters["lambda"][0]
    load_name = 'semi-gradient_sarsa_lambda_tilings_{}_tiledim_{}_update_ss_{}_epsilon_ss_{}_avg_reward_ss_{}_lambda_{}_max_steps_{}'.format(
        num_tilings, num_tiles, update_ss, epsilon, avg_reward_ss, lam, max_steps)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type))
    return data


def run_specific():
    # This function runs the experiment for some specific parameters
    # Experiment parameters
    max_steps = 400000
    experiment_parameters = {
        "max_steps" : max_steps,
        "num_runs" : 50
    }

    # Environment parameters
    environment_parameters = {}

    # Agent parameters
    #   Each element is an array because we will be later sweeping over multiple values
    # actor and critic step-sizes are divided by num. tilings inside the agent
    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "update_step_size": [0.2],
        "epsilon": [0.01],
        "avg_reward_step_size": [2**-6],
        "num_actions": 3,
        "iht_size": 4096,
        "lambda":[0.7]
    }

    current_env = PendulumEnvironment
    current_agent = Sarsa_lambda_Agent


    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
    plot_sarsa_lambda.plot_result(agent_parameters, 'results_sarsa_lambda', max_steps)


def A_2D_mash_grid():
    max_steps = 100000
    experiment_parameters = {
        "max_steps" : max_steps,
        "num_runs" : 50
    }

    # Environment parameters
    environment_parameters = {}

    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "update_step_size": [5.5],
        "epsilon": [0.01],
        "avg_reward_step_size": [2**-6],
        "num_actions": 3,
        "iht_size": 4096*10,
        "lambda":[0.6]
    }
    #lambda 0.9, 0.8, 0.7, 0.6
    max_i = 8
    max_j = 12
    epislon_range = np.linspace(2**-10, 2**-3, num=max_i)
    update_step_size_range = np.linspace(2**-11, 2**0, num=max_j)
    result_exp_error = np.empty([max_i,max_j])
    test_scope = 5000 # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = Sarsa_lambda_Agent
    file_type = "exp_avg_reward"
    directory = "results_sarsa_lambda"
    i = 0
    for epsilon in epislon_range: #ass is actor step size
        j = 0
        for update_step_size in update_step_size_range: #ass is actor step size
            agent_parameters["epsilon"] = [epsilon]
            agent_parameters["update_step_size"] = [update_step_size]
            run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
            data = get_data(agent_parameters, directory, file_type)
            data_mean = np.mean(data, axis=0)
            data_mean = data_mean[-1*test_scope:-1]
            result_exp_error[i][j] = np.average(data_mean)
            j+=1
        i +=1
    epsilon = [round(num, 3) for num in epsilon]
    update_step_size = [round(num, 3) for num in update_step_size]
    result_exp_error = result_exp_error.round(3)
    print(epislon_range)
    print(update_step_size_range)
    print(result_exp_error)
    print(experiment_parameters)
    print(agent_parameters)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(epislon_range, update_step_size_range, result_exp_error,cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()


def test_parameters_epsilon():
    # This function tests the effect of some parameters for the experiment
    # In this algorithm, we have 3 meta parameters in total
    # we 2 parameters and test the relationship between the remaining parameter and the error.
    max_steps = 200000
    experiment_parameters = {
        "max_steps" : max_steps,
        "num_runs" : 50
    }

    environment_parameters = {}
    
    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "update_step_size": [0.15],
        "epsilon": [0.03],
        "avg_reward_step_size": [2**-6],
        "num_actions": 3,
        "iht_size": 4096,
        "lambda":[0.6]
    }
    # We first test the Epsilon
    epislon_range = np.linspace(0.0005, 0.01, num=10)
    
    result_exp_error = []
    test_scope = 5000 # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = Sarsa_lambda_Agent
    file_type = "exp_avg_reward"
    directory = "results_sarsa_lambda"
    for epsilon in epislon_range: #ass is actor step size
        agent_parameters["epsilon"] = [epsilon]
        run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
        data = get_data(agent_parameters, max_steps, directory, file_type)
        data_mean = np.mean(data, axis=0)
        data_mean = data_mean[-1*test_scope:-1]
        result_exp_error.append(np.average(data_mean))

    plt.plot(epislon_range,result_exp_error)
    epislon_range=[round(num, 3) for num in epislon_range]
    plt.xticks(epislon_range)
    plt.tight_layout()
    plt.xlabel("Epsilon")
    plt.ylabel("Exponential Average Reward for the Last 5000 Time Steps")
    plt.title("Effect of Changing Epsilon")
    plt.tight_layout()
    plt.savefig('sarsa_lambda_epsilon_test.png')

    plt.clf() # clear the old plot


def test_parameters_update_step_size():

    # We then test the Ypdate step size
    max_steps = 20000

    experiment_parameters = {
        "max_steps" : max_steps,
        "num_runs" : 50
    }

    environment_parameters = {}
    
    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "update_step_size": [0.15],
        "epsilon": [0.03],
        "avg_reward_step_size": [2**-6],
        "num_actions": 3,
        "iht_size": 4096,
        "lambda":[0.6]
    }

    update_step_size_range = np.linspace(2**-11, 2**0, num=10)
    result_exp_error = []
    current_env = PendulumEnvironment
    current_agent = Sarsa_lambda_Agent
    test_scope = 5000 # we test the exponential return on the last 50000 time steps.
    file_type = "exp_avg_reward"
    directory = "results_sarsa_lambda"
    for update_step_size in update_step_size_range: #ass is actor step size
        agent_parameters["update_step_size"] = [update_step_size]
        run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
        data = get_data(agent_parameters, max_steps, directory, file_type)
        data_mean = np.mean(data, axis=0)
        data_mean = data_mean[-1*test_scope:-1]
        result_exp_error.append(np.average(data_mean))

    plt.plot(update_step_size_range,result_exp_error)
    update_step_size_range = [round(num, 3) for num in update_step_size_range]
    plt.xticks(update_step_size_range)
    plt.xlabel("32 * Update Step Size")
    plt.ylabel("Exponential Average Reward for the Last 5000 Time Steps")
    plt.title("Effect of Changing Update Step Size")
    plt.tight_layout()
    plt.savefig('sarsa_lambda_upate_step_size_test.png')
    plt.clf() # clear the old plot




def test_parameters_avg_step_size():

    max_steps = 200000
    experiment_parameters = {
        "max_steps" : max_steps,
        "num_runs" : 50
    }

    environment_parameters = {}
    
    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "update_step_size": [0.15],
        "epsilon": [0.03],
        "avg_reward_step_size": [2**-6],
        "num_actions": 3,
        "iht_size": 4096,
        "lambda":[0.6]
    }

    # We now test the Average reward step size
    avg_reward_step_size_range = np.linspace(2**-11, 2**-2, num=10)
    result_exp_error = []
    test_scope = 5000  # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = Sarsa_lambda_Agent
    file_type = "exp_avg_reward"
    directory = "results_sarsa_lambda"
    for ass in avg_reward_step_size_range:  # ass is actor step size
        agent_parameters["avg_reward_step_size"] = [ass]
        run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
        data = get_data(agent_parameters, max_steps, directory, file_type)
        data_mean = np.mean(data, axis=0)
        data_mean = data_mean[-1 * test_scope:-1]
        result_exp_error.append(np.average(data_mean))

    plt.plot(avg_reward_step_size_range, result_exp_error)
    avg_reward_step_size_range = [round(num, 3) for num in avg_reward_step_size_range]
    plt.xticks(avg_reward_step_size_range)
    plt.xlabel("Average Reward Step Size")
    plt.ylabel("Exponential Average Reward for the Last 5000 Time Steps")
    plt.title("Effect of Changing Average Reward Step Size")
    plt.tight_layout()
    plt.savefig('sarsa_lambda_avg_reward_step_size_test.png')

    plt.clf()  # clear the old plot

#test_parameters_epsilon()


#p1.join()
#p2.join()
#p3.join()
test_parameters_update_step_size()
#test_parameters()
#test_parameters_epsilon()
#test_parameters_update_step_size()
#test_parameters_avg_step_size()

#A_2D_mash_grid()