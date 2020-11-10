
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from rl_glue import RLGlue
from pendulum_env import PendulumEnvironment
from agent import BaseAgent
import plot_script
import tiles3 as tc
import matplotlib.ticker


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


def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions

    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles

    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """

    # First compute the list of state-action preferences (1~2 lines)
    # state_action_preferences = ? (list of size 3)
    state_action_preferences = []
    for i in range(0, 3):
        state_action_preferences.append(actor_w[i][tiles].sum())

    # Set the constant c by finding the maximum of state-action preferences (use np.max) (1 line)
    # c = ? (float)
    c = np.max(state_action_preferences)
    numerator = []
    for i in range(0, len(state_action_preferences)):
        numerator.append(np.exp(state_action_preferences[i] - c))

    # Next compute the denominator by summing the values in the numerator (use np.sum) (1 line)
    # denominator = ? (float)
    denominator = np.sum(numerator)

    # Create a probability array by dividing each element in numerator array by denominator (1 line)
    # We will store this probability array in self.softmax_prob as it will be useful later when updating the Actor
    # softmax_prob = ? (list of size 3)
    softmax_prob = numerator / denominator

    return softmax_prob


class ActorCriticSoftmaxAgent(BaseAgent):
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size") / num_tilings
        self.critic_step_size = agent_info.get("critic_step_size") / num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.actions = list(range(agent_info.get("num_actions")))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size.
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder

        Returns:
            The action selected according to the policy
        """

        # compute softmax probability
        softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)

        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)

        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (2 lines)
        # set current_action by calling self.agent_policy with active_tiles

        active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self.agent_policy(active_tiles)

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (1 line)
        active_tiles = self.tc.get_tiles(angle, ang_vel)

        ### Compute delta using Equation (1) (1 line)
        pre_state_value = sum(self.critic_w[self.prev_tiles])
        current_state_value = sum(self.critic_w[active_tiles])
        delta = reward - self.avg_reward + current_state_value - pre_state_value

        ### update average reward using Equation (2) (1 line)

        # 𝑅¯←𝑅¯+𝛼𝑅¯𝛿
        self.avg_reward += self.avg_reward_step_size * delta


        # update critic weights using Equation (3) and (5) (1 line)
        # self.critic_w[self.prev_tiles] += ?
        self.critic_w[self.prev_tiles] += self.critic_step_size * delta

        # update actor weights using Equation (4) and (6)
        # We use self.softmax_prob saved from the previous timestep
        # We leave it as an exercise to verify that the code below corresponds to the equation.
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        ### set current_action by calling self.agent_policy with active_tiles (1 line)
        current_action = self.agent_policy(self.prev_tiles)

        self.prev_tiles = active_tiles
        self.last_action = current_action

        return self.last_action

    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    rl_glue = RLGlue(environment, agent)

    # sweep agent parameters
    for num_tilings in agent_parameters['num_tilings']:
        for num_tiles in agent_parameters["num_tiles"]:
            for actor_ss in agent_parameters["actor_step_size"]:
                for critic_ss in agent_parameters["critic_step_size"]:
                    for avg_reward_ss in agent_parameters["avg_reward_step_size"]:

                        env_info = {}
                        agent_info = {"num_tilings": num_tilings,
                                      "num_tiles": num_tiles,
                                      "actor_step_size": actor_ss,
                                      "critic_step_size": critic_ss,
                                      "avg_reward_step_size": avg_reward_ss,
                                      "num_actions": agent_parameters["num_actions"],
                                      "iht_size": agent_parameters["iht_size"]}

                        # results to save
                        return_per_step = np.zeros(
                            (experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
                        exp_avg_reward_per_step = np.zeros(
                            (experiment_parameters["num_runs"], experiment_parameters["max_steps"]))

                        # using tqdm we visualize progress bars
                        for run in tqdm(range(1, experiment_parameters["num_runs"] + 1)):
                            env_info["seed"] = run
                            agent_info["seed"] = run

                            rl_glue.rl_init(agent_info, env_info)
                            rl_glue.rl_start()

                            num_steps = 0
                            total_return = 0.
                            return_arr = []

                            # exponential average reward without initial bias
                            exp_avg_reward = 0.0
                            exp_avg_reward_ss = 0.01
                            exp_avg_reward_normalizer = 0

                            while num_steps < experiment_parameters['max_steps']:
                                num_steps += 1

                                rl_step_result = rl_glue.rl_step()

                                reward = rl_step_result[0]
                                total_return += reward
                                return_arr.append(reward)
                                avg_reward = rl_glue.rl_agent_message("get avg reward")

                                exp_avg_reward_normalizer = exp_avg_reward_normalizer + exp_avg_reward_ss * (
                                            1 - exp_avg_reward_normalizer)
                                ss = exp_avg_reward_ss / exp_avg_reward_normalizer
                                exp_avg_reward += ss * (reward - exp_avg_reward)

                                return_per_step[run - 1][num_steps - 1] = total_return
                                exp_avg_reward_per_step[run - 1][num_steps - 1] = exp_avg_reward

                        if not os.path.exists('results_actor_critic'):
                            os.makedirs('results_actor_critic')

                        save_name = "ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}".format(
                            num_tilings, num_tiles, actor_ss, critic_ss, avg_reward_ss)
                        total_return_filename = "results_actor_critic/{}_total_return.npy".format(save_name)
                        exp_avg_reward_filename = "results_actor_critic/{}_exp_avg_reward.npy".format(save_name)

                        np.save(total_return_filename, return_per_step)
                        np.save(exp_avg_reward_filename, exp_avg_reward_per_step)


def get_data(agent_parameters, directory, file_type):
    # input is the directory of the data result folder
    # file_type is the type of the data we want, either exp_avg_reward or...
    # the function returns the required data
    num_tilings = agent_parameters["num_tilings"][0]
    num_tiles = agent_parameters["num_tiles"][0]
    actor_ss = agent_parameters["actor_step_size"][0]
    critic_ss = agent_parameters["critic_step_size"][0]
    avg_reward_ss = agent_parameters["avg_reward_step_size"][0]
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}'.format(
        num_tilings, num_tiles, actor_ss, critic_ss, avg_reward_ss)
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type))
    return data


def run_specific():
    # This function runs the experiment for some specific parameters
    # Experiment parameters
    experiment_parameters = {
        "max_steps" : 300000,
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
        "actor_step_size": [2**(-2)],
        "critic_step_size": [2**1],
        "avg_reward_step_size": [2**(-6)],
        "num_actions": 3,
        "iht_size": 4096
    }

    current_env = PendulumEnvironment
    current_agent = ActorCriticSoftmaxAgent


    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
    plot_script.plot_result(agent_parameters, 'results_actor_critic')

def test_parameters():
    # This function tests the effect of some parameters for the experiment
    # In this algorithm, we have 3 meta parameters in total
    # we 2 parameters and test the relationship between the remaining parameter and the error.
    experiment_parameters = {
        "max_steps" : 20000,
        "num_runs" : 50
    }

    environment_parameters = {}

    agent_parameters  = {
    "num_tilings": [32],
    "num_tiles": [8],
    "actor_step_size": [2**(-2)],
    "critic_step_size": [2**1],
    "avg_reward_step_size": [2**(-6)],
    "num_actions": 3,
    "iht_size": 4096
}
    # We first test the Actor step size
    actor_step_size_range = np.linspace(2**-6, 2**1, num=10)
    result_exp_error = []
    test_scope = 5000 # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = ActorCriticSoftmaxAgent
    file_type = "exp_avg_reward"
    directory = "results_actor_critic"
    for ass in actor_step_size_range: #ass is actor step size
        agent_parameters["actor_step_size"] = [ass]
#        run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
        data = get_data(agent_parameters, directory, file_type)
        data_mean = np.mean(data, axis=0)
        data_mean = data_mean[-1*test_scope:-1]
        result_exp_error.append(np.average(data_mean))

    plt.plot(actor_step_size_range,result_exp_error)
    plt.xticks(actor_step_size_range)
    plt.xlabel("32 * Actor Step Size", fontsize = 18)
    plt.ylabel("Exponential Average Reward", fontsize= 18)
#    plt.title("Effect of Changing Actor Step Size")
    #plt.savefig('actor_step_size_test.png')

    plt.clf() # clear the old plot

    # Now we consider the Critic step size -------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "actor_step_size": [2 ** (-2)],
        "critic_step_size": [2 ** 1],
        "avg_reward_step_size": [2 ** (-6)],
        "num_actions": 3,
        "iht_size": 4096
    }
    critic_step_size_range = np.linspace(2 ** -4, 2 ** 2, num=10)
    result_exp_error = []
    test_scope = 5000  # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = ActorCriticSoftmaxAgent
    file_type = "exp_avg_reward"
    directory = "results_actor_critic"
    for ass in critic_step_size_range:  # ass is actor step size
        agent_parameters["critic_step_size"] = [ass]
#        run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
        data = get_data(agent_parameters, directory, file_type)
        data_mean = np.mean(data, axis=0)
        data_mean = data_mean[-1 * test_scope:-1]
        result_exp_error.append(np.average(data_mean))

    plt.plot(critic_step_size_range, result_exp_error)
    plt.xticks(critic_step_size_range)
    plt.xlabel("32 * Critic Step Size", fontsize = 18)
    plt.ylabel("Exponential Average Reward", fontsize = 18)
#    plt.title("Effect of Changing Critic Step Size")
    plt.tight_layout()
#    plt.savefig('critic_step_size_test.png')

    plt.clf()  # clear the old plot

    # Now we consider the Averange reward step size -------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "actor_step_size": [2 ** (-2)],
        "critic_step_size": [2 ** 1],
        "avg_reward_step_size": [2 ** (-6)],
        "num_actions": 3,
        "iht_size": 4096
    }
    avg_reward_step_size_range = np.linspace(2 ** -11, 0.1, num=10)
    result_exp_error = []
    test_scope = 5000  # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = ActorCriticSoftmaxAgent
    file_type = "exp_avg_reward"
    directory = "results_actor_critic"
    for ass in avg_reward_step_size_range:  # ass is actor step size
        agent_parameters["avg_reward_step_size"] = [ass]
        run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
        data = get_data(agent_parameters, directory, file_type)
        data_mean = np.mean(data, axis=0)
        data_mean = data_mean[-1 * test_scope:-1]
        result_exp_error.append(np.average(data_mean))

    plt.plot(avg_reward_step_size_range, result_exp_error)
    plt.xticks(avg_reward_step_size_range)
    plt.tight_layout()
    plt.xlabel("Average Reward Step Size", fontsize = 18)
    plt.ylabel("Exponential Average Reward", fontsize = 18)
   # plt.title("Effect of Changing Average Reward Step Size")
    plt.tight_layout()
    plt.savefig('avg_reward_step_size_test.png')

    plt.clf()  # clear the old plot

def A_2D_mash_grid():
    max_steps = 20000
    experiment_parameters = {
        "max_steps" : max_steps,
        "num_runs" : 50
    }

    # Environment parameters
    environment_parameters = {}

    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "actor_step_size": [2 ** (-2)],
        "critic_step_size": [2 ** 1],
        "avg_reward_step_size": [2 ** (-3)],
        "num_actions": 3,
        "iht_size": 4096
    }
    #avg 2^-8, 2^-7, 2^-6, 2^-5
    max_i = 11
    max_j = 11
    actor_range = np.array([2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**-0, 2**1, 2**2])
    critic_step_size_range = np.array([2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-4, 2**-3, 2**-1, 2**0, 2**1, 2**2])
    result_exp_error = np.empty([max_i,max_j])
    test_scope = int(max_steps*0.1) # we test the exponential return on the last 50000 time steps.
    current_env = PendulumEnvironment
    current_agent = ActorCriticSoftmaxAgent
    file_type = "exp_avg_reward"
    directory = "results_actor_critic"
    i = 0
    for actor_step_size in actor_range: #ass is actor step size
        j = 0
        for critic_step_size in critic_step_size_range: #ass is actor step size
            agent_parameters["actor_step_size"] = [actor_step_size]
            agent_parameters["critic_step_size"] = [critic_step_size]
            run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
            data = get_data(agent_parameters, directory, file_type)
            data_mean = np.mean(data, axis=0)
            data_mean = data_mean[-1*test_scope:-1]
            result_exp_error[i][j] = np.average(data_mean)
            j+=1
        i +=1
    actor_range = [round(num, 3) for num in actor_range]
    critic_step_size_range = [round(num, 3) for num in critic_step_size_range]
    result_exp_error = result_exp_error.round(3)
    print(actor_range)
    print(critic_step_size_range)
    print(result_exp_error)
    print(experiment_parameters)
    print(agent_parameters)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(actor_range, critic_step_size_range, result_exp_error,cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()

A_2D_mash_grid()
#test_parameters()
#run_specific()