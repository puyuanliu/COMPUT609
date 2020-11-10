import numpy as np
import matplotlib.pyplot as plt


# Function to plot result
def plot_result(agent_parameters, directory, max_steps):
    plt1_agent_sweeps = []
    plt2_agent_sweeps = []
    x_range = max_steps
    plt_xticks = np.linspace(0, max_steps, 6)
    plt_xlabels = np.linspace(0, max_steps, 6)
    plt1_yticks = range(0, -6001, -2000)
    plt2_yticks = range(-3, 1, 1)

    # single plots: Exp Avg reward
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))

    for num_tilings in agent_parameters['num_tilings']:
        for num_tiles in agent_parameters["num_tiles"]:
            for update_ss in agent_parameters["update_step_size"]:
                for avg_reward_ss in agent_parameters["avg_reward_step_size"]:
                    for lam in agent_parameters["lambda"]:
                        for epsilon in agent_parameters["epsilon"]:
                            load_name = 'semi-gradient_sarsa_lambda_tilings_{}_tiledim_{}_update_ss_{}_epsilon_ss_{}_avg_reward_ss_{}_lambda_{}'.format(
                                    num_tilings, num_tiles, update_ss, epsilon, avg_reward_ss, lam)
    
                            ### plot1
                            file_type1 = "total_return"
                            data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))
    
                            data_mean = np.mean(data, axis=0)
                            data_std_err = np.std(data, axis=0) / np.sqrt(len(data))
    
                            data_mean = data_mean[:x_range]
                            data_std_err = data_std_err[:x_range]
    
                            plt_x_legend = range(0, len(data_mean))[:x_range]
    
                            ax[0].fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
                            graph_current_data, = ax[0].plot(plt_x_legend, data_mean, linewidth=1.0,
                                                             label="update_ss: {}/32, epsilon: {}, avg reward step_size: {}, lambda: {}".format(
                                                                 update_ss, epsilon, avg_reward_ss, lam))
                            plt1_agent_sweeps.append(graph_current_data)
    
                            ### plot2
                            file_type2 = "exp_avg_reward"
                            data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))
    
                            data_mean = np.mean(data, axis=0)
                            data_std_err = np.std(data, axis=0) / np.sqrt(len(data))
    
                            data_mean = data_mean[:x_range]
                            data_std_err = data_std_err[:x_range]
    
                            plt_x_legend = range(1, len(data_mean) + 1)[:x_range]
    
                            ax[1].fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
                            graph_current_data, = ax[1].plot(plt_x_legend, data_mean, linewidth=1.0,
                                                             label="update_ss: {}/32, epsilon: {}, avg reward: {}, lambda: {}".format(
                                                                 update_ss, epsilon, avg_reward_ss, lam))
                            plt2_agent_sweeps.append(graph_current_data)

    # plot 1
    ax[0].legend(handles=[*plt1_agent_sweeps])
    ax[0].set_xticks(plt_xticks)
    ax[0].set_yticks(plt1_yticks)
    ax[0].set_xticklabels(plt_xlabels)
    ax[0].set_yticklabels(plt1_yticks)

    ax[0].set_title("Return per Step")
    ax[0].set_xlabel('Training steps')
    ax[0].set_ylabel('Total Return', rotation=90)
    ax[0].set_xlim([0, max_steps])

    # plot 2
    ax[1].legend(handles=[*plt2_agent_sweeps])
    ax[1].set_xticks(plt_xticks)
    ax[1].set_yticks(plt2_yticks)

    ax[1].set_title("Exponential Average Reward per Step")
    ax[1].set_xlabel('Training steps')
    ax[1].set_ylabel('Exponential Average Reward', rotation=90)
    ax[1].set_xticklabels(plt_xlabels)
    ax[1].set_yticklabels(plt2_yticks)
    ax[1].set_xlim([0, max_steps])
    ax[1].set_ylim([-3, 0.16])

    plt.suptitle("Average Reward Semi-gradient Sarsa(lambda) ({} Runs)".format(len(data)), fontsize=16, fontweight='bold',
                 y=1.03)

    # ax[1].legend(handles=plt2_agent_sweeps)

    # ax[1].set_title("Softmax policy Actor-Critic: Average Reward per Step ({} Runs)".format(len(avg_reward)))
    # ax[1].set_xlabel('Training steps')
    # ax[1].set_ylabel('Average Reward', rotation=0, labelpad=40)
    # ax[1].set_xticklabels(plt_xticks)
    # ax[1].set_yticklabels(plt_yticks)
    # ax.axhline(y=0.1, linestyle='dashed', linewidth=1.0, color='black')

    plt.tight_layout()
    # plt.suptitle("{}-State Aggregation".format(num_agg_states),fontsize=16, fontweight='bold', y=1.03)
    # plt.suptitle("Average Reward ActorCritic",fontsize=16, fontweight='bold', y=1.03)
    plt.savefig('sarsa_lambda_specific.png')
    plt.show()


def plot_sweep_result(directory):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))

    plt_agent_sweeps = []

    agent_parameters = {
        "num_tilings": [32],
        "num_tiles": [8],
        "update_step_size": [0.07],
        "epsilon": [0.03],
        "avg_reward_step_size": [2**-6],
        "num_actions": 3,
        "iht_size": 4096,
        "lambda": [0.7]
    }

    epsilon = agent_parameters["epsilon"]
    lam = agent_parameters["lam"]
    update_ss = agent_parameters["update_step_size"]
    avg_reward_ss = agent_parameters["avg_reward_step_size"]
    x_range = 300000
    #plt_xticks = [0, 4999, 9999, 14999, 19999]
    #plt_xlabels = [1, 5000, 10000, 15000, 20000]
    #plt2_yticks = range(-3, 1, 1)

    top_results = [{"num_tilings":4},
                   {"num_tilings": 8},
                   {"num_tilings": 16},
                   {"num_tilings": 32},
                   {"num_tilings": 64}]

    for setting in top_results:
        num_tilings = setting["num_tilings"]
        num_tiles = 8

        load_name = "semi-gradient_sarsa_lambda_tilings_{}_tiledim_{}_update_ss_{}_epsilon_ss_{}_avg_reward_ss_{}_lambda_{}_max_steps_{}".format(
                                num_tilings, num_tiles, update_ss, epsilon, avg_reward_ss, lam, x_range)

        file_type2 = "exp_avg_reward"
        data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))

        data_mean = np.mean(data, axis=0)
        data_std_err = np.std(data, axis=0) / np.sqrt(len(data))

        data_mean = data_mean[:x_range]
        data_std_err = data_std_err[:x_range]

        plt_x_legend = range(1, len(data_mean) + 1)[:x_range]

        ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
        graph_current_data, = ax.plot(plt_x_legend, data_mean, linewidth=1.0,
                                      label="number of tilings {}".format(num_tilings))
        plt_agent_sweeps.append(graph_current_data)

    ax.legend(handles=[*plt_agent_sweeps])
    ax.set_xticks(plt_xticks)
    ax.set_yticks(plt2_yticks)

    ax.set_title("Exponential Average Reward per Step ({} Runs)".format(len(data)))
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Exponential Average Reward', rotation=90)
    #ax.set_xticklabels(plt_xlabels)
    #ax.set_yticklabels(plt2_yticks)
    ax.set_xlim([0, 300000])
    ax.set_ylim([-3.5, 0.16])

