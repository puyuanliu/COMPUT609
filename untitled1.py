# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:33:54 2020

@author: puyua
"""


import numpy as np
import matplotlib.pyplot as plt



def plot_sweep_result():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))

    plt_agent_sweeps = []
    
    directory = 'results_actor_critic'
    num_tilings = 32
    num_tiles = 8
    actor_step_size = 2**-2
    critic_step_size = 2**1
    avg_reward_step_size = 2**-6
    x_range = 300000
    #plt_xticks = [0, 4999, 9999, 14999, 19999]
    #plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt2_yticks = range(-3, 1, 1)
    load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}_max_steps_{}'.format(num_tilings, num_tiles, actor_step_size, critic_step_size, avg_reward_step_size, x_range)
    file_type2 = "exp_avg_reward"
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))

    data_mean = np.mean(data, axis=0)
    data_std_err = np.std(data, axis=0) / np.sqrt(len(data))

    data_mean = data_mean[:x_range]
    data_std_err = data_std_err[:x_range]

    plt_x_legend = range(1, len(data_mean) + 1)[:x_range]

    ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    graph_current_data, = ax.plot(plt_x_legend, data_mean, linewidth=1.0,
                                      label="Actor-Critic (32 tilings)".format(num_tilings))
    plt_agent_sweeps.append(graph_current_data)
    
    
    
    
    directory = 'results_sarsa'
    num_tilings = 32
    num_tiles = 8
    update_step_size = 0.2
    epsilon = 0.03
    max_steps = 300000
    avg_reward_step_size = 2**-6
    x_range = 300000
    #plt_xticks = [0, 4999, 9999, 14999, 19999]
    #plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt2_yticks = range(-3, 1, 1)
    load_name = 'semi-gradient_sarsa_tilings_{}_tiledim_{}_update_ss_{}_epsilon_ss_{}_avg_reward_ss_{}_max_steps_{}'.format(num_tilings, num_tiles, update_step_size, epsilon, avg_reward_step_size, max_steps)
    file_type2 = "exp_avg_reward"
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))

    data_mean = np.mean(data, axis=0)
    data_std_err = np.std(data, axis=0) / np.sqrt(len(data))

    data_mean = data_mean[:x_range]
    data_std_err = data_std_err[:x_range]

    plt_x_legend = range(1, len(data_mean) + 1)[:x_range]

    ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    graph_current_data, = ax.plot(plt_x_legend, data_mean, linewidth=1.0,
                                      label="Sarsa (32 tilings)".format(num_tilings))
    plt_agent_sweeps.append(graph_current_data)
    
    
    
    directory = 'results_sarsa'
    num_tilings = 64
    num_tiles = 8
    update_step_size = 0.2
    epsilon = 0.03
    max_steps = 300000
    avg_reward_step_size = 2**-6
    x_range = 300000
    #plt_xticks = [0, 4999, 9999, 14999, 19999]
    #plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt2_yticks = range(-3, 1, 1)
    load_name = 'semi-gradient_sarsa_tilings_{}_tiledim_{}_update_ss_{}_epsilon_ss_{}_avg_reward_ss_{}_max_steps_{}'.format(num_tilings, num_tiles, update_step_size, epsilon, avg_reward_step_size, max_steps)
    file_type2 = "exp_avg_reward"
    data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))

    data_mean = np.mean(data, axis=0)
    data_std_err = np.std(data, axis=0) / np.sqrt(len(data))

    data_mean = data_mean[:x_range]
    data_std_err = data_std_err[:x_range]

    plt_x_legend = range(1, len(data_mean) + 1)[:x_range]

    ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha=0.2)
    graph_current_data, = ax.plot(plt_x_legend, data_mean, linewidth=1.0,
                                      label="Sarsa (64 tilings)".format(num_tilings))
    plt_agent_sweeps.append(graph_current_data)
    
    
    
    ax.legend(handles=[*plt_agent_sweeps])
    #ax.set_xticks(plt_xticks)
    ax.set_yticks(plt2_yticks)

    ax.set_title("Exponential Average Reward per Step ({} Runs)".format(len(data)))
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Exponential Average Reward', rotation=90)
    #ax.set_xticklabels(plt_xlabels)
    ax.set_yticklabels(plt2_yticks)
    #ax.set_xlim([0, 20000])
    ax.set_ylim([-3.5, 0.16])
    
plot_sweep_result()