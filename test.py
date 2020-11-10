# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:24:01 2020

@author: puyua
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

x = np.array([0.00048828, 0.02821181, 0.05593533, 0.08365885, 0.11138238, 0.1391059, 0.16682943, 0.19455295, 0.22227648 ,0.25  ])
y = np.array([-1.0518806725051235, -0.13552088189617015, -0.15661589307178346, -0.1913462226047708, -0.36116519052518207, -0.548420018888963, -0.6251022274752721, -0.5386156295295035, -0.6337221520147165, -0.6492557524802693])
figure(figsize=(8,6))
plt.plot(x,y+0.06)

plt.xticks(x.round(3), fontsize = 14)
plt.tight_layout()
plt.xlabel("Average Reward Step Size", fontsize = 18)
plt.ylabel("Exponential Average Reward for the Last 5000 Time Steps", fontsize = 16)
plt.title("Effect of Changing Average Reward Step Size", fontsize = 18)
plt.tight_layout()
#plt.savefig('sarsa_avg_reward_step_size_test_50000.png')