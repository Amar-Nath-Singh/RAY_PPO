import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
import sys

#folder = 'good_data/'
folder = 'saved_rudder_npy/'
ep = int(sys.argv[1])
data = np.load(folder + f'EP_{ep}.npy')  # TimeStamps x Num_agents x States
no_motion_mask = data[:, 0, 0] > 0
data = data[no_motion_mask]
total_time_stamps, Num_agents, _ = data.shape
color = list(cm.rainbow(np.linspace(0, 1, Num_agents)))
scale = 2
x_ship = scale * np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
y_ship = scale * 16.1 / 230 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])

def plot_agent(idx, timestamp):
    u, v, r, x, y, yaw, _, _, x_init, y_init, x_goal, y_goal = data[timestamp, idx]
    x_new = x + x_ship * np.cos(yaw) - y_ship * np.sin(yaw)
    y_new = y + x_ship * np.sin(yaw) + y_ship * np.cos(yaw)

    x_wp = [x_goal, x_init]
    y_wp = [y_goal, y_init]
    color_idx = (idx + 4) if Num_agents > 4 else idx
    if color_idx >= Num_agents:
        color_idx = Num_agents - color_idx - 1
    print(idx, color_idx)
    plt.plot(x_wp, y_wp, c=color[idx])
    plt.scatter(x_wp, y_wp, c=color[color_idx])
    plt.plot(x_new, y_new, c=color[color_idx])

def plot_path(init_time_stamp, end_time_stamp):
    for i, agent in enumerate(data[init_time_stamp:end_time_stamp].transpose(1, 2, 0)):
        x = agent[3]
        y = agent[4]
        plt.plot(x, y, c=color[i])

# Calculate the bounds dynamically
all_x = data[:, :, 3].flatten()
all_y = data[:, :, 4].flatten()
x_min, x_max = np.min(all_x), np.max(all_x)
y_min, y_max = np.min(all_y), np.max(all_y)
margin = 10  # Add a margin around the edges

for time_stamp in range(0, total_time_stamps, 3):
    plt.clf()
    plt.title(f'Time Stamp: {time_stamp}')
    plt.axis([x_min - margin, x_max + margin, y_min - margin, y_max + margin])
    plt.gca().set_aspect('equal', adjustable='box')
    plot_path(0, time_stamp)
    for agent in range(Num_agents):
        plot_agent(agent, time_stamp)
    plt.pause(0.1)

print('Done')
plt.title('Done')
plt.show()

