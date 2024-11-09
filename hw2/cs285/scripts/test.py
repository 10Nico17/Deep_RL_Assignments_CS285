import gym
import matplotlib.pyplot as plt

# Initialize the environment with render_mode="rgb_array"
env = gym.make("Humanoid-v4", render_mode="rgb_array")
env.reset()

for _ in range(100):
    # Get rendered frame as an RGB array
    img = env.render()  # No need to specify mode here

    # Display frame using matplotlib
    if img is not None:
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    # Perform random action
    action = env.action_space.sample()
    env.step(action)

env.close()
