import gym

# Umgebung mit explizitem render_mode initialisieren
env = gym.make('Ant-v4', render_mode="rgb_array")

# Teste Standard-Rendering
try:
    env.reset()
    print("Testing standard rendering...")
    for _ in range(10):
        img = env.render()  # Verwende render_mode="rgb_array"
        print(f"Rendered image shape: {img.shape}")  # Form des gerenderten Bildes
        action = env.action_space.sample()
        env.step(action)
    print("Standard rendering works!")
except Exception as e:
    print(f"Standard rendering failed: {e}")

# Teste Mujoco Simulation-Rendering (falls verfügbar)
if hasattr(env, 'sim'):
    try:
        env.reset()
        print("Testing Mujoco simulation rendering...")
        for _ in range(10):
            img = env.sim.render(camera_name='track', width=500, height=500)
            print(f"Sim rendered image shape: {img.shape}")
            action = env.action_space.sample()
            env.step(action)
    except Exception as e:
        print(f"Sim rendering failed: {e}")
else:
    print("Sim rendering not supported.")
