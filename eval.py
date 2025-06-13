import sys
import os
if sys.platform == "linux":
    print("Setting MUJOCO_GL to egl")
    os.environ["MUJOCO_GL"] = "egl"

import time
import orbax.checkpoint
import jax
import jax.numpy as jnp
import equinox
import rerun as rr
import gymnasium as gym
import numpy as np
import gym_aloha
import json
from jax.sharding import NamedSharding, Mesh
from jax.experimental import mesh_utils


from tqdm import tqdm
import models.pi0 as pi0
from tokenizer import Tokenizer
import cv2
from datetime import datetime
import model_utils.utils.rerun as rerun_utils

folder_path = os.path.dirname(os.path.abspath(__file__))

print("Initializing setup...")

print("Tokenizing prompt...")
prompt = "Transfer cube"
tokenizer = Tokenizer()
language_token_ids, language_mask = tokenizer.tokenize(prompt)
language_token_ids = language_token_ids[language_mask]
print(f"Tokenized prompt into {len(language_token_ids)} tokens")

print("Loading normalization stats...")
norm_stats = json.load(open(os.path.join(folder_path, "pi0_aloha_sim/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json")))["norm_stats"]
print("Normalization stats loaded successfully")

print("Loading model checkpoint...")
is_value = lambda x: isinstance(x, dict) and "value" in x and isinstance(x["value"], jax.Array)
checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.StandardCheckpointHandler())
fine_tuned_checkpoint = checkpointer.restore(os.path.join(folder_path, "pi0_aloha_sim/params"))
fine_tuned_params = fine_tuned_checkpoint["params"]
fine_tuned_params = jax.tree.map(lambda x: x["value"], fine_tuned_params, is_leaf=is_value)
print("Model checkpoint loaded successfully")
print("Initializing model...")
config = pi0.Config.default()
fine_tuned_pi0_model = pi0.load(fine_tuned_params)
fine_tuned_pi0_model = equinox.nn.inference_mode(fine_tuned_pi0_model)

# mesh = Mesh(mesh_utils.create_device_mesh((1,)), ('x',))
# sharding = NamedSharding(mesh, jax.sharding.PartitionSpec(None))
# base_checkpoint = checkpointer.restore(
#     os.path.join(folder_path, "pi0_base/params"),
#     args=orbax.checkpoint.args.StandardRestore(fallback_sharding=sharding),
# )
# base_params = base_checkpoint["params"]
# base_pi0_model = pi0.load(base_params)
# base_pi0_model = equinox.nn.inference_mode(base_pi0_model)

print("Model initialized successfully")

print("Defining utility functions...")
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)
def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val
def gripper_to_angular(value):
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)
def gripper_from_angular(value):
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)
def joint_flip_mask() -> np.ndarray:
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
def env_to_model_state(state: np.ndarray) -> np.ndarray:
    state = joint_flip_mask() * state
    state[[6, 13]] = gripper_to_angular(state[[6, 13]])
    state = jnp.pad(state, (0, config.action_dim - state.shape[0]))
    state = (state - jnp.array(norm_stats["state"]["mean"])) / jnp.array(norm_stats["state"]["std"])
    state = jnp.nan_to_num(state)
    return state    
def model_to_env_actions(actions: jnp.ndarray) -> np.ndarray:
    actions = jax.vmap(lambda action: action * jnp.array(norm_stats["actions"]["std"]) + jnp.array(norm_stats["actions"]["mean"]))(actions)
    actions = actions[:, :14]
    actions = joint_flip_mask() * actions
    actions = actions.at[:, [6, 13]].set(gripper_from_angular(actions[:, [6, 13]]))
    return actions
print("Utility functions defined successfully")

print("Initializing visualization and environment...")
recording = rr.RecordingStream("train_flow", recording_id=f"{sys.platform}-{time.time()}")
recording.save("train_flow.rrd")
if sys.platform == "darwin":
    recording.spawn()
else:
    rerun_utils.connect_grpc(recording, port=9876, detach_process=True)

episode_reward_list = []
for i in tqdm(range(1000), position=0):
    episode_reward = 0
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    print("Environment initialized successfully")
    print("Initial observation structure:")
    observation, info = env.reset()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"aloha_execution_{timestamp}.mp4"
    frame_size = (observation["pixels"]["top"].shape[1], observation["pixels"]["top"].shape[0])
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    print(f"Video writer initialized. Output will be saved to {video_path}")


    print("Starting main execution loop...")
    cfg = 3.0
    print(f"Using cfg scale: {cfg}")
    for step in tqdm(range(1000), position=1):
        image = observation["pixels"]["top"]
        original_image = image.copy()
        image = (image / 255.0) * 2 - 1
        image = jax.image.resize(image, (224, 224, 3), method="bilinear")
        
        state = observation["agent_pos"]
        original_state = state.copy()
        state = env_to_model_state(state)
        
        # actions, deviation = equinox.filter_jit(pi0.sample_actions)(
        #     positive_model=fine_tuned_pi0_model,
        #     negative_model=base_pi0_model,
        #     language_token_ids=language_token_ids,
        #     image=image,
        #     state=state,
        #     key=jax.random.PRNGKey(0),
        #     sample_steps=10,
        #     cfg_scale=cfg
        # )
        
        actions, deviation = equinox.filter_jit(fine_tuned_pi0_model.sample_actions)(
            language_token_ids=language_token_ids,
            image=image,
            state=state,
            key=jax.random.PRNGKey(0),
            sample_steps=10,
            cfg_scale=cfg
        )
        
        # print(f"Deviation: {deviation}")
        actions = model_to_env_actions(actions)
        
        # Take step in environment and log images for each action in the horizon
        for i in range(50):
            rr.log("image", rr.Image(original_image))
            # Write frame to video
            video_writer.write(original_image)
            observation, timestep_reward, terminated, truncated, info = env.step(actions[i])
            episode_reward += timestep_reward
            original_image = observation["pixels"]["top"]
            if terminated or truncated:
                break
        if terminated or truncated:
            break
        
    # Clean up
    video_writer.release()
    env.close()
    print(f"Episode reward: {episode_reward}")
    
    episode_reward_list.append(episode_reward)
    print(f"Average episode reward: {np.mean(episode_reward_list)}")
    print(f"Max episode reward: {np.max(episode_reward_list)}")
    print(f"Min episode reward: {np.min(episode_reward_list)}")

print("Execution completed successfully")
print(f"Video saved to {video_path}")
