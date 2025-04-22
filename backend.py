from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import uvicorn
import io
from PIL import Image
import numpy as np
import gymnasium as gym
from imitation_gym_wrappers.recorder_wrapper import RecorderWrapper
from imitation_workshop.envs import CustomCarRacing
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import os
import stable_baselines3
from imitation_workshop.iqlearn import IQLearn
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

def convert_img_to_str(img):
    # Convert to base64 for sending over API
    buffered = io.BytesIO()
    img = Image.fromarray(img)
    img.save(buffered, format="JPEG", subsampling=0, quality=100)
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

# Initialize FastAPI
app = FastAPI(title="CarRacing API")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def regularizer(x):
    return x**2

env = RecorderWrapper(CustomCarRacing(render_mode='rgb_array', render_states=False), 10000)
obs = None
reset_pos = 0
current_model = None
auto_mode = False

# Define request models
class StepRequest(BaseModel):
    steering: float  # -1.0 to 1.0
    throttle: float  # -1.0 to 1.0

class SaveRecordingRequest(BaseModel):
    name: str

class ResetRequest(BaseModel):
    env: str = "CarRacing"

class AutoModeRequest(BaseModel):
    agent_file: str
    enable: bool = True

class QNet(nn.Module):
  def __init__(self, env):
    super().__init__()
    self.fc1 = nn.Linear(env.observation_space.shape[0]+env.action_space.shape[0], 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, 32)
    self.fc4 = nn.Linear(32, 1)

  def forward(self, x, a):
    x = torch.cat([x,a], 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5
class ActorNet(nn.Module):
  def __init__(self, env):
    super().__init__()
    self.fc1 = nn.Linear(env.observation_space.shape[0], 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, 32)
    self.fc_mean = nn.Linear(32, np.prod(env.action_space.shape))
    self.fc_logstd = nn.Linear(32, np.prod(env.action_space.shape))
    self.register_buffer(
        "action_scale",
        torch.tensor(
            (env.action_space.high - env.action_space.low) / 2.0,
            dtype=torch.float32,
        ),
    )
    self.register_buffer(
        "action_bias",
        torch.tensor(
            (env.action_space.high + env.action_space.low) / 2.0,
            dtype=torch.float32,
        ),
    )


  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    mean = self.fc_mean(x)
    log_std = self.fc_logstd(x)
    log_std = torch.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
        log_std + 1
    )  # From SpinUp / Denis Yarats
    return mean, log_std

# API routes
@app.post("/reset")
async def reset_game(request: ResetRequest):
    """Reset the game to starting position and return the initial game frame"""
    try:
        global current_model, auto_mode, obs, env, reset_pos
        
        # Reset auto mode if a model was loaded
        if current_model is not None and not auto_mode:
            current_model = None
            
        # Create the appropriate environment based on request
        if request.env == "MountainCar":
            if env is None or not isinstance(env.unwrapped, Continuous_MountainCarEnv):
                # Initialize MountainCar environment
                base_env = Continuous_MountainCarEnv(render_mode='rgb_array')
                env = RecorderWrapper(base_env, 10000)
                reset_pos = 0
        elif request.env == "CarRacing":
            if env is None or not isinstance(env.unwrapped, CustomCarRacing):
                # Initialize CarRacing environment
                base_env = CustomCarRacing(render_mode='rgb_array', render_states=False)
                env = RecorderWrapper(base_env, 10000)
                reset_pos = 0
        else:
            # Default to CarRacing if unspecified or unknown
            if env is None or not isinstance(env.unwrapped, CustomCarRacing):
                base_env = CustomCarRacing(render_mode='rgb_array', render_states=False)
                env = RecorderWrapper(base_env, 10000)
                reset_pos = 0
                
        # Reset the environment
        obs, _ = env.reset()
        image = convert_img_to_str(env.render())
        return {"status": "success", "image": image}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Keep the GET endpoint for backward compatibility
@app.get("/reset")
async def reset_game_get():
    """Reset the game to starting position (CarRacing by default)"""
    try:
        # Call the POST version with default environment
        request = ResetRequest(env="CarRacing")
        return await reset_game(request)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto_mode")
async def set_auto_mode(request: AutoModeRequest):
    """Enable or disable automatic mode with a specific agent"""
    global current_model, auto_mode
    try:
        if request.enable:
            # Construct full path to the agent file
            agent_path = os.path.join('agents', request.agent_file)
            
            # Check if the file exists
            if not os.path.exists(agent_path):
                raise HTTPException(status_code=404, detail=f"Agent file {request.agent_file} not found")
            
            # Load the model
            if agent_path.endswith('.zip'):
                current_model = stable_baselines3.SAC.load(agent_path)
            if agent_path.endswith('.agent'):
                with open(agent_path, 'rb') as f:
                    current_model = pickle.load(f)
                current_model.args.device = 'cpu'
            auto_mode = True
            return {"status": "success", "message": f"Auto mode enabled with agent {request.agent_file}"}
        else:
            # Disable auto mode
            current_model = None
            auto_mode = False
            return {"status": "success", "message": "Auto mode disabled"}
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_agents")
async def list_agents():
    """List available agent files"""
    try:
        # Ensure agents directory exists
        if not os.path.exists('agents'):
            os.makedirs('agents')
        
        # Get list of .zip files in the agents directory
        agents = [f for f in os.listdir('agents') if f.endswith('.zip') or f.endswith('.agent')]
        print(agents)
        return {"status": "success", "agents": agents}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_frames_count")
async def get_frames_count():
    """Returns the number of recorded steps"""
    try:
        return {"status": "success", "count": env.pos}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_reset_pos")
async def set_reset_pos():
    """Set the reset position"""
    try:
        global reset_pos
        reset_pos = env.pos
        return {"status": "success"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_recording")
async def save_recording(request: SaveRecordingRequest):
    """Save the recording under the given name"""
    try:
        env.save_buffer(f"recordings/{request.name}")
        return {"status": "success"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restart_recording")
async def restart_recording():
    """Restart the recording, i.e. discard the last recording"""
    try:
        global env, reset_pos
        env = RecorderWrapper(env.unwrapped, 10000)
        reset_pos = 0
        return {"status": "success"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/discard_last_session")
async def discard_last_session():
    """Reset recording progress to last reset position, discarding the last session"""
    try:
        env.pos = reset_pos
        return {"status": "success"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step_game(request: StepRequest):
    """Take a step in the game with the given controls and return the updated game frame"""
    global current_model, auto_mode, obs, env
    try:
        # If in auto mode and model is loaded, use the model to predict action
        if auto_mode and current_model is not None:
            # Predict action from the model
            action, _ = current_model.predict(obs, deterministic=True)
            
            # Convert from model's action space to our action representation
            print(action)
            
            obs, _, terminated, _, _ = env.unwrapped.step(action) # step unwrapped so no recording
            if terminated:
                obs, _ = env.unwrapped.reset()
            # set previous recording to truncated, should not impact performance
            mod_pos = (env.pos - 1) % env.buffer_size
            env.truncated[mod_pos] = True
            image = convert_img_to_str(env.render())
            return {"status": "success", "image": image, "steering": action[0].item() if env.action_space.shape[0] == 2 else 0.0, "throttle": action[1].item() if env.action_space.shape[0] == 2 else action[0].item(), "auto_mode": True}
        else:
            # Manual control
            if env.action_space.shape[0] == 1:
                action = np.array([2*request.steering])
            else:
                action = np.array([request.steering, request.throttle])

            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                obs, _ = env.reset()
            image = convert_img_to_str(env.render())
            return {"status": "success", "image": image, "steering": action[0] if env.action_space.shape[0] == 2 else 0.0, "throttle": action[1] if env.action_space.shape[0] == 2 else action[0], "auto_mode": False}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
