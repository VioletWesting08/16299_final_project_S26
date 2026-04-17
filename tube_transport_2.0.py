import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# 1. GENERATE THE NEW ENVIRONMENT
# We create a new XML file that includes the original scene and adds a tube
xml_content = """
<mujoco>
    <include file="scene.xml"/>
    
    <worldbody>
        <body name="centrifuge_tube" pos="0.6 0.0 0.1">
            <freejoint/>
            <geom type="cylinder" size="0.02 0.05" rgba="0.2 0.6 1 1" mass="0.05" condim="4" friction="1 0.05 0.0001"/>
        </body>
    </worldbody>
</mujoco>
"""

# Save it to disk temporarily so MuJoCo can resolve the <include> path correctly
env_path = "franka_emika_panda/pick_and_place_scene.xml"
with open(env_path, "w") as f:
    f.write(xml_content)

# 2. INITIALIZE MUJOCO
model = mujoco.MjModel.from_xml_path(env_path)
data = mujoco.MjData(model)

# Get Body IDs
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
if hand_id == -1:
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")

# 3. DEFINE THE STATE MACHINE
# A sequence of movements. Gripper target is in meters (0.04 is fully open, 0.00 is fully closed)
phases = [
    {"name": "1. Hover above tube",  "target_xyz": [0.6, 0.0, 0.30], "gripper": 0.04, "time": 3.0},
    {"name": "2. Descend to tube",   "target_xyz": [0.6, 0.0, 0.12], "gripper": 0.04, "time": 2.0},
    {"name": "3. Grasp the tube",    "target_xyz": [0.6, 0.0, 0.12], "gripper": 0.00, "time": 1.0}, # Wait while closing
    {"name": "4. Lift the tube",     "target_xyz": [0.6, 0.0, 0.40], "gripper": 0.00, "time": 2.0},
    {"name": "5. Transport to goal", "target_xyz": [0.6, 0.4, 0.40], "gripper": 0.00, "time": 3.0},
]

def get_current_phase(elapsed_time):
    """Determines which phase of the state machine we are currently in."""
    cumulative_time = 0.0
    for phase in phases:
        cumulative_time += phase["time"]
        if elapsed_time < cumulative_time:
            return phase
    return phases[-1] # Return the last phase if time has run out

# Step once to initialize
mujoco.mj_step(model, data)

# 4. RUN THE SIMULATION
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    start_time = time.time()
    current_phase_name = ""
    
    while viewer.is_running():
        elapsed_time = time.time() - start_time
        
        # --- A. STATE MACHINE LOGIC ---
        phase = get_current_phase(elapsed_time)
        
        if phase["name"] != current_phase_name:
            current_phase_name = phase["name"]
            print(f"Executing Phase: {current_phase_name}")
            
        target_pos = np.array(phase["target_xyz"])
        gripper_ctrl = phase["gripper"]
        
        # --- B. TASK SPACE CONTROLLER (INVERSE KINEMATICS) ---
        # Get current XYZ position of the hand
        current_pos = data.xpos[hand_id]
        
        # Calculate spatial error (how far to move)
        dx = target_pos - current_pos
        
        # Compute the Jacobian for the hand
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        
        # We only care about XYZ position and the 7 arm joints
        J = jacp[:, :7]
        
        # Pseudo-inverse to find required joint velocities
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ dx
        
        # --- C. APPLY CONTROLS ---
        # Move the arm joints (with a 0.5 gain for smooth, damped movement)
        data.ctrl[:7] = data.qpos[:7] + dq * 0.5
        
        # Command the gripper fingers (assuming actuators 7 and 8 are the fingers)
        # We apply the gripper command to both fingers equally
        if model.nu >= 9:
            data.ctrl[7] = gripper_ctrl
            data.ctrl[8] = gripper_ctrl
        
        # Step physics and render
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

# Cleanup the temporary XML file when done
if os.path.exists(env_path):
    os.remove(env_path)