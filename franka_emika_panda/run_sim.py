import mujoco
import mujoco.viewer
import time

# 1. Load the model from the XML file and create a data object
# Use 'scene.xml' to get the robot + floor + lighting
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 2. Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Run until the viewer window is closed
    while viewer.is_running():
        
        # Step the physics simulation forward
        # Apply a control signal to the first joint actuator
        mujoco.mj_step(model, data)
        
        # Sync the viewer to reflect the new physics state
        viewer.sync()
        
        # Sleep to match the simulation to real-time
        time.sleep(model.opt.timestep)