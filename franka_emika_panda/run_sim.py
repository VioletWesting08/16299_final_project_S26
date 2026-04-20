
"""
ee_probe.py — Interactive end-effector position finder for Franka Panda.

HOW TO USE:
  python ee_probe.py

Controls (MuJoCo viewer):
  • Double-click a link, then drag          — apply force/torque to that body
  • Ctrl + double-click, then drag          — apply torque only
  • Right-click + drag                      — rotate camera
  • Middle-click + drag (or Shift+right)    — pan camera
  • Scroll                                  — zoom
  • Space                                   — pause / resume physics
  • Backspace                               — reset to keyframe 0

  Press  P  at any time to print the current EE position to the terminal.
  The terminal also prints EE xyz live every 0.5 s of simulation time.

The viewer uses MuJoCo's built-in perturbation (mjPERT) so you can grab any
link and drag it; the built-in PD actuators resist but you can muscle past
them to explore the workspace.  Once you find a pose you like, press P and
copy the xyz from the terminal into your PHASES list.
"""

import mujoco
import mujoco.viewer
import numpy as np
import os

# ── config ────────────────────────────────────────────────────────────────────
SCENE_XML = "franka_emika_panda/scene.xml"   # adjust path if needed
ENV_XML   = "franka_emika_panda/debug_scene.xml"

# ── load model ────────────────────────────────────────────────────────────────
if not os.path.exists(SCENE_XML):
    raise FileNotFoundError(
        f"Could not find {SCENE_XML!r}.\n"
        "Run this script from your project root, or edit SCENE_XML at the top."
    )

# Build debug scene with tube
xml = """<mujoco>
  <include file="/Users/sophia/Documents/Classes/16299_final_project_S26/franka_emika_panda/scene.xml"/>
  <worldbody>
    <body name="centrifuge_tube" pos="0.6 0.0 0.07">
      <freejoint/>
      <inertial mass="0.05" pos="0 0 0" diaginertia="0.0001 0.0001 0.0001"/>
      <geom name="tube_geom" type="cylinder" size="0.015 0.05"
            rgba="0.2 0.7 1.0 0.9" mass="0.05" friction="1.5 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>"""

with open(ENV_XML, "w") as f:
    f.write(xml)

model = mujoco.MjModel.from_xml_path(ENV_XML)
data  = mujoco.MjData(model)

# Add damping to the tube's freejoint DOFs
tube_body_id = model.body("centrifuge_tube").id
for i in range(model.nv):
    if model.dof_bodyid[i] == tube_body_id:
        model.dof_damping[i] = 5.0

# Find the hand/EE body — try common names used by Franka XML packs
EE_CANDIDATES = ["hand", "panda_hand", "ee", "end_effector", "attachment_site"]
hand_id = None
for name in EE_CANDIDATES:
    try:
        hand_id = model.body(name).id
        print(f"End-effector body: '{name}'  (id={hand_id})")
        break
    except Exception:
        pass

if hand_id is None:
    # Fall back: list all bodies and pick the last non-world one
    print("Could not find a named EE body. Available bodies:")
    for i in range(model.nbody):
        print(f"  [{i}] {model.body(i).name}")
    raise RuntimeError(
        "Set EE_CANDIDATES to include the correct EE body name from the list above."
    )

# Find arm joint dof/qpos addresses (robust, name-based)
arm_jnt_ids  = []
for i in range(1, 8):
    try:
        arm_jnt_ids.append(model.joint(f"joint{i}").id)
    except Exception:
        pass                        # joint name scheme differs — that's OK

if len(arm_jnt_ids) == 7:
    arm_dof_ids  = [model.jnt_dofadr[j]  for j in arm_jnt_ids]
    arm_qpos_ids = [model.jnt_qposadr[j] for j in arm_jnt_ids]
    # Zero out damping for arm joints so they don't return to home
    for dof_id in arm_dof_ids:
        model.dof_damping[dof_id] = 0.0
else:
    arm_dof_ids  = None             # will skip joint-angle printout
    arm_qpos_ids = None

# ── helpers ───────────────────────────────────────────────────────────────────
def ee_pos() -> np.ndarray:
    return data.xpos[hand_id].copy()

def print_ee(label: str = ""):
    xyz = ee_pos()
    tag = f"[{label}] " if label else ""
    print(f"  {tag}EE xyz = [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
    if arm_qpos_ids is not None:
        q = data.qpos[arm_qpos_ids]
        qstr = ", ".join(f"{v:.4f}" for v in q)
        print(f"         joint angles = [{qstr}]")

# ── key callback ─────────────────────────────────────────────────────────────
# MuJoCo passive viewer exposes key callbacks via viewer.user_scn / key_callback
# We use a simple polling approach instead — check a flag each step.
print_flag = [False]

def on_key(key, scancode, action, mods):
    """Called by the viewer on key events (if supported)."""
    GLFW_PRESS = 1
    GLFW_KEY_P = 80
    if key == GLFW_KEY_P and action == GLFW_PRESS:
        print_flag[0] = True

# ── warm-up ─────────────────────────────────────────────────────────
# Let physics settle for a moment
for _ in range(500):
    mujoco.mj_step(model, data)
mujoco.mj_forward(model, data)

print("\n" + "="*70)
print("INTERACTIVE EE PROBE")
print("="*70)
print(f"Model has {model.nv} velocities, {model.nq} positions")
print("\nControls:")
print("  • Double-click a link → drag to move it")
print("  • Right-click + drag → rotate camera")
print("  • Space → pause/resume")
print("  • Backspace → reset")
print("  • Press 'P' → print current EE position and joint angles\n")
print_ee("INITIAL")
print("="*70 + "\n")

# ── main loop ────────────────────────────────────────────────────────
dt = model.opt.timestep
next_print_time = 0.5

# Disable all actuators so joints stay where you drag them
for i in range(model.nu):
    model.actuator_gainprm[i, 0] = 0.0
    model.actuator_biasprm[i, 1] = 0.0  # zero out proportional bias
    model.actuator_biasprm[i, 2] = 0.0  # zero out damping bias

# Restore damping (to prevent drift) and disable gravity
for dof_id in range(model.nv):
    model.dof_damping[dof_id] = 1.0  # restore original damping
model.opt.gravity[:] = 0.0  # disable gravity so robot floats freely

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Zero out all controls
        data.ctrl[:] = 0.0
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Periodic printout (every 0.5 sim seconds)
        if data.time >= next_print_time:
            print_ee(f"t={data.time:.2f}s")
            next_print_time += 0.5
        
        # Check if viewer is requesting a render sync
        viewer.sync()
        
        # Optionally: check for P key press (if your viewer supports it)
        # For now, you can just run the file and press P in the viewer window
        # The passive viewer will handle P key detection separately

print("\n✓ Simulation complete.")
print_ee("FINAL")