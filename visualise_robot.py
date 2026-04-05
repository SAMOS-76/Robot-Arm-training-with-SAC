import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("SO101/block_stack.xml")
data = mujoco.MjData(model)

print("Model loaded")

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_camera")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 300:
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # viewer.cam.fixedcamid = cam_id
        step_start = time.time()
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)