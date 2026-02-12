import warp as wp

@wp.kernel
def compute_thruster_mapping(
    actions: wp.array(dtype=wp.vec3f), 
    thrust_action: wp.array(dtype=wp.vec3f, ndim=2), 
    max_thrust: float # Passed explicitly
):
    tid = wp.tid()
    
    # Load the action for this environment
    act = actions[tid]
    
    # Initialize local thruster values
    t0 = 0.0
    t1 = 0.0
    t2 = 0.0
    t3 = 0.0
    t4 = 0.0
    t5 = 0.0
    t6 = 0.0
    t7 = 0.0
    
    # Forward/Back (actions[0])
    if act[0] != 0.0:
        mag = wp.abs(act[0]) * max_thrust
        if act[0] > 0.0:
            t1 += mag
            t6 += mag
        else:
            t2 += mag
            t5 += mag

    # Left/Right (actions[1])
    if act[1] != 0.0:
        mag = wp.abs(act[1]) * max_thrust
        if act[1] > 0.0:
            t0 += mag
            t3 += mag
        else:
            t4 += mag
            t7 += mag

    # Yaw (actions[2])
    if act[2] != 0.0:
        mag = wp.abs(act[2]) * max_thrust
        if act[2] < 0.0: 
            t0 += mag
            t2 += mag
            t4 += mag
            t6 += mag
        else:
            t1 += mag
            t3 += mag
            t5 += mag
            t7 += mag

    # Clamp and Write to Global Memory
    thrust_action[tid, 0] = wp.vec(0.0, 0.0, wp.clamp(t0, 0.0, 1.0))
    thrust_action[tid, 1] = wp.vec(0.0, 0.0, wp.clamp(t1, 0.0, 1.0))
    thrust_action[tid, 2] = wp.vec(0.0, 0.0, wp.clamp(t2, 0.0, 1.0))
    thrust_action[tid, 3] = wp.vec(0.0, 0.0, wp.clamp(t3, 0.0, 1.0))
    thrust_action[tid, 4] = wp.vec(0.0, 0.0, wp.clamp(t4, 0.0, 1.0))
    thrust_action[tid, 5] = wp.vec(0.0, 0.0, wp.clamp(t5, 0.0, 1.0))
    thrust_action[tid, 6] = wp.vec(0.0, 0.0, wp.clamp(t6, 0.0, 1.0))
    thrust_action[tid, 7] = wp.vec(0.0, 0.0, wp.clamp(t7, 0.0, 1.0))