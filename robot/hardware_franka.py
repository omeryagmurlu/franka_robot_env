""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from typing import Dict, Sized
import time

import numpy as np
from numpy.core.fromnumeric import size
import torch
import bezier 

from polymetis import RobotInterface
import torchcontrol as toco
from torchcontrol.planning import generate_joint_space_min_jerk
from torchcontrol.policies.impedance import HybridJointImpedanceControl

import argparse


def normalize_0_1(lower: float = 0.0, upper: float = 255.0):
    def f(arr):
        arr = arr.clip(lower, upper)
        arr -= arr.min()
        arr /= arr.max()
        return arr

    return f

USE_MOVE_TO_JOINT_POSITIONS = False
BEZIER_NODES = np.array(
    [
        [-1, 1],
        [-0.5, 1],
        [0, 1],
        [0.45, 1],
        [0.5, 0.5],
        [0.55, 0],
        [1, 0],
        [1.5, 0],
        [2, 0],
    ]
)
BEZIER_NODES[:, 0] = normalize_0_1(-1, 2)(BEZIER_NODES[:, 0])
MOTION_PROFILE = bezier.Curve(BEZIER_NODES.T, degree=len(BEZIER_NODES) - 1)


# linear interpolation: w_a * a + (1 - w_a) * b
def lerp(a: torch.Tensor, b: torch.Tensor, w_a: torch.Tensor):
    w_a = w_a.unsqueeze(1)
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    w_b = 1 - w_a
    return w_a * a + w_b * b

class JointPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, desired_joint_pos, kp, kd, **kwargs):
        """
        Args:
            desired_joint_pos (int):    Number of steps policy should execute
            hz (double):                Frequency of controller
            kp, kd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.kp = torch.nn.Parameter(kp)
        self.kd = torch.nn.Parameter(kd)
        self.joint_pos_desired = torch.nn.Parameter(desired_joint_pos)

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(self.kp, self.kd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]
        self.feedback.Kp = torch.diag(self.kp)
        self.feedback.Kd = torch.diag(self.kd)

        # Execute PD control
        output = self.feedback(
            q_current,
            qd_current, 
            self.joint_pos_desired, 
            torch.zeros_like(qd_current)
        )
        # print("#######################################")
        # print(self.joint_pos_desired - q_current)
         #print(f'des q: {self.joint_pos_desired[3]}')

        # print(q_current)
        # print("#######################################")
        return {"joint_torques": output}



class FrankaArm():
    def __init__(self, name, ip_address, gain_scale=1.0, reset_gain_scale=1.0, default_reset_pose=None, hz=1, **kwargs):
        self.name = name
        self.ip_address = ip_address
        self.robot = None
        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale
        self.default_reset_pose = default_reset_pose
        self.hz = hz
        self.old_joint = None
        self.q_error = torch.zeros(7)
        self.velocity_limits = np.array([[-4*np.pi/2, 4*np.pi/2]] * 7).T # from robohive
        self.velocity_limits /= 16 # let's be safe, these are joint limits and we actually need to limit workspace velocity, so be conservative
        self.velocity_limits_norm = np.linalg.norm(self.velocity_limits)

    def clip_velocity_limits(self, q_des_pos):
        step_duration = 1/self.hz
        q_curr = self.robot.get_joint_positions().detach().cpu().numpy()
        desired_vel = (q_des_pos - q_curr) / step_duration

        des_norm = np.linalg.norm(desired_vel)

        if des_norm > self.velocity_limits_norm:
            feasible_vel = (desired_vel / des_norm) * self.velocity_limits_norm
        else:
            feasible_vel = desired_vel

        feasible_norm = np.linalg.norm(feasible_vel)

        # feasible_vel = np.clip(desired_vel, self.velocity_limits[0], self.velocity_limits[1])
        return q_curr + feasible_vel * step_duration, des_norm / feasible_norm

    def connect(self, policy=None, home_pose=None):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = RobotInterface(
                ip_address=self.ip_address,
                enforce_version=False,
            )
            # self.robot.start_joint_impedance()
            print("Success")
        except Exception as e:
            self.robot = None # declare dead
            print("Failed with exception: ", e)
            return connection

        print("Testing {} connection: ".format(self.name), end="")
        connection = self.okay()
        if connection:
            print("okay")
            print(self.robot.get_joint_positions())
            if self.default_reset_pose is not None:
                self.robot.set_home_pose(torch.Tensor(self.default_reset_pose))
            self.reset() # reset the robot before starting operaions
            # if policy==None:
            #     # Create policy instance
            #     s_initial = self.get_sensors()
            #     policy = JointPDPolicy(
            #         desired_joint_pos=s_initial['joint_pos'],
            #         kp=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
            #         kd=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kqd),
            #     )

            # # Send policy
            # print("\nRunning PD policy...")
            # self.robot.send_torch_policy(policy, blocking=False)
        else:
            print("Not ready. Please retry connection")

        return connection


    def okay(self):
        """Return hardware health"""
        okay = False
        if self.robot:
            try:
                state = self.robot.get_robot_state()
                delay = time.time() - (state.timestamp.seconds + 1e-9 * state.timestamp.nanos)
                assert delay < 5, "Acquired state is stale by {} seconds".format(delay)
                okay = True
            except:
                self.robot = None # declare dead
                okay = False
        return okay


    def close(self):
        """Close hardware connection"""
        if self.robot:
            print("Terminating PD policy: ", end="")
            try:
                self.reset()
                state_log = self.robot.terminate_current_policy()
                print("Success")
            except:
                # print("Failed. Resetting directly to home: ", end="")
                print("Resetting Failed. Exiting: ", end="")
            self.robot = None
            print("Done")
        return True


    def reconnect(self):
        print("Attempting re-connection")
        self.connect()
        while not self.okay():
            self.connect()
            time.sleep(2)
        print("Re-connection success")


    def reset(self, reset_pos=None, time_to_go=5):
        """Reset hardware"""
        if self.okay():
            if reset_pos is None:
                if self.default_reset_pose is None:
                    reset_pos = torch.Tensor(self.robot.metadata.rest_pose)
                else:
                    reset_pos = torch.Tensor(self.default_reset_pose)
            elif not torch.is_tensor(reset_pos):
                # pass
                reset_pos = torch.Tensor(reset_pos)
            print(reset_pos)
            # self.robot.move_to_joint_positions(reset_pos)
            waypoints =  generate_joint_space_min_jerk(start=self.robot.get_joint_positions(), goal=reset_pos, time_to_go=time_to_go, hz=1)
            # reset using min_jerk traj
            for i in range(len(waypoints)):
                self.robot.move_to_joint_positions(waypoints[i]['position'])
            if self.robot.is_running_policy():
                print("Policy is already running. Will restart with new policy")
            policy = HybridJointImpedanceControl(
                self.robot.get_joint_positions(),
                # kp=torch.tensor([300.0, 300.0, 300.0, 300.0, 150.0, 90.0, 30.0])*0.4,
                # kd=torch.tensor([20.0 ,20.0, 20.0, 20.0, 12.0, 10.0, 6.0])*0.4,
                Kq=self.robot.Kq_default*1.5,
                Kqd=self.robot.Kqd_default*1,
                Kx=self.robot.Kx_default*1,
                Kxd=self.robot.Kxd_default*1,
                robot_model=self.robot.robot_model,
                ignore_gravity=self.robot.use_grav_comp
            )
            self.robot.send_torch_policy(policy,blocking=False)
            print("Policy initialized.")
        else:
            print("Can't connect to the robot for reset. Attemping reconnection and trying again")
            self.reconnect()
            self.reset(reset_pos, time_to_go)

    def get_sensors(self):
        """Get hardware sensors"""
        try:
            joint_pos = self.robot.get_joint_positions()
            joint_vel = self.robot.get_joint_velocities()
        except:
            print("Failed to get current sensors: ", end="")
            self.reconnect()
            return self.get_sensors()
        return {'joint_pos': joint_pos, 'joint_vel':joint_vel}

    def apply_commands(self, q_desired=None, kp=None, kd=None, direct=False):
        """Apply hardware commands"""
        udpate_pkt = {}
        if q_desired is not None:
            udpate_pkt['q_desired'] = toco.utils.to_tensor(q_desired)
        if kp is not None:
            udpate_pkt['kp'] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt['kd'] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        try:
            # tt1
            # self.robot.move_to_joint_positions(q_desired)
            if not self.robot.is_running_policy():
                self.reset()
            self.robot.update_desired_joint_positions(toco.utils.to_tensor(q_desired))
     
        except Exception as e:
            print("1> Failed to udpate policy with exception", e)
            self.reconnect()

    def follow_trajectory(
            self,
        joint_positions: torch.Tensor,
        time_to_go: float = 4.0,
        fps: int = 100,
        converge_pos_err: bool = False,
        eps: float = 1e-2,
        I_gain: float = 0.035,
        I_clip: float = 0.1,
        torque_limit: float = 15.0,
    ):
        joint_positions = toco.utils.to_tensor(joint_positions)
        if not hasattr(self, "I_term"):
            self.I_term = torch.zeros(7)
        sleep_period = 1.0 / fps
        n_points = round(time_to_go * fps)
        xs = np.linspace(0, 1, n_points)
        xs, ys = MOTION_PROFILE.evaluate_multi(xs)
        ys = torch.from_numpy(ys)
        start = self.robot.get_joint_positions()
        joint_cmds = lerp(start, joint_positions, ys)
        if converge_pos_err:
            joint_cmds = joint_cmds + self.I_term
        for i in range(len(joint_cmds)):
            tstart = time.time()
            # print(f'at interpolation step {i} at time {tstart}')
            cmd = joint_cmds[i]
            self.robot.update_desired_joint_positions(cmd)
            tau = torch.Tensor(self.robot.get_robot_state().motor_torques_external)
            if tau.norm() > torque_limit * 2:
                raise RuntimeError("Torque limit exceeded during motion")
            while time.time() - tstart < sleep_period:
                pass
        if converge_pos_err:
            # integral controller to drive down steady state error
            err = torch.ones(1)  # position error placeholder
            tau = torch.zeros(7)  # torque placeholder
            while err.norm() > eps and tau.norm() < torque_limit:
                err = self.robot.get_joint_positions() - joint_positions
                tau = torch.Tensor(self.robot.get_robot_state().motor_torques_external)
                # logging.info(f"tau: {tau.norm()}")
                self.I_term += -I_gain * err
                self.I_term = torch.clamp(self.I_term, -I_clip, I_clip)
                cmd = joint_positions + self.I_term
                self.robot.update_desired_joint_positions(cmd)
                while time.time() - tstart < sleep_period:
                    pass
            # # minimize external torque
            # while tau.norm() > torque_limit:
            #     tau = torch.Tensor(self.get_robot_state().motor_torques_external)
            #     logging.info(f"tau: {tau.norm()}")
            #     self.I_term += -torque_comp_gain * tau
            #     self.I_term = torch.clamp(self.I_term, -I_clip, I_clip)
            #     cmd = joint_positions + self.I_term
            #     self.update_desired_joint_positions(cmd)
            #     time.sleep(sleep_period)

    def reset_I_term(self):
        self.I_term = torch.zeros(7)


    def __del__(self):
        self.close()


# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Polymetis based Franka client")

    parser.add_argument("-i", "--server_ip",
                        type=str,
                        help="IP address or hostname of the franka server",
                        default="localhost") # 10.0.0.123 # "169.254.163.91",
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # user inputs
    time_to_go = 1*np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    franka = FrankaArm(name="Franka-Demo", ip_address=args.server_ip)

    # connect to robot with default policy
    assert franka.connect(policy=None), "Connection to robot failed."

    # reset using the user controller
    franka.reset()

    # Update policy to execute a sine trajectory on joint 6 for 5 seconds
    print("Starting sine motion updates...")
    s_initial = franka.get_sensors()
    q_initial = s_initial['joint_pos'].clone()
    q_desired = s_initial['joint_pos'].clone()

    for i in range(int(time_to_go * hz)):
        q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
        # q_desired[5] = q_initial[5] + 0.05*np.random.uniform(high=1, low=-1)
        # q_desired = q_initial + 0.01*np.random.uniform(high=1, low=-1, size=7)
        franka.apply_commands(q_desired = q_desired)
        time.sleep(1 / hz)

    # Udpate the gains
    # maybe change here
    kp_new = 1.5* torch.Tensor(franka.robot.metadata.default_Kq)
    kd_new = 1.5* torch.Tensor(franka.robot.metadata.default_Kqd)
    franka.apply_commands(kp=kp_new, kd=kd_new)

    print("Starting sine motion updates again with updated gains.")
    for i in range(int(time_to_go * hz)):
        q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
        franka.apply_commands(q_desired = q_desired)
        time.sleep(1 / hz)

    print("Closing and exiting hardware connection")
    franka.close()


# old stuff from 
# self.q_error = self.robot.get_joint_positions() - toco.utils.to_tensor(q_desired)
            # self.robot.update_desired_joint_positions(q_desired)
            # # tt3
            # if q_desired is not None and kp is None and kd is None:
            #     joint_pos_current = self.robot.get_joint_positions()
            #     joint_pos_desired = torch.Tensor(q_desired)
            #     ttg = self.robot._adaptive_time_to_go(joint_pos_desired - joint_pos_current)
            #     waypoints =  generate_joint_space_min_jerk(start=joint_pos_current, goal=joint_pos_desired, time_to_go=ttg, hz=self.hz)
            #     # reset using min_jerk traj
            #     for i in range(len(waypoints)):
            #         self.robot.update_current_policy({
            #             'q_desired': waypoints[i]['position'] if torch.is_tensor# (waypoints[i]['position']) else torch.tensor(waypoints[i]['position']),
            #         })
            #         time.sleep(1/self.hz)
            # else:
            
            # if direct:
            #     # tt2
            #     self.robot.update_current_policy(udpate_pkt)
            # else:
            #     # tt4
            #     waypoints =  generate_joint_space_min_jerk(start=self.robot.get_joint_positions(), goal=torch.Tensor(q_desired), time_to_go=0.5, hz=12)
            #     for i in range(len(waypoints)):
            #         print(waypoints[i]['position'], len(waypoints))
            #         self.robot.update_current_policy({
            #             'q_desired': waypoints[i]['position'] if torch.is_tensor(waypoints[i]['position']) else torch.tensor(waypoints[i]['position']),
            #         })
            #         time.sleep(1/12)
            # self.robot.update_current_policy(udpate_pkt)   

#####
# old reset stuff
 # if self.robot.is_running_policy(): # Is user controller?
            #     print("Resetting using user controller")

            #     if reset_pos == None:
            #         if self.default_reset_pose is None:
            #             reset_pos = torch.Tensor(self.robot.metadata.rest_pose)
            #         else:
            #             reset_pos = torch.Tensor(self.default_reset_pose)
            #     elif not torch.is_tensor(reset_pos):
            #         reset_pos = torch.Tensor(reset_pos)

            #     # Use registered controller
            #     q_current = self.get_sensors()['joint_pos']
            #     # generate min jerk trajectory
            #     dt = 0.1
            #     waypoints =  generate_joint_space_min_jerk(start=q_current, goal=reset_pos, time_to_go=time_to_go, hz=1/dt)
            #     # reset using min_jerk traj
            #     for i in range(len(waypoints)):
            #         self.apply_commands(
            #                 q_desired=waypoints[i]['position'],
            #                 kp=self.reset_gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
            #                 kd=self.reset_gain_scale * torch.Tensor(self.robot.metadata.default_Kqd),
            #                 direct=True
            #             )
            #         time.sleep(dt)

            #     # reset back gains to gain-policy
            #     self.apply_commands(
            #             kp=self.gain_scale*torch.Tensor(self.robot.metadata.default_Kq),
            #             kd=self.gain_scale*torch.Tensor(self.robot.metadata.default_Kqd),
            #             direct=True
            #         )
            # else:
            #     # Use default controller
            #     print("Resetting using default controller")
            #     self.robot.go_home(time_to_go=time_to_go)