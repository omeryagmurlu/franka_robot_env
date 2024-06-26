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

from polymetis import RobotInterface
import torchcontrol as toco
from torchcontrol.planning.min_jerk import generate_joint_space_min_jerk

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
        self.q_desired = torch.nn.Parameter(desired_joint_pos)
        self.qd_desired = torch.nn.Parameter(torch.zeros_like(desired_joint_pos))

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
            q_current, qd_current, self.q_desired, self.qd_desired
        )
        return {"joint_torques": output}


class FrankaArm():
    def __init__(self, name, ip_address, gain_scale=1.0, reset_gain_scale=1.0, hz=10, **kwargs):
        self.name = name
        self.ip_address = ip_address
        self.robot = None
        self.gain_scale = gain_scale
        self.reset_gain_scale = reset_gain_scale

        self.hz = hz
        self.velocity_limits = np.array([[-4*np.pi/2, 4*np.pi/2]] * 7).T / 32
        self.velocity_limits_norm = np.linalg.norm(self.velocity_limits)


    def connect(self, policy=None):
        """Establish hardware connection"""
        connection = False
        # Initialize self.robot interface
        print("Connecting to {}: ".format(self.name), end="")
        try:
            self.robot = RobotInterface(
                ip_address=self.ip_address,
                enforce_version=False
            )
            print("Success")
        except Exception as e:
            self.robot = None # declare dead
            print("Failed with exception: ", e)
            return connection

        print("Testing {} connection: ".format(self.name), end="")
        connection = self.okay()
        if connection:
            print("okay")
            self.reset() # reset the robot before starting operaions
            if policy==None:
                # Create policy instance
                s_initial = self.get_sensors()
                policy = JointPDPolicy(
                    desired_joint_pos=s_initial['joint_pos'],
                    kp=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
                    kd=self.gain_scale * torch.Tensor(self.robot.metadata.default_Kqd),
                )

            # Send policy
            print("\nRunning PD policy...")
            self.robot.send_torch_policy(policy, blocking=False)
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
            if self.robot.is_running_policy(): # Is user controller?
                print("Resetting using user controller")

                if reset_pos == None:
                    reset_pos = torch.Tensor(self.robot.metadata.rest_pose)
                elif not torch.is_tensor(reset_pos):
                    reset_pos = torch.Tensor(reset_pos)

                # Use registered controller
                q_current = self.get_sensors()['joint_pos']
                # generate min jerk trajectory
                dt = 0.1
                waypoints =  generate_joint_space_min_jerk(start=q_current, goal=reset_pos, time_to_go=time_to_go, hz=1/dt)
                # reset using min_jerk traj
                for i in range(len(waypoints)):
                    self.apply_commands(
                            q_desired=waypoints[i]['position'],
                            kp=self.reset_gain_scale * torch.Tensor(self.robot.metadata.default_Kq),
                            kd=self.reset_gain_scale * torch.Tensor(self.robot.metadata.default_Kqd),
                        )
                    time.sleep(dt)

                # reset back gains to gain-policy
                self.apply_commands(
                        kp=self.gain_scale*torch.Tensor(self.robot.metadata.default_Kq),
                        kd=self.gain_scale*torch.Tensor(self.robot.metadata.default_Kqd)
                    )
            else:
                # Use default controller
                print("Resetting using default controller")
                self.robot.go_home(time_to_go=time_to_go)
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

    def apply_commands(self, q_desired=None, qd_desired=None, kp=None, kd=None):
        """Apply hardware commands"""
        udpate_pkt = {}
        if q_desired is not None:
            udpate_pkt['q_desired'] = q_desired if torch.is_tensor(q_desired) else torch.tensor(q_desired)
        if qd_desired is not None:
            udpate_pkt['qd_desired'] = qd_desired if torch.is_tensor(qd_desired) else torch.tensor(qd_desired)
        if kp is not None:
            udpate_pkt['kp'] = kp if torch.is_tensor(kp) else torch.tensor(kp)
        if kd is not None:
            udpate_pkt['kd'] = kd if torch.is_tensor(kd) else torch.tensor(kd)
        assert udpate_pkt, "Atleast one parameter needs to be specified for udpate"

        try:
            self.robot.update_current_policy(udpate_pkt)
        except Exception as e:
            print("1> Failed to udpate policy with exception", e)
            self.reconnect()

    def generate_waypoints_within_limits(self, start, goal, hz, max_vel_norm=float('inf')):
        step_duration = 1/hz
        vel = (goal - start) / step_duration
        vel_norm = np.linalg.norm(vel)

        if vel_norm > max_vel_norm:
            feasible_vel = (vel / vel_norm) * max_vel_norm
        else:
            feasible_vel = vel

        feasible_norm = np.linalg.norm(feasible_vel)

        n_steps = int(np.ceil(vel_norm / feasible_norm))

        t = torch.linspace(0, 1, n_steps+1)[1:]
        waypoints = (1 - t[:, None]) * start + t[:, None] * goal

        return waypoints, feasible_vel

    def go_to_within_limits(self, goal):
        q_initial = self.robot.get_joint_positions().detach().cpu()
        waypoints, feasible_vel = self.generate_waypoints_within_limits(q_initial, goal, self.hz, max_vel_norm=self.velocity_limits_norm)
        print(len(waypoints))
        dwaypoints = torch.diff(waypoints, dim=0)
        for i in range(len(waypoints)):
            self.apply_commands(q_desired=waypoints[i], qd_desired=dwaypoints[i] if i < len(dwaypoints) else torch.zeros_like(feasible_vel))
            time.sleep(1 / self.hz)

    def __del__(self):
        self.close()


if __name__ == "__main__":

    # user inputs
    time_to_go = 1*np.pi
    m = 0.5  # magnitude of sine wave (rad)
    T = 2.0  # period of sine wave
    hz = 50  # update frequency

    # Initialize robot
    franka = FrankaArm(name="Franka-Demo", ip_address='10.10.10.110')

    # connect to robot with default policy
    assert franka.connect(policy=None), "Connection to robot failed."

    # reset using the user controller
    franka.reset()

    Q1 = torch.tensor([-1.3821e-01,  1.9691e-03, -5.3979e-02, -2.0517e+00,  6.1574e-02,
        1.9851e+00, -9.0278e-01])
    Q2 = torch.tensor([ 0.3845,  0.1215,  0.3458, -0.7015,  0.3703,  1.9355, -0.8570])

    action_hz = 1
    actions = [
        Q1,
        Q2,
        Q1,
        Q2*0.5 + Q1*0.5,
        Q1,
        Q2*0.25 + Q1*0.75,
        Q1,
        Q1 + 0.1*torch.randn(Q1.size()),
        Q1 + 0.1*torch.randn(Q1.size()),
        Q1 + 0.1*torch.randn(Q1.size()),
    ]

    for action in actions:
        curr_time = time.time()
        franka.go_to_within_limits(action)
        elapsed = time.time() - curr_time
        slack = 1/action_hz - elapsed
        print("Time taken: ", elapsed, "Slack: ", slack, "Action duration: ", 1/action_hz)
        time.sleep(max(0, slack))

    # # Update policy to execute a sine trajectory on joint 6 for 5 seconds
    # print("Starting sine motion updates...")
    # s_initial = franka.get_sensors()
    # q_initial = s_initial['joint_pos'].clone()
    # q_desired = s_initial['joint_pos'].clone()

    # for i in range(int(time_to_go * hz)):
    #     q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
    #     # q_desired[5] = q_initial[5] + 0.05*np.random.uniform(high=1, low=-1)
    #     # q_desired = q_initial + 0.01*np.random.uniform(high=1, low=-1, size=7)
    #     franka.apply_commands(q_desired = q_desired)
    #     time.sleep(1 / hz)

    # # Udpate the gains
    # kp_new = 0.1* torch.Tensor(franka.robot.metadata.default_Kq)
    # kd_new = 0.1* torch.Tensor(franka.robot.metadata.default_Kqd)
    # franka.apply_commands(kp=kp_new, kd=kd_new)

    # print("Starting sine motion updates again with updated gains.")
    # for i in range(int(time_to_go * hz)):
    #     q_desired[5] = q_initial[5] + m * np.sin(np.pi * i / (T * hz))
    #     franka.apply_commands(q_desired = q_desired)
    #     time.sleep(1 / hz)

    print("Closing and exiting hardware connection")
    franka.close()
