#!/usr/bin/env python

# Author: Junxiang Wang
# Date: 2024-04-12

# (C) Copyright 2024-2025 Johns Hopkins University (JHU), All Rights Reserved.

# --- begin cisst license - do not edit ---

# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.

# --- end cisst license ---

"""For instructions, see https://dvrk.readthedocs.io, and search for \"dvrk_teleoperation\""""

import argparse
import crtk
from enum import Enum
import geometry_msgs.msg
import math
import numpy
import PyKDL
import std_msgs.msg
import sys
import time

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, ral, master, puppet, clutch_topic, run_period, align_mtm, operator_present_topic = ""):
        print('Initialzing dvrk_teleoperation for {} and {}'.format(master.name, puppet.name))
        self.ral = ral
        self.run_period = run_period

        self.master = master
        self.puppet = puppet

        self.scale = 0.2

        self.gripper_max = 60 * math.pi / 180
        self.gripper_zero = 0.0 # Set to e.g. 20 degrees if gripper cannot close past zero
        self.jaw_min = -20 * math.pi / 180
        self.jaw_max = 80 * math.pi / 180
        self.jaw_rate = 2 * math.pi

        self.can_align_mtm = align_mtm

        # slowly eliminate alignment offset if we can align mtm,
        # otherwise maintain fixed initial alignment offset
        self.align_rate = 0.25 * math.pi if self.can_align_mtm else 0.0

        # don't require alignment before beginning teleop if mtm wrist can't be actuated
        self.operator_orientation_tolerance = 5 * math.pi / 180 if self.can_align_mtm else math.pi
        self.operator_gripper_threshold = 5 * math.pi / 180
        self.operator_roll_threshold = 3 * math.pi / 180

        self.gripper_to_jaw_scale = self.jaw_max / (self.gripper_max - self.gripper_zero)
        self.gripper_to_jaw_offset = -self.gripper_zero * self.gripper_to_jaw_scale

        self.operator_is_active = False
        if operator_present_topic:
            self.operator_is_present = False
            self.operator_button = crtk.joystick_button(ral, operator_present_topic)
            self.operator_button.set_callback(self.on_operator_present)
        else:
            self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.clutch_button = crtk.joystick_button(ral, clutch_topic)
        self.clutch_button.set_callback(self.on_clutch)

    def get_PSM_current_force(self):
        f = self.puppet.body.measured_cf()[0]
        return f
    def get_MTM_current_force(self):
        f_m = self.master.body.measured_cf()[0]
        return f_m

    def filter(self, alpha, oldData):
        nowOutData = alpha * self.f + (1 - alpha) * oldData
        return nowOutData 

    # callback for operator pedal/button
    def on_operator_present(self, present):
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False

    # callback for clutch pedal/button
    def on_clutch(self, clutch_pressed):
        self.clutch_pressed = clutch_pressed

    # compute relative orientation of mtm and psm
    def alignment_offset(self):
        return self.master.measured_cp()[0].M.Inverse() * self.puppet.setpoint_cp()[0].M

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        self.master_cartesian_initial = self.master.measured_cp()[0]
        self.puppet_cartesian_initial = self.puppet.setpoint_cp()[0]
        self.alignment_offset_initial = self.alignment_offset()
        self.offset_angle, self.offset_axis = self.alignment_offset_initial.GetRotAngle()

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset

        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    def check_arm_state(self):
        if not self.puppet.is_homed():
            print(f'ERROR: {self.ral.node_name()}: puppet ({self.puppet.name}) is not homed anymore')
            self.running = False
        if not self.master.is_homed():
            print(f'ERROR: {self.ral.node_name()}: master ({self.master.name}) is not homed anymore')
            self.running = False

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        self.master.use_gravity_compensation(True)
        self.puppet.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min = math.pi * 100
        self.operator_roll_max = -math.pi * 100
        self.operator_gripper_min = math.pi * 100
        self.operator_gripper_max = -math.pi * 100

    def transition_aligning(self):
        if self.operator_is_active and self.clutch_pressed:
            self.enter_clutched()
            return

        orientation_error, _ = self.alignment_offset().GetRotAngle()
        aligned = orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        orientation_error, _ = self.alignment_offset().GetRotAngle()

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if self.operator_is_present:
            gripper = self.master.gripper.measured_js()[0][0]
            self.operator_gripper_max = max(gripper, self.operator_gripper_max)
            self.operator_gripper_min = min(gripper, self.operator_gripper_min)
            gripper_range = self.operator_gripper_max - self.operator_gripper_min
            if gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            master_rotation, puppet_rotation = self.master.measured_cp()[0].M, self.puppet.setpoint_cp()[0].M
            master_y_axis = PyKDL.Vector(master_rotation[0,1], master_rotation[1,1], master_rotation[2,1])
            puppet_y_axis = PyKDL.Vector(puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1])
            roll = math.acos(PyKDL.dot(puppet_y_axis, master_y_axis))

            self.operator_roll_max = max(roll, self.operator_roll_max)
            self.operator_roll_min = min(roll, self.operator_roll_min)
            roll_range = self.operator_roll_max - self.operator_roll_min
            if roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        aligned = orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            move_cp = PyKDL.Frame(self.puppet.setpoint_cp()[0].M, self.master.setpoint_cp()[0].p)
            self.master.move_cp(move_cp)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.operator_is_present and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not aligned:
                print(f'Unable to align master ({self.master.name}), angle error is {orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master ({self.master.name}) gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.master.body.servo_cf(wrench)
        self.master.lock_orientation(self.master.measured_cp()[0].M)

        self.puppet.hold()

    def transition_clutched(self):
        if not self.clutch_pressed or not self.operator_is_present:
            self.enter_aligning()

    def run_clutched(self):
        pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()

        # set up gripper ghost to rate-limit jaw speed
        jaw_setpoint = self.puppet.jaw.setpoint_js()[0]
        if len(jaw_setpoint) != 1:
            print(f'{self.ral.node_name()}: unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
            self.running = False
        self.gripper_ghost = self.jaw_to_gripper(jaw_setpoint[0])

        self.master.use_gravity_compensation(True)

    def transition_following(self):
        if not self.operator_is_present:
            self.enter_aligning()
        elif self.clutch_pressed:
            self.enter_clutched()

    def run_following(self):
        ''' # let arm move freely
        wrench = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.master.body.servo_cf(wrench)

        ### Cartesian pose teleop
        master_position = self.master.measured_cp()[0]

        # translation
        master_translation = master_position.p - self.master_cartesian_initial.p
        puppet_translation = master_translation * self.scale
        puppet_translation = puppet_translation + self.puppet_cartesian_initial.p

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.offset_angle += math.copysign(min(abs(self.offset_angle), max_delta), -self.offset_angle)
        alignment_offset = PyKDL.Rotation.Rot(self.offset_axis, self.offset_angle)
        puppet_rotation = master_position.M * alignment_offset

        puppet_cartesian_goal = PyKDL.Frame(puppet_rotation, puppet_translation)
        self.puppet.servo_cp(puppet_cartesian_goal)

        ### Jaw/gripper teleop
        gripper_measured_js = self.master.gripper.measured_js()
        current_gripper = gripper_measured_js[0][0]

        ghost_lag = current_gripper - self.gripper_ghost
        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(ghost_lag), max_delta), ghost_lag)
        self.puppet.jaw.servo_jp(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))




        #########################################33
        # Force measurement
        self.f_P = self.get_PSM_current_force()
        self.f_M = self.get_MTM_current_force()

        force = -1*PyKDL.Vector(self.f_P[0], self.f_P[1], self.f_P[2])
        torch = 0*PyKDL.Vector(self.f_P[3], self.f_P[4], self.f_P[5])

        force_M = 0*PyKDL.Vector(self.f_M[0], self.f_M[1], self.f_M[2])
        torch_M = 0*PyKDL.Vector(self.f_M[3], self.f_M[4], self.f_M[5])

        # print('force:', force)
        # print('torch:', torch)

        # Velocity measurement
        puppet_velocity = self.puppet.measured_cv()[0] 
        linear_vel = PyKDL.Vector(puppet_velocity[0], puppet_velocity[1], puppet_velocity[2])
        angular_vel = 0 * PyKDL.Vector(puppet_velocity[3], puppet_velocity[4], puppet_velocity[5])



        master_position_f = self.master.measured_cp()[0]
        puppet_position_f = self.puppet.measured_cp()[0]


        P_M2P = master_position_f.M.Inverse() * (puppet_position_f.p - master_position_f.p)
        R_M2P = master_position_f.M.Inverse() * puppet_position_f.M 

        Transform_M2P = PyKDL.Frame(R_M2P, P_M2P)

        force_P2M = Transform_M2P.M * force
        torch_P2M = Transform_M2P.M * torch

        linear_vel_P2M =  Transform_M2P.M.Inverse() * linear_vel
        angular_vel_P2M =  Transform_M2P.M.Inverse() * angular_vel

        wrench_P2M = [force_P2M[0], force_P2M[1], force_P2M[2], torch_P2M[0], torch_P2M[1], torch_P2M[2]]
        velocity_P2M = [linear_vel_P2M[0], linear_vel_P2M[1], linear_vel_P2M[2], angular_vel_P2M[0], angular_vel_P2M[1], angular_vel_P2M[2]]

        wrench_M = [force_M[0], force_M[1], force_M[2], torch_M[0], torch_M[1], torch_M[2]]

        f_servoCS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        length = len(wrench_M)
        for i in range(length):
            f_servoCS[i] = wrench_M[i] + wrench_P2M[i]

        # position_M = Transform_M2P * puppet_position_f
        if self.count == 0 :
            # print(f"force_p :{force} \n")
            # print(f"Transform_M2P.M : {Transform_M2P.M} \n")
            # print(f"force_M: {force_M} \n")
            # # print(f"torch_M: {torch_M} \n")
            print(f"wrench_M: {wrench_M} ")
            print(f"wrench_P2M: {wrench_P2M} ")
            print(f"f_servoCS: {f_servoCS} ")
            # print(f"velocity_M: {velocity_M} ")
            print('#'*60)
            # print(f"f_servoCS: {f_servoCS}")
            self.count = 400
        self.count -= 1
        #self.master.body.servo_cf(wrench_P2M)
        # self.master.servo_cs(None, None, wrench_M)
        # self.master.servo_cs(None, None, [0.0,0.0,1.0,0.0,0.0,0.0])

        puppet_relative_translation = puppet_position_f.p - self.puppet_cartesian_initial.p
        master_relative_translation = puppet_relative_translation / self.scale
        master_translation = master_relative_translation + self.master_cartesian_initial.p

        master_rotation = puppet_position_f.M * alignment_offset.Inverse()

        master_cartesian_goal = PyKDL.Frame(master_rotation, master_translation)
        self.master.servo_cs(master_cartesian_goal, velocity_P2M, f_servoCS)
        #print(self.f)
        



        # self.count = self.count - 1
        # if self.count == 0:
        #     print('force PSM', self.f)
        #     print('force_MTM:', force_M)
        #     print("")
        #     self.count = 1000
        
        
        # # print(self.f)
        
        # puppet_force = self.puppet.measured_cf()

        
        # angle = Transform_P2M.Get

        # # print('R_P2M:', numpy.array((R_P2M)))
        # # print('P_P2M:', numpy.array((P_P2M)))
        # print(c)        print('')
        
        # adj = self.compute_adjoint(R_P2M, P_P2M)
        # print('adj:', adj)
        # master_force = 2*(adj @ self.f)
        # print(master_force)
        # # self.master.servo_cf(master_force)'''

        #########################################33                  
        # let arm move freely
        #wrench = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #self.master.body.servo_cf(wrench)

	    #Position
        ### Cartesian pose teleop
        master_position = self.master.measured_cp()[0]
        puppet_position_f = self.puppet.measured_cp()[0]
        
        # translation
        master_translation = master_position.p - self.master_cartesian_initial.p
        puppet_translation = master_translation * self.scale
        puppet_translation = puppet_translation + self.puppet_cartesian_initial.p

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.offset_angle += math.copysign(min(abs(self.offset_angle), max_delta), -self.offset_angle)
        alignment_offset = PyKDL.Rotation.Rot(self.offset_axis, self.offset_angle)
        puppet_rotation = master_position.M * alignment_offset

        puppet_cartesian_goal = PyKDL.Frame(puppet_rotation, puppet_translation)
        
        #force
        ### Force measurement
        self.f_P_FW = self.get_PSM_current_force()
        self.f_M_FW = self.get_MTM_current_force()
        
        force_P_FW = -0.5*PyKDL.Vector(self.f_P_FW[0], self.f_P_FW[1], self.f_P_FW[2])
        torch_P_FW = 0*PyKDL.Vector(self.f_P_FW[3], self.f_P_FW[4], self.f_P_FW[5])

        force_M_FW = 0*PyKDL.Vector(self.f_M_FW[0], self.f_M_FW[1], self.f_M_FW[2])
        torch_M_FW = 0*PyKDL.Vector(self.f_M_FW[3], self.f_M_FW[4], self.f_M_FW[5])
        
        #force_P2M = Transform_M2P.M * force
        #torch_P2M = Transform_M2P.M * torch
        force_M2P = force_P_FW
        torch_M2P = torch_P_FW
        
        wrench_M2P = [force_M2P[0], force_M2P[1], force_M2P[2], torch_M2P[0], torch_M2P[1], torch_M2P[2]]
	
        wrench_P = [force_P_FW[0], force_P_FW[1], force_P_FW[2], torch_P_FW[0], torch_P_FW[1], torch_P_FW[2]]

        f_servoCS_FW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        length = len(wrench_P)
        for i in range(length):
            f_servoCS_FW[i] = wrench_P[i] + wrench_M2P[i]

 
        #Velocity
        ### Velocity measurement
        master_velocity = self.master.measured_cv()[0] 
        linear_vel_FW = self.scale * PyKDL.Vector(master_velocity[0], master_velocity[1], master_velocity[2])
        angular_vel_FW = PyKDL.Vector(master_velocity[3], master_velocity[4], master_velocity[5])

        linear_vel_M2P = linear_vel_FW
        angular_vel_M2P = angular_vel_FW
        velocity_M2P = [linear_vel_M2P[0], linear_vel_M2P[1], linear_vel_M2P[2], angular_vel_M2P[0], angular_vel_M2P[1], angular_vel_M2P[2]]

        # P_M2P = master_position_f.M.Inverse() * (puppet_position_f.p - master_position_f.p)
        # R_M2P = master_position_f.M.Inverse() * puppet_position_f.M 
        # Transform_M2P = PyKDL.Frame(R_M2P, P_M2P)
        

	    #Move
        self.puppet.servo_cs(puppet_cartesian_goal, velocity_M2P, f_servoCS_FW)
        # self.puppet.servo_cp(puppet_cartesian_goal)

        ### Jaw/gripper teleop
        gripper_measured_js = self.master.gripper.measured_js()
        current_gripper = gripper_measured_js[0][0]

        ghost_lag = current_gripper - self.gripper_ghost
        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(ghost_lag), max_delta), ghost_lag)
        self.puppet.jaw.servo_jp(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))
        
        
        
        
        
        
        
        # Force measurement
        self.f_P = self.get_PSM_current_force()
        self.f_M = self.get_MTM_current_force()

        force = -1*PyKDL.Vector(self.f_P[0], self.f_P[1], self.f_P[2])
        torch = 0*PyKDL.Vector(self.f_P[3], self.f_P[4], self.f_P[5])

        force_M = 0*PyKDL.Vector(self.f_M[0], self.f_M[1], self.f_M[2])
        torch_M = 0*PyKDL.Vector(self.f_M[3], self.f_M[4], self.f_M[5])

        # print('force:', force)
        # print('torch:', torch)

        # Velocity measurement
        puppet_velocity = self.puppet.measured_cv()[0] 
        linear_vel = (1.0 / self.scale) * PyKDL.Vector(puppet_velocity[0], puppet_velocity[1], puppet_velocity[2])
        angular_vel = PyKDL.Vector(puppet_velocity[3], puppet_velocity[4], puppet_velocity[5])

        master_position_f = self.master.measured_cp()[0]
        puppet_position_f = self.puppet.measured_cp()[0]

        P_M2P = master_position_f.M.Inverse() * (puppet_position_f.p - master_position_f.p)
        R_M2P = master_position_f.M.Inverse() * puppet_position_f.M 

        Transform_M2P = PyKDL.Frame(R_M2P, P_M2P)

        #force_P2M = Transform_M2P.M * force
        #torch_P2M = Transform_M2P.M * torch
        force_P2M = force
        torch_P2M = torch

        wrench_P2M = [force_P2M[0], force_P2M[1], force_P2M[2], torch_P2M[0], torch_P2M[1], torch_P2M[2]]
        # velocity_P2M = [linear_vel_P2M[0], linear_vel_P2M[1], linear_vel_P2M[2], angular_vel_P2M[0], angular_vel_P2M[1], angular_vel_P2M[2]]
        linear_vel_P2M = linear_vel
        angular_vel_P2M = angular_vel
        velocity_P2M = [linear_vel_P2M[0], linear_vel_P2M[1], linear_vel_P2M[2], angular_vel_P2M[0], angular_vel_P2M[1], angular_vel_P2M[2]]
	
        wrench_M = [force_M[0], force_M[1], force_M[2], torch_M[0], torch_M[1], torch_M[2]]

        f_servoCS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        length = len(wrench_M)
        for i in range(length):
            f_servoCS[i] = wrench_M[i] + wrench_P2M[i]

        # position_M = Transform_M2P * puppet_position_f
        if self.count == 0 :
            # print(f"force_p :{force} \n")
            # print(f"Transform_M2P.M : {Transform_M2P.M} \n")
            # print(f"force_M: {force_M} \n")
            # # print(f"torch_M: {torch_M} \n")
            print(f"wrench_M: {wrench_M} ")
            print(f"wrench_P2M: {wrench_P2M} ")
            print(f"f_servoCS: {f_servoCS} ")
            # print(f"velocity_M: {velocity_M} ")
            print('#'*60)
            # print(f"f_servoCS: {f_servoCS}")
            self.count = 400
        self.count -= 1
        #self.master.body.servo_cf(wrench_P2M)
        # self.master.servo_cs(None, None, wrench_M)
        # self.master.servo_cs(None, None, [0.0,0.0,1.0,0.0,0.0,0.0])

        puppet_relative_translation = puppet_position_f.p - self.puppet_cartesian_initial.p
        master_relative_translation = puppet_relative_translation / self.scale
        master_translation = master_relative_translation + self.master_cartesian_initial.p

        master_rotation = puppet_position_f.M * alignment_offset.Inverse()

        master_cartesian_goal = PyKDL.Frame(master_rotation, master_translation)
        self.master.servo_cs(master_cartesian_goal, velocity_P2M, f_servoCS)

    ######################################
    def compute_adjoint(R, P):
        """
        input: Frame: PyKDL.Frame
        return: 6x6 adjoint matrix
        """
        # R = Frame.M.GetData()
        # P = Frame.P.GetData()
        P_hat = numpy.array([[0, -P[2], P[1]],
                            [P[2], 0, -P[0]],
                            [-P[1], P[0], 0]])
        adj = numpy.array([[R, P_hat@R], [numpy.zeros((3,3)), R]])
        return adj

    def home(self):
        print("Homing arms...")
        timeout = 10.0 # seconds
        if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
            return False

        if not self.master.enable(timeout) or not self.master.home(timeout):
            print('    ! failed to home {} within {} seconds'.format(self.master.name, timeout))
            return False

        print("    Homing is complete")
        return True

    def run(self):
        homed_successfully = self.home()
        if not homed_successfully:
            return

        teleop_rate = self.ral.create_rate(int(1/self.run_period))
        print("Running teleop at {} Hz".format(int(1/self.run_period)))

        self.enter_aligning()
        self.running = True

        self.count = 200

        while not self.ral.is_shutdown():
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                self.transition_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                self.transition_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                self.transition_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            self.check_arm_state()
            
           
            if not self.running:
                break

            # run teleop state handler
            if self.current_state == teleoperation.State.ALIGNING:
                self.run_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                self.run_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                self.run_following()
                # self.f = self.get_current_force()
                # print(self.f)
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            teleop_rate.sleep()

class MTM:
            
    class ServoMeasCF:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_servo_cf()
            self.utils.add_measured_cf()

    class Gripper:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_js()

    def __init__(self, ral, arm_name, timeout):
        self.name = arm_name
        self.ral = ral.create_child(arm_name)
        self.utils = crtk.utils(self, self.ral, timeout)

        self.utils.add_operating_state()
        self.utils.add_measured_cp()
        self.utils.add_measured_cv()
        self.utils.add_setpoint_cp()
        self.utils.add_move_cp()
        self.utils.add_servo_cs()

        self.gripper = self.Gripper(self.ral.create_child('gripper'), timeout)
        self.body = self.ServoMeasCF(self.ral.create_child('body'), timeout)

        # non-CRTK topics
        self.lock_orientation_pub = self.ral.publisher('lock_orientation',
                                                        geometry_msgs.msg.Quaternion,
                                                        latch = True, queue_size = 1)
        self.unlock_orientation_pub = self.ral.publisher('unlock_orientation',
                                                         std_msgs.msg.Empty,
                                                         latch = True, queue_size = 1)
        self.use_gravity_compensation_pub = self.ral.publisher('use_gravity_compensation',
                                                                std_msgs.msg.Bool,
                                                                latch = True, queue_size = 1)

    def lock_orientation(self, orientation):
        """orientation should be a PyKDL.Rotation object"""
        q = geometry_msgs.msg.Quaternion()
        q.x, q.y, q.z, q.w = orientation.GetQuaternion()
        self.lock_orientation_pub.publish(q)

    def unlock_orientation(self):
        self.unlock_orientation_pub.publish(std_msgs.msg.Empty())

    def use_gravity_compensation(self, gravity_compensation):
        """Turn on/off gravity compensation (only applies to Cartesian effort mode)"""
        msg = std_msgs.msg.Bool(data=gravity_compensation)
        self.use_gravity_compensation_pub.publish(msg)

class PSM:
    class MeasureCF:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_measured_cf()
    class MeasuredCP:
        def __init__(self,ral,timeout):
            self.utils = crtk.utils(self,ral,timeout)
            self.utils.add_measured_cp()
    class Jaw:
        def __init__(self, ral, timeout):
            self.utils = crtk.utils(self, ral, timeout)
            self.utils.add_setpoint_js()
            self.utils.add_servo_jp()

    def __init__(self, ral, arm_name, timeout):
        self.name = arm_name
        self.ral = ral.create_child(arm_name)
        self.utils = crtk.utils(self, self.ral, timeout)

        self.utils.add_operating_state()
        self.utils.add_setpoint_cp()
        self.utils.add_servo_cp()
        self.utils.add_servo_cs()
        self.utils.add_hold()
        self.utils.add_measured_cv()
        # self.utils.add_measured_cf(self.ral.create_child('spatial'), timeout)
        ###########################
        self.body = self.MeasureCF(self.ral.create_child('body'), timeout)
        self.utils.add_measured_cp()
        self.local = self.MeasuredCP(self.ral.create_child('local'),timeout)

        ###########################

        self.jaw = self.Jaw(self.ral.create_child('jaw'), timeout)

    

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mtm', type = str, required = True,
                        choices = ['MTML', 'MTMR'],
                        help = 'MTM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-p', '--psm', type = str, required = True,
                        choices = ['PSM1', 'PSM2', 'PSM3'],
                        help = 'PSM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    parser.add_argument('-c', '--clutch', type = str, default='/footpedals/clutch',
                        help = 'ROS topic corresponding to clutch button/pedal input')
    parser.add_argument('-o', '--operator', type = str, default='/footpedals/coag', const=None, nargs='?',
                        help = 'ROS topic corresponding to operator present button/pedal/sensor input - use "-o" without an argument to disable')
    parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
                        help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    parser.add_argument('-i', '--interval', type=float, default=0.0025,
                        help = 'time interval/period to run at - should be as long as console\'s period to prevent timeouts')
    args = parser.parse_args(argv)

    ral = crtk.ral('dvrk_python_teleoperation')
    mtm = MTM(ral, args.mtm, timeout=10*args.interval)
    psm = PSM(ral, args.psm, timeout=10*args.interval)
    application = teleoperation(ral, mtm, psm, args.clutch, args.interval,
                                not args.no_mtm_alignment, operator_present_topic=args.operator)
    ral.spin_and_execute(application.run)
