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
# import crtk
from enum import Enum
import math
# import std_msgs.msg
import sys
import time
#from dvrk_console import *
import cisstVectorPython as cisstVector
import pdb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial.transform.rotation import Rotation

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, mtml1, mtmr1, mtml2, mtmr2, psm1, psm2, clutch, coag, alpha, beta, clutch_topic, run_period, align_mtm, operator_present_topic = ""):
        # print('Initialzing dvrk_teleoperation for {} and {}'.format(master1.name, puppet.name))
        print(f"running at {1/run_period} frequency")
        self.run_period = run_period

        # dominance factor
        self.alpha = alpha
        self.beta = beta

        # MTML - PSM2
        self.master1 = mtml1 # MTML1
        self.master2 = mtml2 # MTML2
        self.puppet = psm2 # PSM2 - MTML 

        # MTMR - PSM1
        self.master3 = mtmr1 # MTMR1
        self.master4 = mtmr2 # MTMR2
        self.puppet2 = psm1

        self.clutch = clutch
        self.coag = coag

        # operating state
        self.master1_op_state = self.master1.operating_state()
        self.master1_is_busy = True

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

        self.operator_is_active = True
        if operator_present_topic:
            self.operator_is_present = False
        #     self.operator_button = crtk.joystick_button(ral, operator_present_topic)
        #     self.operator_button.set_callback(self.on_operator_present)
        # else:
        #     self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
        self.a = 0

        self.time_data = []
        self.y_data_l = []
        self.y_data_l_expected = []
        self.y_data_r = []
        self.y_data_r_expected = []
        self.m1_force = []
        self.m2_force = []
        self.m3_force = []
        self.m4_force = []
        self.puppet_force = []
        self.puppet2_force = []
        # self.clutch_button = crtk.joystick_button(ral, clutch_topic)
        # self.clutch_button.set_callback(self.on_clutch)
        
    def GetRotAngle(self, R):
        
        # Extract the rotation angle (theta)
        theta = numpy.arccos((numpy.trace(R) - 1) / 2)
        
        # Handle edge case when the angle is 0 or 180 degrees
        if numpy.isclose(theta, 0):
            # Identity matrix, no rotation
            axis = numpy.array([1, 0, 0])  # Arbitrary
        elif numpy.isclose(theta, numpy.pi):
            # 180-degree rotation, axis is arbitrary
            axis = numpy.sqrt(numpy.diagonal(R) / 2)
        else:
            # Extract the rotation axis
            axis = numpy.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * numpy.sin(theta))
        
        # Normalize the axis
        axis = axis / numpy.linalg.norm(axis)
        
        return theta, axis


    def GetRotMatrix(self, axis, theta):
        # Ensure the axis is a unit vector
        axis = axis / numpy.linalg.norm(axis)
        
        # Skew-symmetric matrix K from the axis
        K = numpy.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Identity matrix I
        I = numpy.eye(3)
        
        # Rodrigues' rotation formula
        R = I + numpy.sin(theta) * K + (1 - numpy.cos(theta)) * numpy.dot(K, K)
        
        return R

    # average rotation with quaternion
    def average_rotation(self,rotations, alpha):
        # transfrom into quaternion
        quaternions = numpy.array([Rotation.from_dcm(R_i).as_quat() for R_i in rotations])

        # average and norm
        mean_quat = alpha*(quaternions[0,:]) + (1-alpha)*(quaternions[1,:])
        mean_quat /= numpy.linalg.norm(mean_quat)

        # transform into rotation matrix
        return Rotation.from_quat(mean_quat).as_dcm()


    # callback for operator pedal/button
    def on_operator_present(self, present):
        self.operator_is_present = present
        if not present:
            self.operator_is_active = False

    # callback for clutch pedal/button
    def on_clutch(self, clutch_pressed):
        self.clutch_pressed = clutch_pressed

    # compute relative orientation of mtm and psm
    def alignment_offset1(self):
        # master
        master_measured_cp = self.master1.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot
    
    def alignment_offset2(self):
        # master
        master_measured_cp = self.master2.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot

    def alignment_offset3(self):
        # master
        master_measured_cp = self.master3.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet2.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot
    
    def alignment_offset4(self):
        # master
        master_measured_cp = self.master4.measured_cp()
        master_measured_cp_pos = master_measured_cp.Position()
        master_measured_cp_rot = master_measured_cp_pos.GetRotation()
        # puppet
        puppet_measured_cp = self.puppet2.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        return numpy.linalg.inv(master_measured_cp_rot) @ puppet_measured_cp_rot
    
    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        # master1
        self.master_1_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        m1_measured_cp = self.master1.measured_cp()
        m1_measured_cp_pos = m1_measured_cp.Position()
        m1_measured_cp_rot = m1_measured_cp_pos.GetRotation()
        m1_measured_cp_trans = m1_measured_cp_pos.GetTranslation()
        # set
        self.master_1_cartesian_initial.SetRotation(m1_measured_cp_rot)
        self.master_1_cartesian_initial.SetTranslation(m1_measured_cp_trans)
        #print(f"set to {self.master1.measured_cp().Position().GetTranslation()}")
        #print(f"and is {self.master_1_cartesian_initial.GetTranslation()}")
        # master2
        self.master_2_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        m2_measured_cp = self.master2.measured_cp()
        m2_measured_cp_pos = m2_measured_cp.Position()
        m2_measured_cp_rot = m2_measured_cp_pos.GetRotation()
        m2_measured_cp_trans = m2_measured_cp_pos.GetTranslation()
        # set
        self.master_2_cartesian_initial.SetRotation(m2_measured_cp_rot)
        self.master_2_cartesian_initial.SetTranslation(m2_measured_cp_trans)
        # puppet
        self.puppet_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        puppet_measured_cp = self.puppet.setpoint_cp()
        puppet_measured_cp_pos = puppet_measured_cp.Position()
        puppet_measured_cp_rot = puppet_measured_cp_pos.GetRotation()
        puppet_measured_cp_trans = puppet_measured_cp_pos.GetTranslation()
        # set
        self.puppet_cartesian_initial.SetRotation(puppet_measured_cp_rot)
        self.puppet_cartesian_initial.SetTranslation(puppet_measured_cp_trans)
        self.master_1_alignment_offset_initial = self.alignment_offset1()
        self.master_2_alignment_offset_initial = self.alignment_offset2()
     
        self.master_1_offset_angle, self.master_1_offset_axis = self.GetRotAngle(self.master_1_alignment_offset_initial)
        self.master_2_offset_angle, self.master_2_offset_axis = self.GetRotAngle(self.master_2_alignment_offset_initial)

    def update_initial_state2(self):
        # master3
        self.master_3_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        m3_measured_cp = self.master3.measured_cp()
        m3_measured_cp_pos = m3_measured_cp.Position()
        m3_measured_cp_rot = m3_measured_cp_pos.GetRotation()
        m3_measured_cp_trans = m3_measured_cp_pos.GetTranslation()
        # set
        self.master_3_cartesian_initial.SetRotation(m3_measured_cp_rot)
        self.master_3_cartesian_initial.SetTranslation(m3_measured_cp_trans)
        # master4
        self.master_4_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        m4_measured_cp = self.master4.measured_cp()
        m4_measured_cp_pos = m4_measured_cp.Position()
        m4_measured_cp_rot = m4_measured_cp_pos.GetRotation()
        m4_measured_cp_trans = m4_measured_cp_pos.GetTranslation()
        # set
        self.master_4_cartesian_initial.SetRotation(m4_measured_cp_rot)
        self.master_4_cartesian_initial.SetTranslation(m4_measured_cp_trans)
        # puppet
        self.puppet2_cartesian_initial = cisstVector.vctFrm3()
        # measure cp
        puppet2_measured_cp = self.puppet2.setpoint_cp()
        puppet2_measured_cp_pos = puppet2_measured_cp.Position()
        puppet2_measured_cp_rot = puppet2_measured_cp_pos.GetRotation()
        puppet2_measured_cp_trans = puppet2_measured_cp_pos.GetTranslation()
        # set
        self.puppet2_cartesian_initial.SetRotation(puppet2_measured_cp_rot)
        self.puppet2_cartesian_initial.SetTranslation(puppet2_measured_cp_trans)
        self.master_3_alignment_offset_initial = self.alignment_offset3()
        self.master_4_alignment_offset_initial = self.alignment_offset4()
     
        self.master_3_offset_angle, self.master_3_offset_axis = self.GetRotAngle(self.master_3_alignment_offset_initial)
        self.master_4_offset_angle, self.master_4_offset_axis = self.GetRotAngle(self.master_4_alignment_offset_initial)

    def gripper_to_jaw(self, gripper_angle):
        jaw_angle = self.gripper_to_jaw_scale * gripper_angle + self.gripper_to_jaw_offset

        # make sure we don't set goal past joint limits
        return max(jaw_angle, self.jaw_min)

    def jaw_to_gripper(self, jaw_angle):
        return (jaw_angle - self.gripper_to_jaw_offset) / self.gripper_to_jaw_scale

    # def check_arm_state(self):
    #     if not self.puppet.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: puppet ({self.puppet.name}) is not homed anymore')
    #         self.running = False
    #     if not self.master1.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: master1 ({self.master1.name}) is not homed anymore')
    #         self.running = False

    def enter_aligning(self):
        self.current_state = teleoperation.State.ALIGNING
        self.last_align = None
        self.last_operator_prompt = time.perf_counter()

        self.master1.use_gravity_compensation(True)
        self.master2.use_gravity_compensation(True)
        self.master3.use_gravity_compensation(True)
        self.master4.use_gravity_compensation(True)
        self.puppet.hold()
        self.puppet2.hold()

        # reset operator activity data in case operator is inactive
        self.operator_roll_min = math.pi * 100
        self.operator_roll_max = -math.pi * 100
        self.operator_gripper_min = math.pi * 100
        self.operator_gripper_max = -math.pi * 100

    def transition_aligning(self):
        # without clutch for debug
        if self.operator_is_active and self.clutch.GetButton():
            self.enter_clutched()
            return

        master_1_alignment_offset = self.alignment_offset1()
        master_2_alignment_offset = self.alignment_offset2()
        master_3_alignment_offset = self.alignment_offset3()
        master_4_alignment_offset = self.alignment_offset4()
        master_1_orientation_error, _ = self.GetRotAngle(master_1_alignment_offset)
        master_2_orientation_error, _ = self.GetRotAngle(master_2_alignment_offset)
        master_3_orientation_error, _ = self.GetRotAngle(master_3_alignment_offset)
        master_4_orientation_error, _ = self.GetRotAngle(master_4_alignment_offset)
        aligned = master_1_orientation_error <= self.operator_orientation_tolerance and master_2_orientation_error <= self.operator_orientation_tolerance and master_3_orientation_error <= self.operator_orientation_tolerance and master_4_orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        master_1_orientation_error, _ = self.GetRotAngle(self.alignment_offset1())
        master_2_orientation_error, _ = self.GetRotAngle(self.alignment_offset2())
        master_3_orientation_error, _ = self.GetRotAngle(self.alignment_offset3())
        master_4_orientation_error, _ = self.GetRotAngle(self.alignment_offset4())

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        # only detect master1
        if self.coag.GetButton():
            gripper_init = self.master1.gripper.measured_js()
            gripper = gripper_init.Position()
            self.operator_gripper_max = max(gripper, self.operator_gripper_max)
            self.operator_gripper_min = min(gripper, self.operator_gripper_min)
            gripper_range = self.operator_gripper_max - self.operator_gripper_min
            if gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            master_rotation, puppet_rotation = self.master1.measured_cp().Position().GetRotation(), self.puppet.setpoint_cp().Position().GetRotation()
            master_y_axis = numpy.array([master_rotation[0,1], master_rotation[1,1], master_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll = math.acos(numpy.dot(puppet_y_axis, master_y_axis))

            self.operator_roll_max = max(roll, self.operator_roll_max)
            self.operator_roll_min = min(roll, self.operator_roll_min)
            roll_range = self.operator_roll_max - self.operator_roll_min
            if roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        aligned1 = master_1_orientation_error <= self.operator_orientation_tolerance
        aligned2 = master_2_orientation_error <= self.operator_orientation_tolerance
        aligned3 = master_3_orientation_error <= self.operator_orientation_tolerance
        aligned4 = master_4_orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        # move master1 and master2 spontaneously
        if not self.last_align or now - self.last_align > 4.0:
            # master 1
            move_cp_1 = cisstVector.vctFrm3()
            # setpoint cp
            puppet_setpoint_cp = self.puppet.setpoint_cp()
            m1_setpoint_cp = self.master1.setpoint_cp()
            m2_setpoint_cp = self.master2.setpoint_cp()
            # pos
            puppet_setpoint_pos = puppet_setpoint_cp.Position()
            m1_setpoint_pos = m1_setpoint_cp.Position()
            m2_setpoint_pos = m2_setpoint_cp.Position()
            # rot 
            puppet_setpoint_rot = puppet_setpoint_pos.GetRotation()
            # trans
            m1_setpoint_trans = m1_setpoint_pos.GetTranslation()
            m2_setpoint_trans = m2_setpoint_pos.GetTranslation()
            # set
            move_cp_1.SetRotation(puppet_setpoint_rot)
            move_cp_1.SetTranslation(m1_setpoint_trans)
            arg1 = self.master1.move_cp.GetArgumentPrototype()
            arg1.SetGoal(move_cp_1)
            # master 2
            move_cp_2 = cisstVector.vctFrm3()
            move_cp_2.SetRotation(puppet_setpoint_rot)
            move_cp_2.SetTranslation(m2_setpoint_trans)
            arg2 = self.master2.move_cp.GetArgumentPrototype()
            arg2.SetGoal(move_cp_2)
            self.master1.move_cp(arg1)
            self.master2.move_cp(arg2)
            self.last_align = now

            '''
            MTMR1 MTMR2
            '''
            # master 3
            move_cp_3 = cisstVector.vctFrm3()
            # setpoint cp
            puppet2_setpoint_cp = self.puppet2.setpoint_cp()
            m3_setpoint_cp = self.master3.setpoint_cp()
            m4_setpoint_cp = self.master4.setpoint_cp()
            # pos
            puppet2_setpoint_pos = puppet2_setpoint_cp.Position()
            m3_setpoint_pos = m3_setpoint_cp.Position()
            m4_setpoint_pos = m4_setpoint_cp.Position()
            # rot 
            puppet2_setpoint_rot = puppet2_setpoint_pos.GetRotation()
            # trans
            m3_setpoint_trans = m3_setpoint_pos.GetTranslation()
            m4_setpoint_trans = m4_setpoint_pos.GetTranslation()
            # set
            move_cp_3.SetRotation(puppet2_setpoint_rot)
            move_cp_3.SetTranslation(m3_setpoint_trans)
            arg3 = self.master3.move_cp.GetArgumentPrototype()
            arg3.SetGoal(move_cp_3)
            # master 4
            move_cp_4 = cisstVector.vctFrm3()
            move_cp_4.SetRotation(puppet2_setpoint_rot)
            move_cp_4.SetTranslation(m4_setpoint_trans)
            arg4 = self.master4.move_cp.GetArgumentPrototype()
            arg4.SetGoal(move_cp_4)
            self.master3.move_cp(arg3)
            self.master4.move_cp(arg4)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if self.coag.GetButton() and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not aligned1:
                print(f'Unable to align master1, angle error is {master_1_orientation_error * 180 / math.pi} (deg)')
            elif not aligned2:
                print(f'Unable to align master2, angle error is {master_2_orientation_error * 180 / math.pi} (deg)')
            elif not aligned3:
                print(f'Unable to align master3, angle error is {master_3_orientation_error * 180 / math.pi} (deg)')
            elif not aligned4:
                print(f'Unable to align master4, angle error is {master_4_orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master1 gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        arg1 = self.master1.body.servo_cf.GetArgumentPrototype()
        arg1.SetForce(wrench)
        self.master1.body.servo_cf(arg1)
        arg2 = self.master2.body.servo_cf.GetArgumentPrototype()
        arg2.SetForce(wrench)
        self.master2.body.servo_cf(arg2)
        # master1
        m1_lock_cp = self.master1.measured_cp()
        m1_lock_pos = m1_lock_cp.Position()
        m1_lock_rot = m1_lock_pos.GetRotation()
        self.master1.lock_orientation(m1_lock_rot)
        # master2
        m2_lock_cp = self.master2.measured_cp()
        m2_lock_pos = m2_lock_cp.Position()
        m2_lock_rot = m2_lock_pos.GetRotation()
        self.master2.lock_orientation(m2_lock_rot)

        arg3 = self.master3.body.servo_cf.GetArgumentPrototype()
        arg3.SetForce(wrench)
        self.master3.body.servo_cf(arg3)
        arg4 = self.master4.body.servo_cf.GetArgumentPrototype()
        arg4.SetForce(wrench)
        self.master4.body.servo_cf(arg4)
        # master3
        m3_lock_cp = self.master3.measured_cp()
        m3_lock_pos = m3_lock_cp.Position()
        m3_lock_rot = m3_lock_pos.GetRotation()
        self.master3.lock_orientation(m3_lock_rot)
        # master4
        m4_lock_cp = self.master4.measured_cp()
        m4_lock_pos = m4_lock_cp.Position()
        m4_lock_rot = m4_lock_pos.GetRotation()
        self.master4.lock_orientation(m4_lock_rot)

        self.puppet.hold()
        self.puppet2.hold()

    def transition_clutched(self):
        if not self.clutch.GetButton() or not self.coag.GetButton():
            self.enter_aligning()

    def run_clutched(self):
        pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()
        self.update_initial_state2()

        # set up gripper ghost to rate-limit jaw speed
        jaw_setpoint = cisstVector.vctFrm3()
        jaw_setpoint_position = self.puppet.jaw.setpoint_js()
        jaw_setpoint = jaw_setpoint_position.Position()
        # prevent []
        if len(jaw_setpoint) == 0:
            jaw_setpoint = numpy.array([0.])
            print(f'jaw_setpoint :{jaw_setpoint}')

        
        # if len(jaw_setpoint) != 1:
        #     print(f'{self.ral.node_name()}: unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
        #     self.running = False
        self.gripper_ghost = self.jaw_to_gripper(jaw_setpoint[0])# convert 1-D array to scalar

        # jaw 2
        # set up gripper ghost to rate-limit jaw speed
        jaw2_setpoint = cisstVector.vctFrm3()
        jaw2_setpoint_position = self.puppet2.jaw.setpoint_js()
        jaw2_setpoint = jaw2_setpoint_position.Position()
        # prevent []
        if len(jaw2_setpoint) == 0:
            jaw2_setpoint = numpy.array([0.])
            print(f'jaw2_setpoint :{jaw2_setpoint}')

        
        # if len(jaw_setpoint) != 1:
        #     print(f'{self.ral.node_name()}: unable to get jaw position. Make sure there is an instrument on the puppet ({self.puppet.name})')
        #     self.running = False
        self.gripper2_ghost = self.jaw_to_gripper(jaw2_setpoint[0])# convert 1-D array to scalar


        self.master1.use_gravity_compensation(True)
        self.master2.use_gravity_compensation(True)
        self.master3.use_gravity_compensation(True)
        self.master4.use_gravity_compensation(True)


    def transition_following(self):
        if not self.coag.GetButton():
            self.enter_aligning()
        elif self.clutch.GetButton():
            self.enter_clutched()

    def run_following(self):
        # # let arm move freely
        # wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # arg = self.master1.body.servo_cf.GetArgumentPrototype()
        # arg.SetForce(wrench)
        # self.master1.body.servo_cf(arg)
        # print('master1.body.servo_cf(arg)')

        ### Cartesian pose teleop
        
        '''
        MTML1 MTML2 PSM2
        '''
        #position
        alpha1 = self.alpha
        m1_measured_cp = self.master1.measured_cp()
        m2_measured_cp = self.master2.measured_cp()
        master_1_position = m1_measured_cp.Position()
        master_2_position = m2_measured_cp.Position()
        #master_2_position = cisstVector.vctFrm3()
        # puppet_position_fw = self.puppet.measured_cp().Position()

        # rot+trans master1
        master_1_rotation1 = master_1_position.GetRotation()
        master_1_trans1 = master_1_position.GetTranslation()
        master_1_trans2 = self.master_1_cartesian_initial.GetTranslation()
        # rot+trans master2
        master_2_rotation1 = master_2_position.GetRotation()
        print(f"master_2_rotation : {master_2_rotation1}")
        master_2_trans1 = master_2_position.GetTranslation()
        master_2_trans2 = self.master_2_cartesian_initial.GetTranslation()
        # puppet_rotation_fw = puppet_position_fw.GetRotation()
        # translation master1
        master_1_translation = master_1_trans1 - master_1_trans2
        master_1_puppet_translation = master_1_translation * self.scale


        puppet_trans2 = self.puppet_cartesian_initial.GetTranslation()
        master_1_puppet_translation = master_1_puppet_translation + puppet_trans2

        # translation master2
        master_2_translation = master_2_trans1 - master_2_trans2
        master_2_puppet_translation = master_2_translation * self.scale
        
        master_2_puppet_translation += puppet_trans2

        # average translation
        #puppet_translation = (master_1_puppet_translation + master_2_puppet_translation) / 2.0
        puppet_translation = alpha1 * master_1_puppet_translation + (1-alpha1) * master_2_puppet_translation 
        #print(f"target puppet translation: {puppet_translation}")
        #print(f"current puppet translation: {self.puppet.measured_cp().Position().GetTranslation()}")
        #print(f"{master_1_translation}")
        #print(f"{master_2_translation}")

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master_1_offset_angle += math.copysign(min(abs(self.master_1_offset_angle), max_delta), -self.master_1_offset_angle)
        self.master_2_offset_angle += math.copysign(min(abs(self.master_2_offset_angle), max_delta), -self.master_2_offset_angle)
        # rotation offset master1
        master_1_alignment_offset = self.GetRotMatrix(self.master_1_offset_axis, self.master_1_offset_angle)
        master_1_puppet_rotation = master_1_rotation1 @ master_1_alignment_offset
        # rotation offset master2
        master_2_alignment_offset = self.GetRotMatrix(self.master_2_offset_axis,self.master_2_offset_angle)
        master_2_puppet_rotation = master_2_rotation1 @ master_2_alignment_offset

        # average rotation
        #puppet_rotations = Rotation.from_dcm((master_1_puppet_rotation,master_2_puppet_rotation))
        puppet_rotation = self.average_rotation(numpy.array([master_1_puppet_rotation,master_2_puppet_rotation]), alpha1)
        print(f"puppet_rotation : {puppet_rotation}")

        puppet_cartesian_goal = cisstVector.vctFrm3()
        puppet_cartesian_goal.SetRotation(puppet_rotation)
        puppet_cartesian_goal.SetTranslation(puppet_translation)
        print(f'puppet_cartesian_goal : {puppet_cartesian_goal}')

        # Force measurement
        # master 1
        m1_measured_cf = self.master1.body.measured_cf()
        m1_measured_cf_force = m1_measured_cf.Force()

        # force_MTM_cs = self.master1.body.measured_cf().Force()
        m1_measured_cf_force[0:3] = m1_measured_cf_force[0:3] * (-1)
        m1_measured_cf_force[3:6] = m1_measured_cf_force[3:6] * 0 * 2

        # master 2
        m2_measured_cf = self.master2.body.measured_cf()
        m2_measured_cf_force = m2_measured_cf.Force()
        #master_2_force = numpy.zeros(6)
        m2_measured_cf_force[0:3] = m2_measured_cf_force[0:3] * (-1)
        m2_measured_cf_force[3:6] = m2_measured_cf_force[3:6] * 0 * 2

        beta1 = self.beta
        average_master_force = beta1 * m1_measured_cf_force + (1-beta1) * m2_measured_cf_force
        
        # Velocity measurement
        # R_M2P = numpy.linalg.inv(puppet_rotation_fw) @ master_1_rotation1
        # linear_vel_fw = self.scale * (R_M2P @  self.master1.measured_cv().VelocityLinear() )
        linear_vel_fw = self.scale * self.master1.measured_cv().VelocityLinear() 
        # angular_vel_fw = self.scale * (R_M2P @  self.master1.measured_cv().VelocityAngular() )
        angular_vel_fw = self.master1.measured_cv().VelocityAngular()
        vel_fw = numpy.hstack((linear_vel_fw, angular_vel_fw))

        # execute
        arg_fw = self.puppet.servo_cs.GetArgumentPrototype()
        arg_fw.SetPositionIsValid(True)
        arg_fw.SetPosition(puppet_cartesian_goal)
        #print(f'master_cartesian_goal: {master_cartesian_goal}')
        arg_fw.SetVelocityIsValid(True)
        arg_fw.SetVelocity(vel_fw)
        #print(f'vel_cs : {vel_cs}')
        arg_fw.SetForceIsValid(True)
        arg_fw.SetForce(average_master_force)
        #print(f'force_PSM_cs : {force_PSM_cs}')

        self.puppet.servo_cs(arg_fw)
        puppet_measured_cf_plot = self.puppet.body.measured_cf()
        puppet_measured_force_plot = puppet_measured_cf_plot.Force()
        puppet_measured_force_plot_cat = puppet_measured_force_plot[0:3] * 1
        # print(f"puppet_measured_force_plot_cat:{puppet_measured_force_plot_cat}")
        print(f"arg_fw : {arg_fw}")

        # arg = self.puppet.servo_cp.GetArgumentPrototype()
        # arg.SetGoal(puppet_cartesian_goal)
        # self.puppet.servo_cp(arg)
        # print('self.puppet.servo_cp(arg_cp)')

        ### Jaw/gripper teleop
        # master 1
        master_1_gripper_measured_js_init = self.master1.gripper.measured_js()
        master_1_current_gripper = master_1_gripper_measured_js_init.Position()

        master_1_ghost_lag = master_1_current_gripper - self.gripper_ghost
        # master 2
        # master_2_gripper_measured_js_init = self.master2.gripper.measured_js()
        # master_2_current_gripper = master_2_gripper_measured_js_init.Position()
        # master_2_ghost_lag = master_2_current_gripper - self.gripper_ghost
        master_2_ghost_lag = numpy.array([0])

        # average
        average_ghost_lag = (master_1_ghost_lag + master_2_ghost_lag) /2.0

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(average_ghost_lag), max_delta), average_ghost_lag)
        
        # gripper_to_jaw = self.gripper_to_jaw(self.gripper_ghost)
        arg = self.puppet.jaw.servo_jp.GetArgumentPrototype()
        arg.SetGoal(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))
        self.puppet.jaw.servo_jp(arg)
        #print('self.puppet.servo_jp(arg)')

        '''### TEMPORARY!! pls delete
        servo_zero = self.master1.body.servo_cf.GetArgumentPrototype()
        servo_zero.SetForce(numpy.zeros_like(servo_zero.Force()))
        #self.master1.body.servo_cf(servo_zero)
        servo_zero = self.master2.body.servo_cf.GetArgumentPrototype()
        servo_zero.SetForce(numpy.zeros_like(servo_zero.Force()))
        #self.master2.body.servo_cf(servo_zero)'''

        '''
        backward
        '''
        # MTML_servo_cs Position
        puppet_measured_cp = self.puppet.measured_cp()
        puppet_measured_pos = puppet_measured_cp.Position()
        #master_position_bw = self.master1.measured_cp().Position()
        #master_rotation_bw = master_position_bw.GetRotation()
        puppet_measured_rot = puppet_measured_pos.GetRotation()
        puppet_measured_trans = puppet_measured_pos.GetTranslation()

        #R_P2M = numpy.linalg.inv(master_rotation_bw) @ puppet_rotation_cs
        puppet_relative_translation = puppet_measured_trans - self.puppet_cartesian_initial.GetTranslation()
        master_relative_translation = puppet_relative_translation / self.scale
        # relative trans
        m1_translation_cs = master_relative_translation + self.master_1_cartesian_initial.GetTranslation()
        m2_translation_cs = master_relative_translation + self.master_2_cartesian_initial.GetTranslation()
        # relative rot
        m1_rotation_cs = puppet_measured_rot @ numpy.linalg.inv(master_1_alignment_offset)
        m2_rotation_cs = puppet_measured_rot @ numpy.linalg.inv(master_2_alignment_offset)
        # set
        # master1
        m1_cartesian_goal = cisstVector.vctFrm3()
        m1_cartesian_goal.SetRotation(m1_rotation_cs)
        m1_cartesian_goal.SetTranslation(m1_translation_cs)
        # master2
        m2_cartesian_goal = cisstVector.vctFrm3()
        m2_cartesian_goal.SetRotation(m2_rotation_cs)
        m2_cartesian_goal.SetTranslation(m2_translation_cs)

        # MTML_servo_cs Velocity
        #linear_vel_cs = (1/self.scale) *  (R_P2M @ self.puppet.measured_cv().VelocityLinear() )
        #angular_vel_cs = R_P2M @ self.puppet.measured_cv().VelocityAngular() 
        linear_vel_cs = (1/self.scale) *  self.puppet.measured_cv().VelocityLinear() 
        angular_vel_cs = self.puppet.measured_cv().VelocityAngular()
        vel_cs = numpy.hstack((linear_vel_cs, angular_vel_cs))

        # MTML_servo_cs Force
        puppet_measured_cf = self.puppet.body.measured_cf()
        puppet_measured_cf_force = puppet_measured_cf.Force()
        # force_MTM_cs = self.master1.body.measured_cf().Force()
        puppet_measured_cf_force[0:3] = puppet_measured_cf_force[0:3] * (-2.5)
        puppet_measured_cf_force[3:6] = puppet_measured_cf_force[3:6] * 0 * 2

        puppet_measured_cf_force_1 = beta1*puppet_measured_cf_force + (1-beta1)*m2_measured_cf_force
        puppet_measured_cf_force_2 = beta1*puppet_measured_cf_force + (1-beta1)*m1_measured_cf_force

        # master1 arg
        arg = self.master1.servo_cs.GetArgumentPrototype()
        arg.SetPositionIsValid(True)
        arg.SetPosition(m1_cartesian_goal)
        print(f'master_cartesian_goal: {m1_cartesian_goal}')
        arg.SetVelocityIsValid(True)
        arg.SetVelocity(vel_cs)
        print(f'vel_cs : {vel_cs}')
        arg.SetForceIsValid(True)
        arg.SetForce(puppet_measured_cf_force_1)
        #print(f'puppet_measured_cf_force : {puppet_measured_cf_force}')
        self.master1.servo_cs(arg)
        m1_measured_cf_plot = self.master1.body.measured_cf()
        m1_measured_force_plot = m1_measured_cf_plot.Force()
        m1_measured_force_plot = m1_measured_force_plot[0:3] * (-1)
        # master2 arg
        arg2 = self.master2.servo_cs.GetArgumentPrototype()
        arg2.SetPositionIsValid(True)
        arg2.SetPosition(m2_cartesian_goal)
        arg2.SetVelocityIsValid(True)
        arg2.SetVelocity(vel_cs)
        arg2.SetForceIsValid(True)
        arg2.SetForce(puppet_measured_cf_force_2)
        self.master2.servo_cs(arg2)
        # plot for master2
        m2_measured_cf_plot = self.master2.body.measured_cf()
        m2_measured_force_plot = m2_measured_cf_plot.Force()
        m2_measured_force_plot = m2_measured_force_plot[0:3] * (-1)

        '''
        MTMR1 MTMR2 PSM1
        '''
        #position
        m3_measured_cp = self.master3.measured_cp()
        m4_measured_cp = self.master4.measured_cp()
        master_3_position = m3_measured_cp.Position()
        master_4_position = m4_measured_cp.Position()

        # rot+trans master3
        master_3_rotation1 = master_3_position.GetRotation()
        master_3_trans1 = master_3_position.GetTranslation()
        master_3_trans2 = self.master_3_cartesian_initial.GetTranslation()
        # rot+trans master4
        master_4_rotation1 = master_4_position.GetRotation()
        print(f"master_4_rotation : {master_4_rotation1}")
        master_4_trans1 = master_4_position.GetTranslation()
        master_4_trans2 = self.master_4_cartesian_initial.GetTranslation()
        # translation master3
        master_3_translation = master_3_trans1 - master_3_trans2
        master_3_puppet_translation = master_3_translation * self.scale


        puppet2_trans2 = self.puppet2_cartesian_initial.GetTranslation()
        master_3_puppet_translation = master_3_puppet_translation + puppet2_trans2

        # translation master4
        master_4_translation = master_4_trans1 - master_4_trans2
        master_4_puppet_translation = master_4_translation * self.scale
        
        master_4_puppet_translation += puppet2_trans2

        # average translation
        puppet2_translation = alpha1 * master_3_puppet_translation +  (1-alpha1) * master_4_puppet_translation
        #print(f"target puppet translation: {puppet_translation}")
        #print(f"current puppet translation: {self.puppet.measured_cp().Position().GetTranslation()}")
        #print(f"{master_1_translation}")
        #print(f"{master_2_translation}")

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.master_3_offset_angle += math.copysign(min(abs(self.master_3_offset_angle), max_delta), -self.master_3_offset_angle)
        self.master_4_offset_angle += math.copysign(min(abs(self.master_4_offset_angle), max_delta), -self.master_4_offset_angle)

        # rotation offset master3
        master_3_alignment_offset = self.GetRotMatrix(self.master_3_offset_axis, self.master_3_offset_angle)
        master_3_puppet_rotation = master_3_rotation1 @ master_3_alignment_offset
        # rotation offset master4
        master_4_alignment_offset = self.GetRotMatrix(self.master_4_offset_axis,self.master_4_offset_angle)
        master_4_puppet_rotation = master_4_rotation1 @ master_4_alignment_offset

        # average rotation
        #puppet_rotations = Rotation.from_dcm((master_1_puppet_rotation,master_2_puppet_rotation))
        puppet2_rotation = self.average_rotation(numpy.array([master_3_puppet_rotation,master_4_puppet_rotation]), alpha1)
        print(f"puppet2_rotation : {puppet2_rotation}")

        puppet2_cartesian_goal = cisstVector.vctFrm3()
        puppet2_cartesian_goal.SetRotation(puppet2_rotation)
        puppet2_cartesian_goal.SetTranslation(puppet2_translation)
        print(f'puppet2_cartesian_goal : {puppet2_cartesian_goal}')

        # Force measurement
        # master 3
        m3_measured_cf = self.master3.body.measured_cf()
        m3_measured_cf_force = m3_measured_cf.Force()

        # force_MTM_cs = self.master1.body.measured_cf().Force()
        m3_measured_cf_force[0:3] = m3_measured_cf_force[0:3] * (-1)
        m3_measured_cf_force[3:6] = m3_measured_cf_force[3:6] * 0 * 2

        # master 4
        m4_measured_cf = self.master4.body.measured_cf()
        m4_measured_cf_force = m4_measured_cf.Force()
        m4_measured_cf_force[0:3] = m4_measured_cf_force[0:3] * (-1)
        m4_measured_cf_force[3:6] = m4_measured_cf_force[3:6] * 0 * 2

        average_master_force = beta1 * m3_measured_cf_force + (1-beta1) * m4_measured_cf_force
        
        # Velocity measurement
        # R_M2P = numpy.linalg.inv(puppet_rotation_fw) @ master_1_rotation1
        # linear_vel_fw = self.scale * (R_M2P @  self.master1.measured_cv().VelocityLinear() )
        linear_vel_fw = self.scale * self.master3.measured_cv().VelocityLinear() 

        # angular_vel_fw = self.scale * (R_M2P @  self.master1.measured_cv().VelocityAngular() )
        angular_vel_fw = self.master3.measured_cv().VelocityAngular()
        vel_fw = numpy.hstack((linear_vel_fw, angular_vel_fw))

        # execute
        arg_fw = self.puppet2.servo_cs.GetArgumentPrototype()
        arg_fw.SetPositionIsValid(True)
        arg_fw.SetPosition(puppet2_cartesian_goal)
        #print(f'master_cartesian_goal: {master_cartesian_goal}')
        arg_fw.SetVelocityIsValid(True)
        arg_fw.SetVelocity(vel_fw)
        #print(f'vel_cs : {vel_cs}')
        arg_fw.SetForceIsValid(True)
        arg_fw.SetForce(average_master_force)
        #print(f'force_PSM_cs : {force_PSM_cs}')

        self.puppet2.servo_cs(arg_fw)
        puppet2_measured_cf_plot = self.puppet2.body.measured_cf()
        puppet2_measured_force_plot = puppet2_measured_cf_plot.Force()
        puppet2_measured_force_plot_cat = puppet2_measured_force_plot[0:3] * 1
        print(f"arg_fw : {arg_fw}")

        ### Jaw/gripper teleop
        # master 3
        master_3_gripper_measured_js_init = self.master3.gripper.measured_js()
        master_3_current_gripper = master_3_gripper_measured_js_init.Position()

        master_3_ghost_lag = master_3_current_gripper - self.gripper2_ghost
        # master 2
        # master_2_gripper_measured_js_init = self.master2.gripper.measured_js()
        # master_2_current_gripper = master_2_gripper_measured_js_init.Position()
        # master_2_ghost_lag = master_2_current_gripper - self.gripper_ghost
        master_4_ghost_lag = numpy.array([0])

        # average
        average_ghost_lag = (master_3_ghost_lag + master_4_ghost_lag) /2.0

        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper2_ghost += math.copysign(min(abs(average_ghost_lag), max_delta), average_ghost_lag)
        
        # gripper_to_jaw = self.gripper_to_jaw(self.gripper_ghost)
        arg = self.puppet2.jaw.servo_jp.GetArgumentPrototype()
        arg.SetGoal(numpy.array([self.gripper_to_jaw(self.gripper2_ghost)]))
        self.puppet2.jaw.servo_jp(arg)

        '''
        backward
        '''
        # MTMR_servo_cs Position
        puppet2_measured_cp = self.puppet2.measured_cp()
        puppet2_measured_pos = puppet2_measured_cp.Position()
        puppet2_measured_rot = puppet2_measured_pos.GetRotation()
        puppet2_measured_trans = puppet2_measured_pos.GetTranslation()

        #R_P2M = numpy.linalg.inv(master_rotation_bw) @ puppet_rotation_cs
        puppet2_relative_translation = puppet2_measured_trans - self.puppet2_cartesian_initial.GetTranslation()
        master_relative_translation = puppet2_relative_translation / self.scale
        # relative trans
        m3_translation_cs = master_relative_translation + self.master_3_cartesian_initial.GetTranslation()
        m4_translation_cs = master_relative_translation + self.master_4_cartesian_initial.GetTranslation()
        # relative rot
        m3_rotation_cs = puppet2_measured_rot @ numpy.linalg.inv(master_3_alignment_offset)
        m4_rotation_cs = puppet2_measured_rot @ numpy.linalg.inv(master_4_alignment_offset)
        # set
        # master3
        m3_cartesian_goal = cisstVector.vctFrm3()
        m3_cartesian_goal.SetRotation(m3_rotation_cs)
        m3_cartesian_goal.SetTranslation(m3_translation_cs)
        # master4
        m4_cartesian_goal = cisstVector.vctFrm3()
        m4_cartesian_goal.SetRotation(m4_rotation_cs)
        m4_cartesian_goal.SetTranslation(m4_translation_cs)

        # MTMR_servo_cs Velocity
        #linear_vel_cs = (1/self.scale) *  (R_P2M @ self.puppet.measured_cv().VelocityLinear() )
        #angular_vel_cs = R_P2M @ self.puppet.measured_cv().VelocityAngular() 
        linear_vel_cs = (1/self.scale) *  self.puppet2.measured_cv().VelocityLinear() 
        angular_vel_cs = self.puppet2.measured_cv().VelocityAngular()
        vel_cs = numpy.hstack((linear_vel_cs, angular_vel_cs))

        # MTML_servo_cs Force
        puppet2_measured_cf = self.puppet2.body.measured_cf()
        puppet2_measured_cf_force = puppet2_measured_cf.Force()
        # force_MTM_cs = self.master1.body.measured_cf().Force()
        puppet2_measured_cf_force[0:3] = puppet2_measured_cf_force[0:3] * (-2.5)
        puppet2_measured_cf_force[3:6] = puppet2_measured_cf_force[3:6] * 0 * 2

        puppet2_measured_cf_force_1 = beta1*puppet2_measured_cf_force + (1-beta1)*m4_measured_cf_force
        puppet2_measured_cf_force_2 = beta1*puppet2_measured_cf_force + (1-beta1)*m3_measured_cf_force

        # master3 arg
        arg = self.master3.servo_cs.GetArgumentPrototype()
        arg.SetPositionIsValid(True)
        arg.SetPosition(m3_cartesian_goal)
        print(f'master3_cartesian_goal: {m3_cartesian_goal}')
        arg.SetVelocityIsValid(True)
        arg.SetVelocity(vel_cs)
        print(f'vel_cs : {vel_cs}')
        arg.SetForceIsValid(True)
        arg.SetForce(puppet2_measured_cf_force_1)
        #print(f'puppet_measured_cf_force : {puppet_measured_cf_force}')
        self.master3.servo_cs(arg)
        m3_measured_cf_plot = self.master3.body.measured_cf()
        m3_measured_force_plot = m3_measured_cf_plot.Force()
        m3_measured_force_plot = m3_measured_force_plot[0:3] * (-1)
        # master4 arg
        arg2 = self.master4.servo_cs.GetArgumentPrototype()
        arg2.SetPositionIsValid(True)
        arg2.SetPosition(m4_cartesian_goal)
        arg2.SetVelocityIsValid(True)
        arg2.SetVelocity(vel_cs)
        arg2.SetForceIsValid(True)
        arg2.SetForce(puppet2_measured_cf_force_2)
        self.master4.servo_cs(arg2)


        '''
        plot
        '''
        puppet_measured_cp_plot = self.puppet.measured_cp()
        puppet_measured_pos_plot = puppet_measured_cp_plot.Position()
        puppet_measured_trans_plot = puppet_measured_pos_plot.GetTranslation()
        print(f"puppet_measured_trans_plot: {puppet_measured_trans_plot}")
        puppet2_measured_cp_plot = self.puppet2.measured_cp()
        puppet2_measured_pos_plot = puppet2_measured_cp_plot.Position()
        m4_measured_cf_plot = self.master4.body.measured_cf()
        m4_measured_force_plot = m4_measured_cf_plot.Force()
        m4_measured_force_plot = m4_measured_force_plot[0:3] * (-1)


        #master_position_bw = self.master1.measured_cp().Position()
        #master_rotation_bw = master_position_bw.GetRotation()
        puppet2_measured_trans_plot = puppet2_measured_pos_plot.GetTranslation()
        print(f"puppet2_measured_trans_plot: {puppet2_measured_trans_plot}")
        self.y_data_l.append(puppet_measured_trans_plot)
        self.y_data_l_expected.append(puppet_translation)
        self.y_data_r.append(puppet2_measured_trans_plot)
        self.y_data_r_expected.append(puppet2_translation)
        self.m1_force.append(m1_measured_force_plot)
        self.m2_force.append(m2_measured_force_plot)
        self.m3_force.append(m3_measured_force_plot)
        self.m4_force.append(m4_measured_force_plot)
        self.puppet_force.append(puppet_measured_force_plot_cat)
        self.puppet2_force.append(puppet2_measured_force_plot_cat)
        # print(self.puppet_force)
        self.a += 1


    # def home(self):
    #     print("Homing arms...")
    #     timeout = 10.0 # seconds
    #     if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
    #         print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
    #         return False

    #     if not self.master1.enable(timeout) or not self.master1.home(timeout):
    #         print('    ! failed to home {} within {} seconds'.format(self.master1.name, timeout))
    #         return False

    #     print("    Homing is complete")
    #     return True

    def run(self):
        #pdb.set_trace()
        homed_successfully1 = console.home()
        homed_successfully = console_davinci.home()
        time.sleep(15)
        # plotting
        self.fig, self.ax = plt.subplots()
        line, = self.ax.plot([], [], lw=2)
        print("home complete")
        if not homed_successfully:
            print("home not success")
            return

        
        #teleop_rate = self.ral.create_rate(int(1/self.run_period))
        # print("Running teleop at {} Hz".format(int(1/self.run_period)))
        freq = int(1/self.run_period)

        self.enter_aligning()
        print("aligned complete")
        self.running = True

        #while not self.ral.is_shutdown():
        while True:
        #while self.a <= 6000:
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                print("current state transit aligning")
                self.transition_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                print("current state transit clutched")
                self.transition_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                print("current state transit following")
                self.transition_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            # self.check_arm_state()
            if not self.running:
                break

            # run teleop state handler
            if self.current_state == teleoperation.State.ALIGNING:
                print("current state aligning")
                self.run_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                print("current state clutched")
                self.run_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                print("current state following")
                self.run_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))
            
            self.master1_op_state = self.master1.operating_state()
            self.master1_is_busy = self.master1_op_state.GetIsBusy()

            print(f"master1_is_busy : {self.master1_is_busy}")

            time.sleep(self.run_period)

        numpy.savetxt('pos_l.txt', self.y_data_l, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('pos_r.txt', self.y_data_r, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('pos_l_exp.txt', self.y_data_l_expected, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('pos_r_exp.txt', self.y_data_r_expected, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('m1_force.txt', self.m1_force, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('m2_force.txt', self.m2_force, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('m3_force.txt', self.m3_force, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('m4_force.txt', self.m4_force, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('puppet_force.txt', self.puppet_force, fmt='%f', delimiter=' ', header='x y z', comments='')
        numpy.savetxt('puppet2_force.txt', self.puppet2_force, fmt='%f', delimiter=' ', header='x y z', comments='')

        print(f"run terminated, MTML is busy: {self.master1_is_busy}")

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    # argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    # parser = argparse.ArgumentParser(description = __doc__,
    #                                  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-m', '--mtm', type = str, required = True,
    #                     choices = ['MTML', 'MTMR'],
    #                     help = 'MTM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    # parser.add_argument('-p', '--psm', type = str, required = True,
    #                     choices = ['PSM1', 'PSM2', 'PSM3'],
    #                     help = 'PSM arm name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace')
    # parser.add_argument('-c', '--clutch', type = str, default='/footpedals/clutch',
    #                     help = 'ROS topic corresponding to clutch button/pedal input')
    # parser.add_argument('-o', '--operator', type = str, default='/footpedals/coag', const=None, nargs='?',
    #                     help = 'ROS topic corresponding to operator present button/pedal/sensor input - use "-o" without an argument to disable')
    # parser.add_argument('-n', '--no-mtm-alignment', action='store_true',
    #                     help="don't align mtm (useful for using haptic devices as MTM which don't have wrist actuation)")
    # parser.add_argument('-i', '--interval', type=float, default=0.005,
    #                     help = 'time interval/period to run at - should be as long as console\'s period to prevent timeouts')
    # args = parser.parse_args(argv)
    parser = argparse.ArgumentParser(description = __doc__,
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alpha', type = float, required = False, default= 0.5,
                         help = 'dominance factor alpha, between 0 and 1')
    parser.add_argument('-b', '--beta', type = float, required = False, default= 0.5,
                         help = 'dominance factor beta, between 0 and 1')
    args = parser.parse_args()

    # ral = crtk.ral('dvrk_python_teleoperation')
    from dvrk_console_multi import *
    # console.power_on()
    #pdb.set_trace()
    mtm1 = MTML
    mtm2 = MTML2
    mtm3 = MTMR
    mtm4 = MTMR2
    psm1 = PSM1
    psm2 = PSM2
    clutch = Clutch
    coag = Coag

    application = teleoperation(mtm1,mtm3,mtm2,mtm4, psm1, psm2, clutch, coag, args.alpha, args.beta, 1, 0.001,
                                True, 1)

    application.run()