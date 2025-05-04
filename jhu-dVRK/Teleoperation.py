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
from dvrk_console import *
import cisstVectorPython as cisstVector
import pdb

class teleoperation:
    class State(Enum):
        ALIGNING = 1
        CLUTCHED = 2
        FOLLOWING = 3

    def __init__(self, master, puppet, clutch_topic, run_period, align_mtm, operator_present_topic = ""):
        # print('Initialzing dvrk_teleoperation for {} and {}'.format(master.name, puppet.name))
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

        self.operator_is_active = True
        if operator_present_topic:
            self.operator_is_present = False
        #     self.operator_button = crtk.joystick_button(ral, operator_present_topic)
        #     self.operator_button.set_callback(self.on_operator_present)
        # else:
        #     self.operator_is_present = True # if not given, then always assume present

        self.clutch_pressed = False
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
        master_measured_cp = self.master.measured_cp().Position().GetRotation()
        puppet_measured_cp = self.puppet.setpoint_cp().Position().GetRotation()
        return numpy.linalg.inv(master_measured_cp) @ puppet_measured_cp

    # set relative origins for clutching and alignment offset
    def update_initial_state(self):
        
        self.master_cartesian_initial = cisstVector.vctFrm3()
        self.master_cartesian_initial.SetRotation(self.master.measured_cp().Position().GetRotation())
        self.master_cartesian_initial.SetTranslation(self.master.measured_cp().Position().GetTranslation())
      
        self.puppet_cartesian_initial = cisstVector.vctFrm3()
        self.puppet_cartesian_initial.SetRotation(self.puppet.setpoint_cp().Position().GetRotation())
        self.puppet_cartesian_initial.SetTranslation(self.puppet.setpoint_cp().Position().GetTranslation())
        self.alignment_offset_initial = self.alignment_offset()
     
        self.offset_angle, self.offset_axis = self.GetRotAngle(self.alignment_offset_initial)

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
    #     if not self.master.is_homed():
    #         print(f'ERROR: {self.ral.node_name()}: master ({self.master.name}) is not homed anymore')
    #         self.running = False

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
        # without clutch for debug
        if self.operator_is_active and Clutch.GetButton():
            self.enter_clutched()
            return

        alignment_offset = self.alignment_offset()
        orientation_error, _ = self.GetRotAngle(alignment_offset)
        aligned = orientation_error <= self.operator_orientation_tolerance
        if aligned and self.operator_is_active:
            self.enter_following()

    def run_aligning(self):
        orientation_error, _ = self.GetRotAngle(self.alignment_offset())

        # if operator is inactive, use gripper or roll activity to detect when the user is ready
        if Coag.GetButton():
            gripper_init = self.master.gripper.measured_js()
            gripper = gripper_init.Position()
            self.operator_gripper_max = max(gripper, self.operator_gripper_max)
            self.operator_gripper_min = min(gripper, self.operator_gripper_min)
            gripper_range = self.operator_gripper_max - self.operator_gripper_min
            if gripper_range >= self.operator_gripper_threshold:
                self.operator_is_active = True

            # determine amount of roll around z axis by rotation of y-axis
            master_rotation, puppet_rotation = self.master.measured_cp().Position().GetRotation(), self.puppet.setpoint_cp().Position().GetRotation()
            master_y_axis = numpy.array([master_rotation[0,1], master_rotation[1,1], master_rotation[2,1]])
            puppet_y_axis = numpy.array([puppet_rotation[0,1], puppet_rotation[1,1], puppet_rotation[2,1]])
            roll = math.acos(numpy.dot(puppet_y_axis, master_y_axis))

            self.operator_roll_max = max(roll, self.operator_roll_max)
            self.operator_roll_min = min(roll, self.operator_roll_min)
            roll_range = self.operator_roll_max - self.operator_roll_min
            if roll_range >= self.operator_roll_threshold:
                self.operator_is_active = True

        # periodically send move_cp to MTM to align with PSM
        aligned = orientation_error <= self.operator_orientation_tolerance
        now = time.perf_counter()
        if not self.last_align or now - self.last_align > 4.0:
            move_cp = cisstVector.vctFrm3()
            move_cp.SetRotation(self.puppet.setpoint_cp().Position().GetRotation())
            move_cp.SetTranslation(self.master.setpoint_cp().Position().GetTranslation())
            arg = self.master.move_cp.GetArgumentPrototype()
            arg.SetGoal(move_cp)
            self.master.move_cp(arg)
            self.last_align = now

        # periodically notify operator if un-aligned or operator is inactive
        if Coag.GetButton() and now - self.last_operator_prompt > 4.0:
            self.last_operator_prompt = now
            if not aligned:
                print(f'Unable to align master ({self.master.name}), angle error is {orientation_error * 180 / math.pi} (deg)')
            elif not self.operator_is_active:
                print(f'To begin teleop, pinch/twist master ({self.master.name}) gripper a bit')

    def enter_clutched(self):
        self.current_state = teleoperation.State.CLUTCHED

        # let MTM position move freely, but lock orientation
        wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        arg = self.master.body.servo_cf.GetArgumentPrototype()
        arg.SetForce(wrench)
        self.master.body.servo_cf(arg)
        ''' wait for editting'''
        lock_cp = self.master.measured_cp()
        lock_pos = lock_cp.Position()
        lock_rot = lock_pos.GetRotation()
        # self.master.lock_orientation(lock_cp)

        self.puppet.hold()

    def transition_clutched(self):
        if not Clutch.GetButton() or not Coag.GetButton():
            self.enter_aligning()

    def run_clutched(self):
        pass

    def enter_following(self):
        self.current_state = teleoperation.State.FOLLOWING
        # update MTM/PSM origins position
        self.update_initial_state()

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

        self.master.use_gravity_compensation(True)

    def transition_following(self):
        if not Coag.GetButton():
            self.enter_aligning()
        elif Clutch.GetButton():
            self.enter_clutched()

    def run_following(self):
        # let arm move freely
        '''wrench = numpy.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        arg = self.master.body.servo_cf.GetArgumentPrototype()
        arg.SetForce(wrench)
        self.master.body.servo_cf(arg)
        print('master.body.servo_cf(arg)')'''

        ### Cartesian pose teleop
        
        #position
        master_position = self.master.measured_cp().Position()
        puppet_position_fw = self.puppet.measured_cp().Position()

        #rotation
        master_rotation1 = master_position.GetRotation()
        master_trans1 = master_position.GetTranslation()
        master_trans2 = self.master_cartesian_initial.GetTranslation()
        puppet_rotation_fw = puppet_position_fw.GetRotation()
        # translation
        master_translation = master_trans1 - master_trans2
        puppet_translation = master_translation * self.scale

        puppet_trans2 = self.puppet_cartesian_initial.GetTranslation()
        puppet_translation = puppet_translation + puppet_trans2

        # set rotation of psm to match mtm plus alignment offset
        # if we can actuate the MTM, we slowly reduce the alignment offset to zero over time
        max_delta = self.align_rate * self.run_period
        self.offset_angle += math.copysign(min(abs(self.offset_angle), max_delta), -self.offset_angle)
        alignment_offset = self.GetRotMatrix(self.offset_axis, self.offset_angle)
        puppet_rotation = master_rotation1 @ alignment_offset

        puppet_cartesian_goal = cisstVector.vctFrm3()
        puppet_cartesian_goal.SetRotation(puppet_rotation)
        puppet_cartesian_goal.SetTranslation(puppet_translation)

        # Force measurement
        force_MTM = self.master.body.measured_cf().Force()
        # force_MTM_cs = self.master.body.measured_cf().Force()
        force_MTM[0:3] = force_MTM[0:3] * (-0.5)
        force_MTM[3:6] = force_MTM[3:6] * 0 * 2

        # Velocity measurement
        #R_M2P = numpy.linalg.inv(puppet_rotation_fw) @ master_rotation1
        #linear_vel_fw = self.scale * (R_M2P @  self.master.measured_cv().VelocityLinear() )
        linear_vel_fw = self.scale * self.master.measured_cv().VelocityLinear() 
        #angular_vel_fw = 0* (R_M2P @  self.master.measured_cv().VelocityAngular() )
        angular_vel_fw = self.master.measured_cv().VelocityAngular()
        vel_fw = numpy.hstack((linear_vel_fw, angular_vel_fw))

        # execute
        # arg_fw = self.puppet.servo_cs.GetArgumentPrototype()
        # arg_fw.SetPositionIsValid(True)
        # arg_fw.SetPosition(puppet_cartesian_goal)
        # #print(f'master_cartesian_goal: {master_cartesian_goal}')
        # arg_fw.SetVelocityIsValid(True)
        # arg_fw.SetVelocity(vel_fw)
        # #print(f'vel_cs : {vel_cs}')
        # arg_fw.SetForceIsValid(True)
        # arg_fw.SetForce(force_MTM)
        # #print(f'force_PSM_cs : {force_PSM_cs}')

        # self.puppet.servo_cs(arg_fw)

        arg = self.puppet.servo_cp.GetArgumentPrototype()
        arg.SetGoal(puppet_cartesian_goal)
        self.puppet.servo_cp(arg)
        print('self.puppet.servo_cp(arg_cp)')

        ### Jaw/gripper teleop
        gripper_measured_js_init = self.master.gripper.measured_js()
        current_gripper = gripper_measured_js_init.Position()

        ghost_lag = current_gripper - self.gripper_ghost
        max_delta = self.jaw_rate * self.run_period
        # move ghost at most max_delta towards current gripper
        self.gripper_ghost += math.copysign(min(abs(ghost_lag), max_delta), ghost_lag)
        # gripper_to_jaw = self.gripper_to_jaw(self.gripper_ghost)
        arg = self.puppet.jaw.servo_jp.GetArgumentPrototype()
        arg.SetGoal(numpy.array([self.gripper_to_jaw(self.gripper_ghost)]))
        self.puppet.jaw.servo_jp(arg)
        print('self.puppet.servo_jp(arg)')




        
        # MTML_servo_cs Position
        puppet_position_cs = self.puppet.measured_cp().Position()
        master_position_bw = self.master.measured_cp().Position()
        master_rotation_bw = master_position_bw.GetRotation()
        puppet_rotation_cs = puppet_position_cs.GetRotation()

        R_P2M = numpy.linalg.inv(master_rotation_bw) @ puppet_rotation_cs
        puppet_translation_cs = puppet_position_cs.GetTranslation()
        puppet_relative_translation = puppet_translation_cs - self.puppet_cartesian_initial.GetTranslation()
        master_relative_translation = puppet_relative_translation / self.scale
        master_translation_cs = master_relative_translation + self.master_cartesian_initial.GetTranslation()
        master_rotation_cs = puppet_rotation_cs @ numpy.linalg.inv(alignment_offset)
        master_cartesian_goal = cisstVector.vctFrm3()
        master_cartesian_goal.SetRotation(master_rotation_cs)
        master_cartesian_goal.SetTranslation(master_translation_cs)

        # MTML_servo_cs Velocity
        # linear_vel_cs = (1/self.scale) *  (R_P2M @ self.puppet.measured_cv().VelocityLinear() )
        # angular_vel_cs = 0 * ( R_P2M @ self.puppet.measured_cv().VelocityAngular() )
        linear_vel_cs = (1/self.scale) *  self.puppet.measured_cv().VelocityLinear() 
        angular_vel_cs = 0 * ( self.puppet.measured_cv().VelocityAngular() )
        vel_cs = numpy.hstack((linear_vel_cs, angular_vel_cs))

        # MTML_servo_cs Force
        force_PSM_cs = self.puppet.body.measured_cf().Force()
        # force_MTM_cs = self.master.body.measured_cf().Force()
        force_PSM_cs[0:3] = force_PSM_cs[0:3] * (-1)
        force_PSM_cs[3:6] = force_PSM_cs[3:6] * 0 * 2

        # arg = self.master.servo_cs.GetArgumentPrototype()
        # arg.SetPositionIsValid(True)
        # arg.SetPosition(master_cartesian_goal)
        # print(f'master_cartesian_goal: {master_cartesian_goal}')
        # arg.SetVelocityIsValid(True)
        # arg.SetVelocity(vel_cs)
        # print(f'vel_cs : {vel_cs}')
        # arg.SetForceIsValid(True)
        # arg.SetForce(force_PSM_cs)
        # print(f'force_PSM_cs : {force_PSM_cs}')
        # self.master.servo_cs(arg)


        

    # def home(self):
    #     print("Homing arms...")
    #     timeout = 10.0 # seconds
    #     if not self.puppet.enable(timeout) or not self.puppet.home(timeout):
    #         print('    ! failed to home {} within {} seconds'.format(self.puppet.name, timeout))
    #         return False

    #     if not self.master.enable(timeout) or not self.master.home(timeout):
    #         print('    ! failed to home {} within {} seconds'.format(self.master.name, timeout))
    #         return False

    #     print("    Homing is complete")
    #     return True

    def run(self):
        #pdb.set_trace()
        homed_successfully = console.home()
        time.sleep(10)
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
            # check if teleop state should transition
            if self.current_state == teleoperation.State.ALIGNING:
                #print("current state transit aligning")
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
                #print("current state aligning")
                self.run_aligning()
            elif self.current_state == teleoperation.State.CLUTCHED:
                print("current state clutched")
                self.run_clutched()
            elif self.current_state == teleoperation.State.FOLLOWING:
                print("current state following")
                self.run_following()
            else:
                raise RuntimeError("Invalid state: {}".format(self.current_state))

            time.sleep(0.0008)

'''class MTM:
    def __init__(self, arm_name, timeout):
        self.name = arm_name

        # non-CRTK topics
        # self.lock_orientation_pub = self.ral.publisher('lock_orientation',
        #                                                 geometry_msgs.msg.Quaternion,
        #                                                 latch = True, queue_size = 1)
        # self.unlock_orientation_pub = self.ral.publisher('unlock_orientation',
        #                                                  std_msgs.msg.Empty,
        #                                                  latch = True, queue_size = 1)
        # self.use_gravity_compensation_pub = self.ral.publisher('use_gravity_compensation',
        #                                                         std_msgs.msg.Bool,
        #                                                         latch = True, queue_size = 1)

    def lock_orientation(self, orientation):
        """orientation should be a PyKDL.Rotation object"""
        q = geometry_msgs.msg.Quaternion()
        q.x, q.y, q.z, q.w = orientation.GetQuaternion()
        # self.lock_orientation_pub.publish(q)

    def unlock_orientation(self):
        self.unlock_orientation_pub.publish(std_msgs.msg.Empty())

    def use_gravity_compensation(self, gravity_compensation):
        """Turn on/off gravity compensation (only applies to Cartesian effort mode)"""
        msg = std_msgs.msg.Bool(data=gravity_compensation)
        # self.use_gravity_compensation_pub.publish(msg)

class PSM:
    def __init__(self, arm_name, timeout):
        self.name = arm_name'''

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

    # ral = crtk.ral('dvrk_python_teleoperation')
    from dvrk_console import *
    console.power_on()
    #pdb.set_trace()
    mtm = MTML
    psm = PSM2
    application = teleoperation(mtm, psm, 1, 0.002,
                                True, 1)
    application.run()