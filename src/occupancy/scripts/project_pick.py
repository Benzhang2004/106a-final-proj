#!/usr/bin/env python

import sys
import argparse
import numpy as np
import rospkg
import roslaunch
import time


import rospy
import tf2_ros
import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState, PlanningScene, PlanningSceneWorld
from moveit_msgs.srv import ApplyPlanningScene
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from octomap_msgs.msg import Octomap 
import moveit_commander

from geometry_msgs.msg import PoseStamped
from intera_interface import gripper as robot_gripper

octomap = None

def on_receive_octomap(msg):
    global octomap
    octomap = msg

def generate_plannning_scene_msg(empty = False):
    psw = PlanningSceneWorld()
    psw.octomap.header.stamp = rospy.Time.now()
    psw.octomap.header.frame_id = 'base'
    psw.octomap.origin = Pose()
    psw.octomap.origin.position.x = 0.293
    psw.octomap.origin.position.y = 0.447
    psw.octomap.origin.position.z = -0.159
    if empty:
        psw.octomap.octomap = Octomap()
    else:
        psw.octomap.octomap = octomap
    psw.octomap.origin.orientation.z = 1
    ps = PlanningScene()
    ps.is_diff = True
    ps.world = psw
    return ps

def main():
    rospy.init_node('project_pick')
    rospy.wait_for_service('apply_planning_scene')
    rospy.wait_for_service('/clear_octomap')

    group = moveit_commander.MoveGroupCommander('right_arm')
    right_gripper = robot_gripper.Gripper('right_gripper')

    rospy.Subscriber('octomap_full', Octomap, on_receive_octomap, queue_size=1)
    service_proxy = rospy.ServiceProxy('apply_planning_scene', ApplyPlanningScene)
    clear_proxy = rospy.ServiceProxy('/clear_octomap', Empty)

    right_gripper.calibrate()

    right_gripper.open()

    group.set_max_velocity_scaling_factor(0.5)
    group.set_max_acceleration_scaling_factor(0.5)

    TARGET_OBJ_X = 0.5618
    TARGET_OBJ_Y = 0.2868
    TABLE = -0.159
    
    GOAL_X, GOAL_Y = 0.672, -0.253 # near duster

    # GOAL_X, GOAL_Y = 0.516, -0.063

    # GOAL_X, GOAL_Y = 0.672


    # new_pose = PoseStamped()
    # new_pose.pose.position.x = 0.643
    # new_pose.pose.position.y = 0.511
    # new_pose.pose.position.z = 0.101

    # new_pose.pose.orientation.w = 1

    # new_pose.header.stamp = rospy.Time.now()
    # new_pose.header.frame_id = "base"

    print("Waiting for Octomap...")
    while octomap is None:
        time.sleep(0.1)
    print("Got Octomap")
    service_proxy(generate_plannning_scene_msg())

    group.set_pose_target([TARGET_OBJ_X,TARGET_OBJ_Y,0.109, 0, 1, 0, 0])
    plan = group.plan()
    input("READY?")
    group.execute(plan[1])

    # clear_proxy()
    group.set_pose_target([TARGET_OBJ_X,TARGET_OBJ_Y,TABLE, 0, 1, 0, 0])
    plan = group.plan()
    input("READY?")
    group.execute(plan[1])
    input("READY?")
    right_gripper.close()

    # clear_proxy()
    group.set_pose_target([TARGET_OBJ_X,TARGET_OBJ_Y,0.129, 0, 1, 0, 0])
    plan = group.plan()
    input("READY?")
    group.execute(plan[1])

    service_proxy(generate_plannning_scene_msg())
    group.set_pose_target([GOAL_X,GOAL_Y, 0.129, 0, 1, 0, 0])
    plan = group.plan()
    input("READY?")
    group.execute(plan[1])

    clear_proxy()
    group.set_pose_target([GOAL_X,GOAL_Y, TABLE, 0, 1, 0, 0])
    plan = group.plan()
    input("READY?")
    group.execute(plan[1])

    right_gripper.open()



if __name__ == "__main__":
    main()