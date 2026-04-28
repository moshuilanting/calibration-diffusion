import sys
import os

# 获取当前脚本所在的目录
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(current_path)
# 将当前路径添加到 sys.path 中
sys.path.append(current_path)

from rotations import rot_matrix_to_quat, euler_to_rot_matrix,quat_to_rot_matrix,euler_angles_to_quat,quat_to_euler_angles,matrix_to_euler_angles
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import time

from model.diffusion.configuration_diffusion import DiffusionConfig
from model.diffusion.modeling_diffusion import DiffusionPolicy
 
from diffusers.training_utils import EMAModel
import torch
from torchvision import transforms
import math
import random
import numpy as np
import copy
random.seed(10)
torch.manual_seed(10)
np.random.seed(10)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core import World
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.kit.actions.core
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim

import omni.isaac.core.utils.prims as prims_utils

seq_data = [[], [], []]

transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量，像素值范围是[0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def T2DoF(frame_positions, frame_rotation):
    xyz = frame_positions
    euler = matrix_to_euler_angles(frame_rotation)
    
    return np.hstack((euler,xyz))


def Jacobian(Solver,current_joint):
    """
    current_joint: numpy or list,  当前关节角度
    arm_property, str, 求解那一组关节运动学
    
    """
    h = 1e-8
    n = len(current_joint)

    J = np.zeros((6, n))  

    for i in range(n):
        theta_p = copy.deepcopy(current_joint)
        theta_m = copy.deepcopy(current_joint)
        theta_p[i] = theta_p[i] + h
        theta_m[i] = theta_m[i] - h
        p_frame_positions , p_frame_rotation = Solver.get_kinematics_solver().compute_forward_kinematics("right_gripper", theta_p)
        m_frame_positions , m_frame_rotation= Solver.get_kinematics_solver().compute_forward_kinematics("right_gripper",theta_m)
        J[:, i] = (T2DoF(p_frame_positions , p_frame_rotation) - T2DoF(m_frame_positions , m_frame_rotation)) / (2*h)
    return J


def MoveL(Solver,current_joints, target_position, target_orientation, mid_inter_num, Use_Jacbian=True):
    
    current_positions , current_rotation = Solver.get_kinematics_solver().compute_forward_kinematics("right_gripper",current_joints)

    current_pos_euler =  T2DoF(current_positions,current_rotation)

    target_pos_euler = np.hstack((quat_to_euler_angles(target_orientation),target_position))

    '''
    针对insert任务单独处理
    '''
    for k in range(3):
        if current_pos_euler[k] > 0 and target_pos_euler[k]  < 0:
            current_pos_euler[k]  = -current_pos_euler[k]
        elif current_pos_euler[k] < 0 and target_pos_euler[k] > 0:
            current_pos_euler[k] = abs(current_pos_euler[k])

        
        # if abs(abs(target_pos_euler[k] - current_pos_euler[k]) - 6.28) < epsilon:
        #     target_pos_euler[k] = current_pos_euler[k]

    #print("current_pos_euler: ",current_pos_euler)
    #print("target_pos_euler: ",target_pos_euler)

    inter_pos_list = np.linspace(current_pos_euler, target_pos_euler, mid_inter_num)
    #input()
    if Use_Jacbian == False:
        return inter_pos_list
    
    trajectory = []
    for j in range(len(inter_pos_list)):
        jacbian = Jacobian(Solver, current_joint=current_joints)
        Jacbian_pinv = np.linalg.pinv(jacbian)
        delta_vector = inter_pos_list[j] - current_pos_euler
        delta_joint = np.dot(Jacbian_pinv, delta_vector)
        next_joints = list(np.sum([delta_joint, current_joints], axis=0))

        current_joints = next_joints
        current_positions , current_rotation = Solver.get_kinematics_solver().compute_forward_kinematics("right_gripper",current_joints)
        current_pos_euler =  T2DoF(current_positions,current_rotation)

        trajectory.append(next_joints)

    return trajectory



def diffusion_action_generation(count, image, depth, status, Base2Cam_RT, policy, seq_data, seq_length= 5):
    '''
    count: 当前第几步
    image: 
    depth: 
    status: 
    seq_length: 序列长度

    return: 
    末端位置、四元数、开合
    '''

    seq_image, seq_depth, seq_status = seq_data[0],seq_data[1],seq_data[2]

    #print(status)
    #print('seq_image: ', len(seq_image), count)

    current_image = transform(image)
    current_depth = np.nan_to_num(depth).astype(np.float32)
    if np.any(np.array(status['EE']) <0.035):
        EE = np.array([0,0])
    else:
        EE = np.array([0.04,0.04])


    #Cam_Vec, Cam_Quat = Coord_Trans(status['Translation'], status['Rotation'], Base2Cam_RT)
    current_EE = np.hstack((np.array(status['Translation']), np.array(status['Rotation']), EE))
    #current_Cam_EE = np.hstack((Cam_Vec, Cam_Quat, EE))


    if count == 0:
        seq_image = [current_image] * seq_length
        seq_depth = [current_depth] *seq_length
        seq_status = [current_EE] * seq_length
    else:
        seq_image = seq_image[1:]
        seq_image.append(current_image)
        seq_depth = seq_depth[1:]
        seq_depth.append(current_depth)
        seq_status = seq_status[1:]
        seq_status.append(current_EE)

    #print('seq_image: ', len(seq_image), count)
    np_seq_image = np.stack(seq_image)
    np_seq_depth = np.stack(seq_depth)
    np_seq_depth = np.expand_dims(np_seq_depth, axis = 1)
    np_seq_status = np.stack(seq_status)
    np_seq_combined_image = np.concatenate((np_seq_image, np_seq_depth), axis = 1)

    np_seq_combined_image = np.expand_dims(np_seq_combined_image, axis = 0)
    np_seq_status = np.expand_dims(seq_status, axis = 0)

    torch_seq_status = torch.from_numpy(np_seq_status).float().cuda()
    torch_np_seq_combined_image = torch.from_numpy(np_seq_combined_image).float().cuda()

    #print(torch_np_seq_combined_image.shape)

    # Trans_Center_Translation = Base2Cam_RT[:3, 3]
    # Trans_Center_Orientation = rot_matrix_to_quat(Base2Cam_RT[:3, :3])
    # Trans_Center = np.hstack((Trans_Center_Translation, Trans_Center_Orientation, [0,0]))
    # Seq_Trans_Center = np.tile(Trans_Center, (cfg.set_seq_length+cfg.action_seq_length, 1))
    # Seq_Trans_Center = np.expand_dims(Seq_Trans_Center, axis = 0)
    # Seq_Trans_Center = torch.from_numpy(Seq_Trans_Center).float().cuda()

    pred_action = policy.select_action(obs=torch_np_seq_combined_image, state= torch_seq_status).cpu().detach().numpy()[0][0]

    #print(pred_action.shape)

    # quat = euler_angles_to_quat(pred_action[3:6])
    # translation = np.array(pred_action[:3])
    # EE_vector = np.array(pred_action[6:])

    quat = pred_action[3:7]
    translation = np.array(pred_action[:3])
    EE_vector = np.array(pred_action[7:])

    translation[np.isnan(translation)] = 0
    quat[np.isnan(quat)] = 0

    if np.any(np.array(pred_action[7:]) <0.035):
        EE_vector = np.array([0,0])
    else:
        EE_vector = np.array([0.04,0.04])

    jump = 5


    return translation, quat, EE_vector , [seq_image, seq_depth, seq_status], jump




def rotate_square_center(center, side_length, rotation_center, angle_radians):

    x_c, y_c, z_c = center
    x_o, y_o, z_o = rotation_center

    # 平移
    x_translated = x_c - x_o
    y_translated = y_c - y_o

    # 旋转
    x_rotated = x_translated * math.cos(angle_radians) - y_translated * math.sin(angle_radians)
    y_rotated = x_translated * math.sin(angle_radians) + y_translated * math.cos(angle_radians)

    # 反向平移
    new_center_x = x_rotated + x_o
    new_center_y = y_rotated + y_o

    return [new_center_x, new_center_y, z_c], side_length


device = torch.device("cuda")
cfg = DiffusionConfig()

policy = DiffusionPolicy(cfg)
policy.eval()
policy.to(device)

model_name = "Eplison_DDPM_0505_Franka_Insert_Stick_diffusion_sample_300_1200.pth"
print(model_name)
#policy.load_state_dict(torch.load('hunder_diffusion_pull_1000.pth', map_location=device,weights_only=True))
policy.load_state_dict(torch.load(model_name, map_location=device,weights_only=True))

if __name__=="__main__":
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane(static_friction=2,dynamic_friction=1,restitution = 0)

    action_registry = omni.kit.actions.core.get_action_registry()
    light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    #light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    light_action.execute()

    franka = my_world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    my_IK = KinematicsSolver(franka, end_effector_frame_name="right_gripper")

    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
        events_dt = [0.008, 0.005, 1, 0.1, 0.05, 0.02, 0.008, 1, 0.008, 0.08]
    )
    '''
    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.
    
    '''

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube",
            name="fixed_cube",
            position=np.array([0.7 , 0.0, 0.1]),
            scale=np.array([1.2, 1.6, 0.02]),
            color=np.array([0.5, 0.3, 0.0])
        )
    )

    while simulation_app.is_running():

        success_count = 0
        for t in range(50):

            for cube_index in range(4):
                my_world.scene.remove_object(name="red_cube_{:d}".format(cube_index))
            my_world.scene.remove_object(name="blue_cube")

            my_world.reset()


            center_x, center_y,center_z = random.uniform(0.55,  0.75), random.uniform(-0.15,  0), 0.1
            #第一个正方形
            forward_squar_center = [center_x+0.18 , center_y, center_z+0.1]
            forward_square_size = [0.04, 0.03, 0.02]
            back_squar_center = [center_x-0.18 , center_y, center_z+0.1]
            back_square_size = [0.04, 0.03, 0.02]
            left_squar_center = [center_x , center_y+0.1, center_z+0.1]
            right_squar_center = [center_x , center_y-0.1, center_z+0.1]
            left_square_size = [0.4, 0.16, 0.02] 
            right_square_size = [0.4, 0.16, 0.02] 

            angle_radians = random.uniform(-1,  1)
            #print("angle_radians: ",angle_radians)
            forward_squar_center, forward_square_size = rotate_square_center(forward_squar_center, forward_square_size, [center_x, center_y,center_z], angle_radians)
            back_squar_center, back_square_size = rotate_square_center(back_squar_center, back_square_size, [center_x, center_y,center_z], angle_radians)
            left_squar_center, left_square_size = rotate_square_center(left_squar_center, left_square_size, [center_x, center_y,center_z], angle_radians)
            right_squar_center, right_square_size = rotate_square_center(right_squar_center, right_square_size, [center_x, center_y,center_z], angle_radians)


            my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/red_cube_0",
                name="red_cube_0",
                position=np.array(forward_squar_center),
                orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, angle_radians]), degrees=False),
                scale=np.array(forward_square_size),
                color=np.array([0.5, 0.0, 0.2]),
            ))

            my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/red_cube_1",
                name="red_cube_1",
                position=np.array(back_squar_center),
                orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, angle_radians]), degrees=False),
                scale=np.array(back_square_size),
                color=np.array([0.5, 0.0, 0.2]),
            ))

            my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/red_cube_2",
                name="red_cube_2",
                position=np.array(left_squar_center),
                orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, angle_radians]), degrees=False),
                scale=np.array(left_square_size),
                color=np.array([0.5, 0.0, 0.2]),
            ))

            my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/red_cube_3",
                name="red_cube_3",
                position=np.array(right_squar_center),
                orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, angle_radians]), degrees=False),
                scale=np.array(right_square_size),
                color=np.array([0.5, 0.0, 0.2]),
            ))

            stick = DynamicCuboid(
                prim_path="/World/blue_cube",
                name="blue_cube",
                position=np.array([0.6, -0.45, 0.1]),
                orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=False),
                scale=np.array([0.3, 0.02, 0.04]),
                color=np.array([0.0, 0.0, 1]),
            )
            my_world.scene.add(stick)


            camera_x = random.uniform(0.5, 2)
            camera_y = random.uniform(-2, 1)
            camera_z = random.uniform(2, 4)

            #camera_x, camera_y, camera_z = 0.5 , -1,  2
            
            camera_delta_yaw = math.atan2(camera_y,(camera_x-0.4))
            camera_delta_pitch = math.atan2(camera_z-0.2,math.sqrt((camera_x-0.4)**2 + (camera_y+0.3)**2))
            camera_rotato = [0, camera_delta_pitch, math.pi+camera_delta_yaw] # [0,45,180]


            camera = Camera(
            prim_path="/World/camera",
            name="camera",
            position=np.array([camera_x, camera_y, camera_z]),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array(camera_rotato), degrees=False),
            )

            camera.initialize()
            camera.add_distance_to_image_plane_to_frame()



            rotation_matrix = euler_to_rot_matrix(np.array(camera_rotato), degrees=False)
            translation_vector = np.array([camera_x, camera_y, camera_z])
            RT = np.eye(4)  
            RT[:3, :3] = rotation_matrix  
            RT[:3, 3] = translation_vector  

            # target_position = np.array([ 0.48125303, -0.3530148 ,  0.3374037 ])
            # # 假设目标姿态为单位姿态（无旋转），实际应用中可能需要根据任务定义旋转
            # target_orientation =  np.array([ 0.00623622,  0.9999677 , -0.00152156,  0.00485821])

            
            # actions, succ = my_IK.compute_inverse_kinematics(
            #     target_position=target_position,
            #     target_orientation=target_orientation,
            # )
            # print(actions.joint_positions)
            # action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
            # franka.apply_action(action)
            # gripper_action = ArticulationAction(joint_positions=np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
            # franka.apply_action(gripper_action)

            i = 0

            #机械臂回初始位置
            #franka.set_joint_positions(np.array([0.01871685, -0.56626755,  0.00598636, -2.8088658 ,  0.0096361,   3.0243037, 0.7554296,  0.04, 0.04]))

            '''
            franka.gripper.set_joint_positions
            给9个参数，就是控制机械臂各个关节和夹爪
            给2个参数，就直接控制末端夹爪
            '''

            time.sleep(2)
            gif_image = []
            jump = 0
            effort = False

            med_inter_num = 400
            stage = 0
            while i < 1500:
                my_world.step(render=True)

                if i < 20:
                    time.sleep(0.01)
                    last_stage_i = i

                
                current_joint_positions = franka.get_joint_positions()[:7]

                if stage == 0:
                    if i == last_stage_i:
                        actions, succ = my_IK.compute_inverse_kinematics(
                        target_position=np.array([0.6, -0.45, 0.2]),
                        target_orientation= euler_angles_to_quat(np.array([0, np.pi, 0])))
                        interpolation = np.linspace(current_joint_positions, actions.joint_positions, med_inter_num, axis=0)


                    if i-last_stage_i >= med_inter_num:
                        inter_k = med_inter_num-1
                    else:
                        inter_k = i-last_stage_i
                    
                    action = ArticulationAction(joint_positions=interpolation[inter_k], joint_indices=np.array([0,1,2,3,4,5,6]))
                    franka.apply_action(action)

                    gripper_action = ArticulationAction(joint_positions=[0.4, 0.4], joint_indices=np.array([7, 8]))
                    franka.apply_action(gripper_action)

                    current_joint_positions = franka.get_joint_positions()[:7]
                    joint_delta = np.max(np.abs(current_joint_positions - actions.joint_positions))

                    if joint_delta < 0.0035:
                        stage = 1  
                        last_stage_i = i
                    
                if stage == 1:

                    if i == last_stage_i:
                        current_joint_positions = franka.get_joint_positions()[:7]
                        interpolation = MoveL(my_IK, 
                                                current_joint_positions, 
                                                np.array([0.6, -0.45, 0.14]),
                                                euler_angles_to_quat(np.array([0, np.pi, 0])),
                                                med_inter_num//2,
                                                Use_Jacbian = False)
                    if i-last_stage_i >= (med_inter_num//2):
                        inter_k = med_inter_num//2-1
                    else:
                        inter_k = i-last_stage_i

                    actions, succ = my_IK.compute_inverse_kinematics(
                    target_position= np.array([interpolation[inter_k][3], interpolation[inter_k][4], interpolation[inter_k][5]]),
                    target_orientation= euler_angles_to_quat(np.array([interpolation[inter_k][0], interpolation[inter_k][1], interpolation[inter_k][2]])))
                    
                    action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
                    franka.apply_action(action)

                    if (i-last_stage_i) >= med_inter_num//2:
                        stage = 2
                        last_stage_i = i
                        jump = 5

                if stage == 2:
                    gripper_action = ArticulationAction(joint_positions=[0,0], joint_indices=np.array([7, 8]))
                    franka.apply_action(gripper_action)
                    jump -= 1
                    if jump == 0:
                        stage = 3
                        last_stage_i = i

                    gripper_action = ArticulationAction(joint_positions=[0,0], joint_efforts = [-10,-10], joint_indices=np.array([7, 8]))
                    franka.apply_action(gripper_action)


                if stage ==3:

                    camera.get_current_frame()
                    image = camera.get_rgba()[:, :, :3]
                    
                    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    depth = camera.get_depth().astype(np.float16)
                    franka_arm_para = {}
                    current_joint_positions = franka.get_joint_positions()
                    #franka_arm_para['Translation']=franka.end_effector.get_world_pose()[0].tolist()
                    #franka_arm_para['Rotation']=quat_to_euler_angles(franka.end_effector.get_world_pose()[1].tolist())
                    franka_arm_para['Translation']=my_IK.compute_end_effector_pose()[0].tolist()
                    #franka_arm_para['Rotation']= matrix_to_euler_angles(my_IK.compute_end_effector_pose()[1]).tolist()
                    franka_arm_para['Rotation']= rot_matrix_to_quat(my_IK.compute_end_effector_pose()[1]).tolist()
                    franka_arm_para['EE'] = current_joint_positions[7:].tolist()

                    if jump == 0:
                        target_position, target_orientation, gripper_close, seq_data , jump =  diffusion_action_generation(i-last_stage_i, bgr_img, depth, franka_arm_para, RT, policy, seq_data, seq_length= cfg.set_seq_length)
                        actions, succ = my_IK.compute_inverse_kinematics(
                        target_position=target_position,
                        target_orientation=target_orientation,)
                        gif_image.append(Image.fromarray(image))

                    else:
                        jump -= 1
                        #print('-----------JUMP---------')

                    
                    action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
                    franka.apply_action(action)


                    if effort==True and jump == 0 :
                        gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([-10,-10]), joint_indices=np.array([7, 8]))
                    else:
                        gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([0,0]),  joint_indices=np.array([7, 8]))
                    franka.apply_action(gripper_action)
                    

                    if (gripper_close>0.035).all():
                        pose, quat = stick.get_world_pose()
                        place_angle = quat_to_euler_angles(quat)
                        #print(place_angle[2] , angle_radians)
                        #print(np.array(pose)[:2], np.array([center_x,center_y]))
                        if (np.max(np.abs(np.array(pose)[:2] - np.array([center_x,center_y]))) <= 0.03) and np.abs(place_angle[2] - angle_radians)<0.06 and pose[2]<0.15:
                            success_count += 1 
                            print('success times {:d}, current eposide {:d}'.format(success_count, t+1))
                            break

                

                i += 1

            gif_image[0].save('diffusion_insert_stick_{:d}.gif'.format(t),
               save_all=True,
               append_images=gif_image[1:],
               optimize=True,  # 启用优化，有助于减小文件大小
               duration=int( 0.2 * 1000),  # 转换为毫秒
               quality=50)
        break


    simulation_app.close()
