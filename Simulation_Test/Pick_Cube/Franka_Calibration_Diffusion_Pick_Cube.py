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

from model.calibration_diffusion.configuration_diffusion import DiffusionConfig
from model.calibration_diffusion.modeling_calibration_diffusion import DiffusionPolicy

 
from diffusers.training_utils import EMAModel
import torch
from torchvision import transforms
import math
import random


from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
from omni.isaac.core import World
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.kit.actions.core



seq_data = [[], [], []]

transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量，像素值范围是[0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def generate_random_numbers():
    # 随机选择一个位置放置大于 0.7 的数
    index = random.randint(0, 5)

    colors_options = [[0.9,0.05,0.05],[0.8,0.1,0.1],
                      [0.05,0.9,0.05],[0.1,0.8,0.1],
                      [0.05,0.05,0.9],[0.1,0.1,0.8]
                      ]
    num1, num2, num3 = colors_options[index] 

    # num1, num2, num3 = 1, 0, 0
    
    # # 生成大于 0.7 的随机数
    # large_num = random.uniform(0.7, 1)

    # remaining = 1 - large_num
    # # 生成另外两个数
    # num1 = random.uniform(0, remaining)
    # num2 = remaining - num1

    return num1, num2, num3


def Coord_Trans(vector, quat, RT_matrix):
    """
    vector : 在当前坐标系下的位置
    quat : 在当前坐标系下的四元数
    RT_matrix: 旋转矩阵

    return:
    新坐标系下的位移、新坐标系下的四元数
    """
    rotation_matrix = quat_to_rot_matrix(np.array(quat))
    translation_vector = np.array(vector)
    BaseEE = np.eye(4)  #Base 坐标系下的位置
    BaseEE[:3, :3] = rotation_matrix  
    BaseEE[:3, 3] = translation_vector

    CamEE = np.dot(RT_matrix,BaseEE)
    Cam_Orientation = rot_matrix_to_quat(CamEE[:3, :3])
    Cam_Translation = CamEE[:3, 3]
    Cam_Euler = quat_to_euler_angles(Cam_Orientation)
    #current_Cam_EE = np.hstack((Cam_Translation, Cam_Orientation, EE))

    return Cam_Translation, Cam_Orientation



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
    
    #print('seq_image: ', len(seq_image), count)

    current_image = transform(image)
    current_depth = np.nan_to_num(depth).astype(np.float32)
    if np.any(np.array(status['EE']) <0.03):
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

    Trans_Center_Translation = Base2Cam_RT[:3, 3]
    Trans_Center_Orientation = rot_matrix_to_quat(Base2Cam_RT[:3, :3])
    Trans_Center = np.hstack((Trans_Center_Translation, Trans_Center_Orientation, [0,0]))
    Seq_Trans_Center = np.tile(Trans_Center, (cfg.set_seq_length+cfg.action_seq_length, 1))
    Seq_Trans_Center = np.expand_dims(Seq_Trans_Center, axis = 0)
    Seq_Trans_Center = torch.from_numpy(Seq_Trans_Center).float().cuda()

    pred_action = policy.select_action(obs=torch_np_seq_combined_image, state= torch_seq_status, trans_center = Seq_Trans_Center).cpu().detach().numpy()[0][0]

    #print(pred_action.shape)

    # quat = euler_angles_to_quat(pred_action[3:6])
    # translation = np.array(pred_action[:3])
    # EE_vector = np.array(pred_action[6:])

    quat = pred_action[3:7]
    translation = np.array(pred_action[:3])
    EE_vector = np.array(pred_action[7:])
    
    if np.sum(np.isnan(translation)) > 0:
        print(pred_action)
        print(status)
        input('error')
    translation[np.isnan(translation)] = np.array(status['Translation'])[np.isnan(translation)]
    quat[np.isnan(quat)] = np.array(status['Rotation'])[np.isnan(quat)]

    Base_Vec, Base_Quat = Coord_Trans(translation, quat, np.linalg.inv(Base2Cam_RT))
    # print(Base_Vec, Base_Quat)
    # input()

    #current_EE = np.hstack((np.array(status['Translation']), np.array(status['Rotation']), EE))
    #current_Cam_EE = np.hstack((Cam_Vec, Cam_Quat, EE))

    # if np.any(np.array(pred_action[7:]) <0.035):
    #     EE_vector = np.array([0,0])
    # else:
    #     EE_vector = np.array([0.04,0.04])

    jump = 3
    if np.any(np.array(status['EE']) > 0.03) and np.any(EE_vector < 0.01):
        jump = 6
        print(np.array(status['EE']), EE_vector)

    # rotation_matrix = quat_to_rot_matrix(pred_action[3:7])
    # translation_vector = np.array(pred_action[:3])
    # EE_vector = np.array(pred_action[7:])
    # CameraEE = np.eye(4)
    # CameraEE[:3, :3] = rotation_matrix  
    # CameraEE[:3, 3] = translation_vector  
    
    # BaseEE = np.dot(np.linalg.inv(Base2Cam_RT),CameraEE)

    # Base_Orientation = rot_matrix_to_quat(BaseEE[:3, :3])
    # Base_Translation = BaseEE[:3, 3]

    #print(Base_Translation, Base_Orientation)
    #return Base_Translation, Base_Orientation, EE_vector , [seq_image, seq_depth, seq_status]
    #print('pred_action: ', pred_action)

    return translation, quat, EE_vector , [seq_image, seq_depth, seq_status], jump


device = torch.device("cuda:0")
cfg = DiffusionConfig()
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
policy = DiffusionPolicy(cfg)
policy.eval()
policy.to(device)
model_name = "Eplison_DDPM_0425_0424_Franka_Pick_Cube_Y_2.0_control_2_diffusion_finetune_sample_300_200.pth"
print(model_name)
policy.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))


if __name__=="__main__":
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane(static_friction=2,dynamic_friction=1,restitution = 0)
    franka = my_world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    #franka_gripper = franka.gripper()
    action_registry = omni.kit.actions.core.get_action_registry()
    light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    #light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    light_action.execute()

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube",
            name="fixed_cube",
            position=np.array([0.4, -0.2, 0.0]),
            scale=np.array([0.2, 0.2, 0.01]),
            color=np.array([0, 1.0, 0.0]),
        )
    )
                                                                                
    my_IK = KinematicsSolver(franka, end_effector_frame_name="right_gripper")
    #my_IK = KinematicsSolver(franka, end_effector_frame_name="panda_rightfinger")

    articulation_controller = franka.get_articulation_controller()


    my_world.reset()

    while simulation_app.is_running():
        success_count = 0

        for t in range(50):

            goal_position = [0.4, -0.2]
                
            while True:
                '''
                极坐标系内取点
                '''
                cuboid_position = [0.4, 0.2]

                u = random.uniform(0, 1)
                v = random.uniform(0, 1)

                # # 扇形半径范围是[0,1]，起始角度是0，终止角度是pi/2
                # r_min = 0.3
                # r_max = 0.7
                # theta_start =  - np.pi/2 + np.pi/5
                # theta_end = np.pi/2 - np.pi/5

                r_min = 0.3
                r_max = 0.5
                theta_start =  -0.3  #-0.942
                theta_end = 0.3       #0.942

                # 根据u确定半径
                r = r_min + u * (r_max - r_min)
                # 根据v确定角度
                theta = theta_start + v * (theta_end - theta_start)
                # 将极坐标转换为直角坐标
                cuboid_position[0] = r * np.cos(theta)
                cuboid_position[1] = r * np.sin(theta)
                #cuboid_position = [0.4, 0.2]

                #cuboid_position = [random.uniform(0.1, 0.6), random.uniform(-0.45, 0.45)]
                #cuboid_position = [0.4, 0.2]
                distance = math.sqrt(cuboid_position[0] ** 2 + cuboid_position[1] ** 2)
                target_dis = math.sqrt((goal_position[0]-cuboid_position[0])** 2 + (goal_position[1]-cuboid_position[1]) ** 2)
                if distance<0.7 and distance>0.25 and target_dis> 0.15:
                    break

            print(t)
            print("距离goal的距离:",target_dis, "距离base的距离:", distance)


            my_world.scene.remove_object(name="camera")
            my_world.scene.remove_object(name="fancy_cube")

            camera_x = random.uniform(2, 3)
            camera_y = random.uniform(-2, 2)
            #camera_y = 0
            camera_z = random.uniform(2, 3)
            camera_delta_yaw = math.atan2(camera_y,camera_x)
            camera_delta_pitch = math.atan2(camera_z,math.sqrt((camera_x-0.5)**2 + camera_y**2))
            camera_rotato = [0, camera_delta_pitch, math.pi+camera_delta_yaw] # [0,45,180]

            #camera_x,camera_y,camera_z =  2.8784, -0.1935,  2.913
            #camera_rotato = [0 , 8.84465421e-01 , 3.0744]

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

            red_color, green_color, blue_color = generate_random_numbers()
            #cube_yaw = random.uniform(-1, 1)
            cube_yaw = 0
            my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([cuboid_position[0], cuboid_position[1], 0.1]),
                orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, cube_yaw]), degrees=False),
                scale=np.array([0.0515, 0.0515, 0.0515]),
                #color=np.array([1, 0, 0]),
                color=np.array([red_color, green_color, blue_color]),
                
            ))

            #_franka = my_world.scene.get_object("fancy_franka")
            _fancy_cube = my_world.scene.get_object("fancy_cube")


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
            franka.set_joint_positions(np.array([0.01871685, -0.56626755,  0.00598636, -2.8088658 ,  0.0096361,   3.0243037, 0.7554296,  0.04, 0.04]))
            
            '''
            franka.gripper.set_joint_positions
            给9个参数，就是控制机械臂各个关节和夹爪
            给2个参数，就直接控制末端夹爪
            '''

            time.sleep(3)
            gif_image = []
            jump = 0
            while i < 300:
                my_world.step(render=True)

                if i < 15:
                    i+=1
                    #time.sleep(0.01)
                    continue

                camera.get_current_frame()
                image = camera.get_rgba()[:, :, :3]
                
                # plt.clf()
                # plt.imshow(camera.get_rgba()[:, :, :3])
                # plt.show()
                # time.sleep(1)
                # plt.close()

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
                    target_position, target_orientation, gripper_close, seq_data, jump =  diffusion_action_generation(i-15, bgr_img, depth, franka_arm_para, RT, policy, seq_data, seq_length= cfg.set_seq_length)
                    #print(target_position, target_orientation)
                    actions, succ = my_IK.compute_inverse_kinematics(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    )
                    gif_image.append(Image.fromarray(image))
                else:
                    jump -= 1
                    #print('-----------JUMP---------')

                # target_position = np.array([ 0.48125303, -0.3530148+i ,  0.3374037 ])
                # print(target_position)
                # # 假设目标姿态为单位姿态（无旋转），实际应用中可能需要根据任务定义旋转
                # target_orientation =  np.array([ 0.00623622,  0.9999677 , -0.00152156,  0.00485821])

            
                
                action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
                franka.apply_action(action)

                gripper_action = ArticulationAction(joint_positions=gripper_close, joint_indices=np.array([7, 8]))
                franka.apply_action(gripper_action)

                time.sleep(0.1)

                #franka.apply_action(actions)
                #gripper_action = ArticulationAction(joint_positions=gripper_close, joint_indices=np.array([7, 8]))
                #franka.apply_action(gripper_action)

                set_action = np.concatenate((actions.joint_positions, gripper_close), axis=0)

                #print('set_joint_action: ',set_action)
                #franka.set_joint_positions(actions.joint_positions)

                #franka.end_effector.set_world_pose(target_position, target_orientation)

                # if i <=0.2:
                #     franka.apply_action(actions)

                # if i >0.2:
                #     print(i)
                #     gripper_action = ArticulationAction(joint_positions=gripper_close, joint_indices=np.array([7, 8]))
                #     franka.apply_action(gripper_action)

                #articulation_controller.apply_action(actions)

                #print('目标关节角度:',actions.joint_positions)

                current_joint_positions = franka.get_joint_positions()
                cube_position, _ = _fancy_cube.get_world_pose()

                #print('当前关节角度： ', current_joint_positions)

                distance = np.sqrt(np.sum((cube_position[:2] - np.array(cuboid_position))**2))
                if distance<0.1 and  cube_position[2] > 0.05:
                    success_count += 1 
                    print('success times {:d}, current eposide {:d}'.format(success_count, t+1))
                    break
                elif np.sqrt(np.sum(cube_position[:2]**2))>0.8:
                    print('{:d} fail '.format(t), np.sqrt(np.sum(cube_position[:2]**2)))
                    break

                # distance = np.sqrt(np.sum((cube_position[:2] - np.array(goal_position))**2))
                # if distance<0.1 and  cube_position[2] < 0.045:
                #     success_count += 1 
                #     print('success times {:d}, current eposide {:d}'.format(success_count, t+1))
                #     break
                # elif np.sqrt(np.sum(cube_position[:2]**2))>0.8:
                #     print('{:d} fail '.format(t), np.sqrt(np.sum(cube_position[:2]**2)))
                #     break

                #if cube_position[:2] - np.array(target_position[:2]) and cube_position[2] == 0.03075:
                #print('--------------------------')
                i += 1
            print('Final Distance: ',distance)
            gif_image[0].save('control_diffusion_pick_cube_{:d}.gif'.format(t),
               save_all=True,
               append_images=gif_image[1:],
               optimize=True,  # 启用优化，有助于减小文件大小
               duration=int( 0.2 * 1000),  # 转换为毫秒
               quality=50)

        break


    simulation_app.close()
