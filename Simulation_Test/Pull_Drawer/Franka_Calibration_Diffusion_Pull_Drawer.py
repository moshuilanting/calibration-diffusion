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
import numpy as np
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

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

    Trans_Center_Translation = Base2Cam_RT[:3, 3]
    Trans_Center_Orientation = rot_matrix_to_quat(Base2Cam_RT[:3, :3])
    Trans_Center = np.hstack((Trans_Center_Translation, Trans_Center_Orientation, [0,0]))
    Seq_Trans_Center = np.tile(Trans_Center, (cfg.set_seq_length+cfg.action_seq_length, 1))
    Seq_Trans_Center = np.expand_dims(Seq_Trans_Center, axis = 0)
    Seq_Trans_Center = torch.from_numpy(Seq_Trans_Center).float().cuda()

    #Seq_Trans_Center = torch.zeros_like(Seq_Trans_Center).float().cuda()

    pred_action = policy.select_action(obs=torch_np_seq_combined_image, state= torch_seq_status, trans_center = Seq_Trans_Center).cpu().detach().numpy()[0][0]

    #print(pred_action.shape)

    # quat = euler_angles_to_quat(pred_action[3:6])
    # translation = np.array(pred_action[:3])
    # EE_vector = np.array(pred_action[6:])

    quat = pred_action[3:7]
    translation = np.array(pred_action[:3])
    EE_vector = np.array(pred_action[7:])

    translation[np.isnan(translation)] = 0
    quat[np.isnan(quat)] = 0

    Base_Vec, Base_Quat = Coord_Trans(translation, quat, np.linalg.inv(Base2Cam_RT))
    #current_EE = np.hstack((np.array(status['Translation']), np.array(status['Rotation']), EE))
    #current_Cam_EE = np.hstack((Cam_Vec, Cam_Quat, EE))
    
    if np.any(np.array(pred_action[7:]) <0.035):
        EE_vector = np.array([0,0])
    else:
        EE_vector = np.array([0.04,0.04])

    jump = 5
    if np.any(np.array(status['EE']) > 0.035) and np.any(EE_vector < 0.01):
        print(np.array(status['EE']), EE_vector)
        jump = 10
    
    #print("EE_vector: ",EE_vector)

    # if translation[2] < 0.045:
    #     EE_vector = np.array([0,0])


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


device = torch.device("cuda")
cfg = DiffusionConfig()

policy = DiffusionPolicy(cfg)
policy.eval()
policy.to(device)


model_name = 'Target_DDPM_0425_Franka_Pull_Drawer_control_2_diffusion_finetune_sample_300_200.pth'
print('model_name: ',model_name)
#policy.load_state_dict(torch.load('hunder_diffusion_pull_1000.pth', map_location=device,weights_only=True))
policy.load_state_dict(torch.load(model_name, map_location=device,weights_only=True))


'''
from model.diffusion.configuration_diffusion import DiffusionConfig as BaseDiffusionConfig
from model.diffusion.modeling_diffusion import DiffusionPolicy as BaseDiffusionPolicy

random.seed(10)
torch.manual_seed(10)
cfg = BaseDiffusionConfig()
policy = BaseDiffusionPolicy(cfg)


# 从文件加载模型 2 的权重
ControlPolicy = torch.load(model_name, map_location=device)
# 创建一个新的状态字典，只包含模型 1 有的部分
pretrained_dict = {k: v for k, v in ControlPolicy.items() if k in policy.state_dict()}

# 更新模型 1 的状态字典
model1_state_dict = policy.state_dict()
model1_state_dict.update(pretrained_dict)

# 加载更新后的状态字典到模型 1
policy.load_state_dict(model1_state_dict)
policy.eval()
policy.to(device)
'''

if __name__=="__main__":
    usd_home = '/home/jtl/ISAAC/cross_perspective_robotics_manipulation/calibration-diffusion/objects/usd/pull/'

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane(static_friction=2,dynamic_friction=1,restitution = 0)
    franka = my_world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    action_registry = omni.kit.actions.core.get_action_registry()
    light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    #light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    light_action.execute()

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube",
            name="fixed_cube",
            position=np.array([1.2 , 0.0, 0.1]),
            scale=np.array([1.2, 2, 0.2]),
            color=np.array([0.0, 0.0, 0.0])
        )
    )
                                                                                
    my_IK = KinematicsSolver(franka, end_effector_frame_name="right_gripper")
    #my_IK = KinematicsSolver(franka, end_effector_frame_name="panda_rightfinger")

    articulation_controller = franka.get_articulation_controller()


    #my_world.reset()

    #old_joints = np.array([-0.35299405, -0.25012702, -0.1060075,  -2.4666014, -0.03301693,  2.217925, -0.64524966, 0.03999992, 0.04])
    while simulation_app.is_running():
        success_count = 0
        for t in range(0,50):
            
            my_world.scene.remove_object(name="camera")
            my_world.scene.remove_object(name="drawer2")
            my_world.reset()
            delta_x, delta_y, delta_z = random.uniform(-0.3, 0.0),random.uniform(-0.5, 0.1), 0
            #delta_x, delta_y, delta_z = 0,0,0

            print('drawer pos: ', np.array([1.2+delta_x, 0.2+delta_y, 0.5]))

            prim_path = "/World/drawer2"
            add_reference_to_stage(usd_path=usd_home + "/drawer3.usd", prim_path=prim_path)
            drawer2 = XFormPrim(prim_path=prim_path ,
                            name="drawer2",
                            position=np.array([1.2+delta_x, 0.2+delta_y, 0.5]),
                            orientation=euler_angles_to_quat(np.array([0,0,np.pi])),
                            scale = np.array([0.6,0.6,0.7]),
                            )

            my_world.scene.add(drawer2)


            camera_x = random.uniform(-2, 0.5)
            if random.choice([True, False]):
                camera_y = random.uniform(3, 5)
            else:
                camera_y = random.uniform(-5, -3)
            camera_z = random.uniform(1.5, 4)

            camera_delta_yaw = math.atan2(camera_y,(camera_x-1))
            camera_delta_pitch = math.atan2(camera_z-0.2,math.sqrt((camera_x-1)**2 + camera_y**2))
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
            franka.set_joint_positions(np.array([0.01871685, -0.56626755,  0.00598636, -2.8088658 ,  0.0096361,   3.0243037, 0.7554296,  0.04, 0.04]))

            '''
            franka.gripper.set_joint_positions
            给9个参数，就是控制机械臂各个关节和夹爪
            给2个参数，就直接控制末端夹爪
            '''

            time.sleep(2)
            gif_image = []
            jump = 0
            effort = False
            while i < 500:
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
                if i == 15:
                    green_channel = image[:, :, 1]
                    # 检查红色和蓝色分量（第一个和第三个维度）是否小于 100 且绿色分量大于 200
                    condition = ((image[:, :, 0] < 100) & (image[:, :, 2] < 100) & (green_channel > 200))
                    # 统计满足条件的像素点的数量
                    init_count = np.sum(condition)
                    print("init_count: ", init_count)

                ''' 
                # 查看绿色区域

                image_array = np.array(image)
                green_channel = image_array[:, :, 1]
                mask = ((green_channel > 200) & (image_array[:, :, 0] < 100) & (image_array[:, :, 2] < 100))
                result_image = np.zeros_like(image_array)
                # 将满足条件的像素设置为原始图像的像素值
                result_image[mask] = image_array[mask]
                # 保存结果图像
                cv2.imwrite('processed_image.jpg', result_image)
                input()
                '''

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
                    target_position, target_orientation, gripper_close, seq_data , jump =  diffusion_action_generation(i-15, bgr_img, depth, franka_arm_para, RT, policy, seq_data, seq_length= cfg.set_seq_length)
                    actions, succ = my_IK.compute_inverse_kinematics(
                    target_position=target_position,
                    target_orientation=target_orientation,)

                    gif_image.append(Image.fromarray(image))

                    green_channel = image[:, :, 1]
                    # 检查红色和蓝色分量（第一个和第三个维度）是否小于 100 且绿色分量大于 200
                    condition = ((image[:, :, 0] < 100) & (image[:, :, 2] < 100) & (green_channel > 200))
                    # 统计满足条件的像素点的数量
                    current_count = np.sum(condition)
                    if (current_count > 3*init_count and current_count > 800) or current_count>3000:
                        success_count += 1 
                        print("current_count: ",current_count)
                        print('success times {:d}, current eposide {:d}'.format(success_count, t+1))
                        break

                    if np.any(gripper_close<0.03):
                        effort=True
                    else:
                        effort=False
                    
                else:
                    jump -= 1
                    #print('-----------JUMP---------')
                    

                # target_position = np.array([ 0.48125303, -0.3530148+i ,  0.3374037 ])
                # print(target_position)
                # # 假设目标姿态为单位姿态（无旋转），实际应用中可能需要根据任务定义旋转
                # target_orientation =  np.array([ 0.00623622,  0.9999677 , -0.00152156,  0.00485821])

                action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
                franka.apply_action(action)

                if effort==True and jump==0 :
                    gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([-10,-10]), joint_indices=np.array([7, 8]))
                else:
                    gripper_action = ArticulationAction(joint_positions=gripper_close,  joint_indices=np.array([7, 8]))

                franka.apply_action(gripper_action)

                #print(gripper_close,franka.get_joint_positions()[7:])

                #franka.apply_action(actions)
                #gripper_action = ArticulationAction(joint_positions=gripper_close, joint_indices=np.array([7, 8]))
                #franka.apply_action(gripper_action)

                set_action = np.concatenate((actions.joint_positions, gripper_close), axis=0)

                #print('set_joint_action: ',set_action)
                #franka.set_joint_positions(set_action)

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

                #drawer_position, _ = drawer2.get_world_pose()
                #print("get_world_pose: ", drawer_position, _)
                #get_local_pose = drawer2.get_local_pose()
                #print("get_local_pose: ", get_local_pose)
                #drawer_state = articulation_view.get_world_poses()
                #print("drawer_state: ", drawer_state)
                #drawer_default_state = drawer2.get_default_state()
                #print("drawer_default_state: ", drawer_default_state.position)
                #input()

                #if cube_position[:2] - np.array(target_position[:2]) and cube_position[2] == 0.03075:
                #print('--------------------------')
                i += 1
                #print(my_IK.compute_end_effector_pose()[0], np.array([0.87+delta_x, 0.2+delta_y, 0.32]))
            gif_image[0].save('control_diffusion_pull_drawer_{:d}.gif'.format(t),
               save_all=True,
               append_images=gif_image[1:],
               optimize=True,  # 启用优化，有助于减小文件大小
               duration=int( 0.2 * 1000),  # 转换为毫秒
               quality=50)

        break


    simulation_app.close()
