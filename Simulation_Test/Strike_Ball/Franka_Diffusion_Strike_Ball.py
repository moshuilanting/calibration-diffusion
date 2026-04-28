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
random.seed(10)
torch.manual_seed(10)
np.random.seed(10)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, FixedCuboid
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core import World
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.kit.actions.core
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name


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

    Base_Vec, Base_Quat = Coord_Trans(translation, quat, np.linalg.inv(Base2Cam_RT))
    #current_EE = np.hstack((np.array(status['Translation']), np.array(status['Rotation']), EE))
    #current_Cam_EE = np.hstack((Cam_Vec, Cam_Quat, EE))
    
    if np.any(np.array(pred_action[7:]) <0.035):
        EE_vector = np.array([0,0])
    else:
        EE_vector = np.array([0.04,0.04])

    jump = 5


    return translation, quat, EE_vector , [seq_image, seq_depth, seq_status], jump



device = torch.device("cuda")
cfg = DiffusionConfig()

policy = DiffusionPolicy(cfg)
policy.eval()
policy.to(device)

model_name = "Eplison_DDPM_0505_Franka_Strike_Ball_diffusion_sample_300_1200.pth"
print(model_name)
#policy.load_state_dict(torch.load('hunder_diffusion_pull_1000.pth', map_location=device,weights_only=True))
policy.load_state_dict(torch.load(model_name, map_location=device,weights_only=True))


if __name__=="__main__":
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane(static_friction=2,dynamic_friction=1, restitution = 0)

    action_registry = omni.kit.actions.core.get_action_registry()
    light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    #light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    light_action.execute()

    franka = my_world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    my_IK = KinematicsSolver(franka, end_effector_frame_name="right_gripper")


    static_friction = 0.0
    dynamic_friction = 0.0
    restitution = 1
    physics_material_path = find_unique_string_name(
                        initial_name="/World/Physics_Materials/physics_material",
                        is_unique_fn=lambda x: not is_prim_path_valid(x),
                    )

    physics_material = PhysicsMaterial(
                        prim_path=physics_material_path,
                        dynamic_friction=dynamic_friction,
                        static_friction=static_friction,
                        restitution=restitution,
                    )


    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube1",
            name="fixed_cube",
            position=np.array([0.6 , 0.0, 0.2]),
            scale=np.array([1.0, 0.6, 0.02]),
            color=np.array([0.1, 0.1, 0.1]),
            physics_material = physics_material
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube2",
            name="fixed_cube_forward",
            position=np.array([1.1 , 0.0, 0.1]),
            scale=np.array([0.02, 0.6, 0.2]),
            color=np.array([0.1, 0.1, 0.7])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube3",
            name="fixed_cube_back",
            position=np.array([0.1 , 0.0, 0.1]),
            scale=np.array([0.02, 0.6, 0.2]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube4",
            name="fixed_cube_left",
            position=np.array([0.6 , 0.3, 0.1]),
            scale=np.array([1.0, 0.02, 0.2]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube5",
            name="fixed_cube_right",
            position=np.array([0.6 , -0.3, 0.1]),
            scale=np.array([1.0, 0.02, 0.2]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube6",
            name="fixed_cube_fruits_clip6",
            position=np.array([1.2 , 0.0, 0.05]),
            scale=np.array([0.2, 0.6, 0.02]),
            color=np.array([0.1, 0.1, 0.7])
        )
    )


    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube7",
            name="fixed_cube_fruits_clip7",
            position=np.array([1.3 , 0.0, 0.1]),
            scale=np.array([0.02, 0.6, 0.2]),
            color=np.array([0.1, 0.1, 0.7])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube8",
            name="fixed_cube_fruits_clip8",
            position=np.array([1.2 , -0.3, 0.1]),
            scale=np.array([0.2, 0.02, 0.2]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube9",
            name="fixed_cube_fruits_clip9",
            position=np.array([1.2 , -0.1, 0.1]),
            scale=np.array([0.2, 0.02, 0.2]),
            color=np.array([0.1, 0.1, 0.7])
        )
    )
    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube10",
            name="fixed_cube_fruits_clip10",
            position=np.array([1.2 , 0.1, 0.1]),
            scale=np.array([0.2, 0.02, 0.2]),
            color=np.array([0.1, 0.1, 0.7])
        )
    )
    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube11",
            name="fixed_cube_fruits_clip11",
            position=np.array([1.2 , 0.3, 0.1]),
            scale=np.array([0.2, 0.02, 0.2]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube12",
            name="fixed_cube_fruits_clip12",
            position=np.array([1.2 , -0.2, 0.2]),
            scale=np.array([0.2, 0.2, 0.02]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube13",
            name="fixed_cube_fruits_clip13",
            position=np.array([1.2 , 0.2, 0.2]),
            scale=np.array([0.2, 0.2, 0.02]),
            color=np.array([0.1, 0.1, 0.1])
        )
    )

    #my_world.reset()

    #old_joints = np.array([-0.35299405, -0.25012702, -0.1060075,  -2.4666014, -0.03301693,  2.217925, -0.64524966, 0.03999992, 0.04])
    while simulation_app.is_running():

        success_count = 0
        for t in range(50):

            my_world.scene.remove_object(name="Dynamic_Sphere")

            my_world.reset()


            ball_x , ball_y, ball_z = random.uniform(0.5, 0.7) , random.uniform(-0.25, 0.25), 0.24
            target_x, target_y  = 1.2, 0 

            sphere =DynamicSphere(
                    prim_path="/World/Dynamic_Sphere",
                    name="Dynamic_Sphere",
                    position=np.array([ball_x , ball_y, ball_z]),
                    scale=np.array([0.04, 0.04, 0.04]),
                    color=np.array([1, 0.01, 0.01]),
                    physics_material = physics_material)
            my_world.scene.add(sphere)


            camera_x = random.uniform(1.6, 2.2)
            camera_y = random.uniform(-1.5, 1.5)
            camera_z = random.uniform(1.5, 2)
            
            #camera_x, camera_y, camera_z = 1.8 , 0.0,  2.0
            
            camera_delta_yaw = math.atan2(camera_y,(camera_x-0.7))
            camera_delta_pitch = math.atan2(camera_z-0.2,math.sqrt((camera_x-0.8)**2 + (camera_y)**2))
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
            #print(camera.get_intrinsics_matrix())
            rotation_matrix = euler_to_rot_matrix(np.array(camera_rotato), degrees=False)
            translation_vector = np.array([camera_x, camera_y, camera_z])
            RT = np.eye(4)  
            RT[:3, :3] = rotation_matrix  
            RT[:3, 3] = translation_vector  


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
            while i < 350:
                my_world.step(render=True)

                if i < 20:
                    i+=1
                    #time.sleep(0.01)
                    last_stage_i = i
                    continue
                

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

                #input()
                # if effort==True and jump == 0 :
                #     gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([-10,-10]), joint_indices=np.array([7, 8]))
                # else:
                #     gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([0,0]),  joint_indices=np.array([7, 8]))

                pose = sphere.get_world_pose()[0]
                if np.max(np.abs(np.array(pose)[:2] - np.array([target_x,target_y]))) <= 0.03:
                    success_count += 1 
                    print('success times {:d}, current eposide {:d}'.format(success_count, t+1))
                    break
                elif np.max(np.abs(np.array(pose)[:2] - np.array([target_x,target_y]))) > 0.03 and i>300:
                    print('{:d} fail '.format(t))
                    break

                i += 1

            gif_image[0].save('diffusion_strike_ball_{:d}.gif'.format(t),
               save_all=True,
               append_images=gif_image[1:],
               optimize=True,  # 启用优化，有助于减小文件大小
               duration=int( 0.2 * 1000),  # 转换为毫秒
               quality=50)
        break


    simulation_app.close()
