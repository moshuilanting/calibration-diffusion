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
from omni.isaac.core.articulations import Articulation

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



def diffusion_action_generation(count, image, depth, status, Base2Cam_RT, policy, seq_data, seq_length, Trans_Noise):
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
    Trans_Center = Trans_Center + Trans_Noise

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
    if np.any(np.array(status['EE']) > 0.03) and np.any(EE_vector < 0.01):
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

model_name = "0727_noise_0.01_Franka_Arrange_Bottle_control_2_diffusion_finetune_sample_300_200.pth"
print(model_name)
#policy.load_state_dict(torch.load('hunder_diffusion_pull_1000.pth', map_location=device,weights_only=True))
policy.load_state_dict(torch.load(model_name, map_location=device,weights_only=True))



def put_bottle1(count,pos=[0.2, 0.65, 0.6]):
    prim_path = "/World/bottle1_{}".format(count)
    add_reference_to_stage(usd_path=usd_home + "/bottle_bottle1.usd", prim_path=prim_path)
    name = "usd_bottle1_{}".format(count)
    bottle1 = Articulation(prim_path=prim_path ,
                      name=name,
                      position=np.array([pos[0], pos[1],pos[2]+0.05]),
                      scale = np.array([0.6,0.6,0.5])
                      )
    
    return name, bottle1


def put_chip1(count, pos= [0.1, 0.65, 0.6]):
    prim_path = "/World/chip1_{}".format(count)
    add_reference_to_stage(usd_path=usd_home + "/bottle_chip1.usd", prim_path=prim_path)
    name="usd_chip1_{}".format(count)
    chip1 = Articulation(prim_path=prim_path ,
                      name=name,
                      position=np.array([pos[0],pos[1],pos[2]+0.1]),
                      #orientation = euler_angles_to_quat(np.array([1.57, 0, 0])),
                      scale = np.array([0.7,0.7,0.7])
                      )
    return name , chip1

def put_chip2(count, pos=[0.0, 0.65, 0.6]):
    prim_path = "/World/chip2_{}".format(count)
    add_reference_to_stage(usd_path=usd_home + "/bottle_chip2.usd", prim_path=prim_path)
    name="usd_chip2_{}".format(count)
    chip2 = Articulation(prim_path=prim_path ,
                      name=name,
                      position=np.array([pos[0],pos[1],pos[2]+0.1]),
                      #orientation = euler_angles_to_quat(np.array([1.57, 0, 0])),
                      scale = np.array([0.7,0.7,0.7])
                      )
    return name , chip2

def put_cola(count,pos =[-0.1, 0.65, 0.6]):    
    prim_path = "/World/cola_{}".format(count)
    add_reference_to_stage(usd_path=usd_home + "/bottle_cola.usd", prim_path=prim_path)
    name="usd_cola_{}".format(count)
    cola = Articulation(prim_path=prim_path ,
                      name=name,
                      position=np.array([pos[0],pos[1],pos[2]+0.1]),
                      scale = np.array([0.7, 0.7, 0.7])
                      ) 
    return name, cola


def put_milk(count ,pos = [-0.2, 0.65, 0.6]):
    prim_path = "/World/milk_{}".format(count)
    add_reference_to_stage(usd_path=usd_home + "/bottle_milk.usd", prim_path=prim_path)
    name="usd_milk_{}".format(count)
    milk = Articulation(prim_path=prim_path ,
                      name=name,
                      position=np.array([pos[0],pos[1],pos[2]+0.1]),
                      scale = np.array([0.4,0.4,0.5])
                      )
    return name, milk


if __name__=="__main__":  
    usd_home = '/home/jtl/ISAAC/cross_perspective_robotics_manipulation/calibration-diffusion/objects/usd/fruits/'
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane(static_friction=2,dynamic_friction=1, restitution = 0)

    action_registry = omni.kit.actions.core.get_action_registry()
    light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    #light_action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    light_action.execute()

    franka = my_world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
    my_IK = KinematicsSolver(franka, end_effector_frame_name="right_gripper")

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
            prim_path="/World/fixed_cube1",
            name="fixed_cube1",
            position=np.array([0.0 , 0.7, 0.4]),
            scale=np.array([0.8, 0.02, 0.8]),
            color=np.array([0.1, 0.3, 0.1])
        )
    )


    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube2",
            name="fixed_cube2",
            position=np.array([0.0 , 0.65, 0.4]),
            scale=np.array([0.8, 0.1, 0.02]),
            color=np.array([0.1, 0.1, 0.4])
        )
    )

    my_world.scene.add(
        FixedCuboid(
            prim_path="/World/fixed_cube3",
            name="fixed_cube3",
            position=np.array([0.0 , 0.65, 0.6]),
            scale=np.array([0.8, 0.1, 0.02]),
            color=np.array([0.1, 0.1, 0.4])
        )
    )

                                                                        
    my_IK = KinematicsSolver(franka, end_effector_frame_name="right_gripper")

    articulation_controller = franka.get_articulation_controller()

    object_names = []

    #old_joints = np.array([-0.35299405, -0.25012702, -0.1060075,  -2.4666014, -0.03301693,  2.217925, -0.64524966, 0.03999992, 0.04])
    while simulation_app.is_running():
        fruits = []
        success_count = 0 
        for t in range(50):
            print(t)
            for i in range(len(object_names)):
                my_world.scene.remove_object(name = object_names[i])

            object_names = []
            my_world.reset()


            location_pos = [0.2, 0.65, 0.4]
            location_pos[0] = random.uniform(0.15, 0.3)

            n = random.randint(0, 4)
            name, target_chip = put_chip1(1,pos = [location_pos[0]-0.1*n, location_pos[1], location_pos[2]])    
            my_world.scene.add(target_chip)
            object_names.append(name)
            
            obj_m = np.random.randint(3, 7)


            start_pos = [random.uniform(0, 0.3),0.65, 0.6]
            remove_m = np.random.randint(1, obj_m)

            for m in range(obj_m):
                if  m != remove_m:
                    name, chip1 = put_chip1(10+m,pos = [start_pos[0]-0.1*m, start_pos[1], start_pos[2]])    
                    my_world.scene.add(chip1)
                    object_names.append(name)

            target_pos = [start_pos[0]-0.1*remove_m, start_pos[1], start_pos[2]]
            #input()

            my_world.reset()


            r = random.choice([-1, 1])
            if r == 1:
                camera_x = random.uniform(0.3, 1)
            else:
                camera_x = random.uniform(-1,  -0.3)

            camera_y = random.uniform(-1.2, -0.8)
            camera_z = random.uniform(2, 2.5)


            camera_delta_pitch = math.atan2(camera_z-0.7, 0.65 - camera_y)
            camera_delta_yaw = math.atan2(camera_x, 0.7-camera_y)
            camera_rotato = [0, camera_delta_pitch, math.pi/2+camera_delta_yaw]



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

            my_world.reset()
            

            #机械臂回初始位置
            franka.set_joint_positions(np.array([0.01871685, -0.56626755,  0.00598636, -2.8088658 ,  0.0096361,   3.0243037, 0.7554296,  0.04, 0.04]))

            '''
            franka.gripper.set_joint_positions
            给9个参数，就是控制机械臂各个关节和夹爪
            给2个参数，就直接控制末端夹爪
            '''

            i = 0

            gif_image = []
            jump = 0
            effort = False
            Trans_Noise = np.random.normal(loc=0.0, scale=0.0, size=(9,))

            med_inter_num = 400

            time.sleep(1)
            while i < 1000:
                my_world.step(render=True)

                if i < 20:
                    time.sleep(0.01)
                    i+=1
                    last_stage_i = i
                    actions, succ = my_IK.compute_inverse_kinematics(
                        target_position=np.array([0, 0.5, 0.5]),
                        target_orientation= euler_angles_to_quat(np.array([0, np.pi/2, np.pi/2])))
                    
                    action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
                    franka.apply_action(action)

                    gripper_action = ArticulationAction(joint_positions=[0.4, 0.4], joint_indices=np.array([7, 8]))
                    franka.apply_action(gripper_action)
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
                    target_position, target_orientation, gripper_close, seq_data , jump =  diffusion_action_generation(i-20, bgr_img, depth, franka_arm_para, RT, policy, seq_data, seq_length= cfg.set_seq_length, Trans_Noise=Trans_Noise)
                    
                    if i > 20:
                        actions, succ = my_IK.compute_inverse_kinematics(
                        target_position=target_position,
                        target_orientation=target_orientation)
                        #print('diffusion start')

                        gif_image.append(Image.fromarray(image))

                        if np.any(gripper_close<0.02):
                            effort=True
                        else:
                            effort=False

                    else:
                        actions, succ = my_IK.compute_inverse_kinematics(
                        #target_position= np.array([0.0,0.5,0.4]),#np.array(my_IK.compute_end_effector_pose()[0].tolist()) + np.array([-0.1, 0.1, 0]), #+ np.random.randn(3)*0.05,
                        target_position= np.array([0, 0.5, 0.5]),
                        target_orientation= euler_angles_to_quat(np.array([0, np.pi/2, np.pi/2])))

                        # interpolation = np.linspace(current_joint_positions[:7], actions.joint_positions, 10, axis=0)
                        # actions.joint_positions = interpolation[1]

                        # print(np.array(my_IK.compute_end_effector_pose()[0].tolist()))
                        # print(np.array(my_IK.compute_end_effector_pose()[1].tolist()))

                else:
                    jump -= 1
                    #print('-----------JUMP---------')
                    
                action = ArticulationAction(joint_positions=actions.joint_positions, joint_indices=np.array([0,1,2,3,4,5,6]))
                franka.apply_action(action)

                gripper_action = ArticulationAction(joint_positions=gripper_close,  joint_indices=np.array([7, 8]))

                if effort== True and jump == 0 :
                    gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([-20,-20]), joint_indices=np.array([7, 8]))
                elif effort == False:
                    gripper_action = ArticulationAction(joint_positions=gripper_close, joint_efforts=np.array([0,0]),  joint_indices=np.array([7, 8]))
                
                franka.apply_action(gripper_action)


                # if stage == 4:
                #     for fruit_num,fruit_dict in enumerate(fruits):
                #         for key,food in list(fruit_dict.items()):
                #             #my_world.scene.remove_object(name="random_red_cube_{:d}".format(cube_index))
                #             pose = food.get_world_pose()[0] 
                #             if np.max(np.abs(np.array(pose)[:2] - np.array(fruits_location[key[:-2]])[:2])) > 0.05:
                #                 remove = True


                #     if remove == False:
                #         success_count += 1 
                #         print('success times {:d}, current eposide {:d}'.format(success_count, t+1))

                #     break

                #print(target_pos, target_chip.get_world_pose()[0])
                current_joint_positions = franka.get_joint_positions()
                if np.max(np.abs(current_joint_positions)) > 4:
                    break


                current_chip = target_chip.get_world_pose()[0]
                euclidean_distance = np.linalg.norm(target_pos[:2] - current_chip[:2])
                #print(euclidean_distance)
                if current_chip[2]> 0.6 and euclidean_distance < 0.02:
                    success_count += 1 
                    print('success times {:d}, current eposide {:d}'.format(success_count, t+1))
                    break

                i += 1

            gif_image[0].save('control_arrange_bottle_{:d}.gif'.format(t),
               save_all=True,
               append_images=gif_image[1:],
               optimize=True,  # 启用优化，有助于减小文件大小
               duration=int( 0.2 * 1000),  # 转换为毫秒
               quality=50)
        break


    simulation_app.close()
