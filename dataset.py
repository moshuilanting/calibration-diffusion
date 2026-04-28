import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import ujson
import numpy as np
from rotations import quat_to_rot_matrix, euler_to_rot_matrix,rot_matrix_to_quat,quat_to_euler_angles
import pickle
from tqdm import tqdm


class Franka_Dataset_Memory(Dataset):
    def __init__(self, folder_path,  seq_length=5,  action_length=5, sample_num = -1):
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        # new_subfolders = []
        # for i in range(len(subfolders)):
        #     if int(subfolders[i].split('_')[-1]) % 6 ==0 or int(subfolders[i].split('_')[-1]) % 6 == 1:
        #         new_subfolders.append(subfolders[i])
        # subfolders = new_subfolders

        self.num_of_each_group = []

        self.seq_length = seq_length
        self.action_length = action_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量，像素值范围是[0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(os.path.join(folder_path, "data.pkl")):
            with open(os.path.join(folder_path, "data.pkl"), 'rb') as f:
                data = pickle.load(f)

            self.all_images = data['all_images'][:sample_num]
            self.all_depths = data['all_depths'][:sample_num]
            self.all_actions = data['all_actions'][:sample_num]
            self.all_RT = data['all_RT'][:sample_num]
            self.num_of_each_group = [len(self.all_images[i]) for i in range(len(self.all_images))]
            
            
            # self.all_images = []
            # self.all_depths = []
            # self.all_actions = []
            # self.all_RT = []

            # for i in range(len(data['all_images'])):
            #     s_images = []
            #     s_depths = []
            #     s_actions = []
            #     for j in range(len(data['all_images'][i])):
            #         if j>=18:
            #             s_images.append(data['all_images'][i][j])
            #             s_depths.append(data['all_depths'][i][j])
            #             s_actions.append(data['all_actions'][i][j])

            #     self.all_images.append(s_images)
            #     self.all_depths.append(s_depths)
            #     self.all_actions.append(s_actions)

            #     self.all_RT.append(data['all_RT'][i])
            # self.num_of_each_group = [len(self.all_images[i]) for i in range(len(self.all_images))]
            
            print("有效序列组数：", len(self.all_images))
            
            print("总计图片数：", sum(self.num_of_each_group))

        else:
            self.all_images = []
            self.all_depths = []
            self.all_actions = []
            self.all_RT = []
            for subfolder in tqdm(subfolders):
                image_files = [f for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.jpg', '.png', '.jpeg'))]
                image_files.sort(key=lambda x: int(os.path.splitext(x.name)[0]))

                depth_file = [f for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.npy')) and f.name.split('.')[0].isdigit()]
                depth_file.sort(key=lambda x: int(os.path.splitext(x.name)[0]))
                
                RT_file = [f for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('RT.npy'))]

                action_files = [f for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.json'))]
                action_files.sort(key=lambda x: int(os.path.splitext(x.name)[0]))
                
                assert len(image_files) == len(depth_file) == len(action_files), "RGB、Depth、动作, 数量不相等"

                self.num_of_each_group.append(len(image_files))

                group_images = []
                group_depths = []
                group_actions = []
                for i in range(len(image_files)):
                    image = cv2.imread(image_files[i].path)
                    group_images.append(image)

                    depth = np.load(depth_file[i].path)
                    group_depths.append(depth)

                    with open(action_files[i].path, 'r', encoding='UTF-8') as f:
                        #json_data = f.read()
                        action_data = ujson.load(f)
                    group_actions.append(action_data)

                # print(subfolder)
                # # 创建窗口并显示图像
                # cv2.namedWindow("Image Display", cv2.WINDOW_NORMAL)  # 可调整窗口大小
                # cv2.imshow("Image Display", image)
                
                # # 等待用户按键（0 表示无限等待，其他数字表示毫秒数）
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()

                self.all_images.append(group_images)
                self.all_depths.append(group_depths)
                self.all_actions.append(group_actions)
                self.all_RT.append(np.load(RT_file[0].path))
            
            '''
            import random
            # 生成一个随机排列的索引
            indices = list(range(len(self.all_images)))
            random.shuffle(indices)

            # 根据随机索引重排列表 A 和 B
            self.shuffled_images = [self.all_images[i] for i in indices]
            self.shuffled_depths = [self.all_depths[i] for i in indices]
            self.shuffled_actions = [self.all_actions[i] for i in indices]
            self.shuffled_RT = [self.all_RT[i] for i in indices]
            ''' 
            pkl_data = {'all_images': self.all_images, 'all_depths': self.all_depths, 'all_actions': self.all_actions , 'all_RT': self.all_RT}

            with open(os.path.join(folder_path, "data.pkl"), 'wb') as f:
                pickle.dump(pkl_data, f)

    def __len__(self):
        return sum(self.num_of_each_group)

    def __getitem__(self, index):
        """
        根据index进行选择
        """
        count = 0
        for k_group, size in enumerate(self.num_of_each_group):
            if count<=index and (count + size) > index:
                break
            count += size

        seq_image = []
        seq_depth = []
        seq_eef_state = []
        seq_action = []
        
        Base2Cam_RT = self.all_RT[k_group]

        # RT_Orientation = rot_matrix_to_quat(Base2Cam_RT[:3, :3])
        # RT_Translation = Base2Cam_RT[:3, 3]
        # Trans_Center = np.hstack((RT_Translation, RT_Orientation, np.array([0,0])))
        # Seq_Trans_Center = np.tile(Trans_Center, (self.seq_length+self.action_length, 1))

        # 当前状态的action
        for q in range(index - count - self.seq_length + 1, index - count + 1):
            if q < 0:
                image = self.all_images[k_group][0]
                depth = self.all_depths[k_group][0]
                action = self.all_actions[k_group][0]

            else:
                image = self.all_images[k_group][q]
                depth = self.all_depths[k_group][q]
                action = self.all_actions[k_group][q]

            current_image = self.transform(image)
            current_depth = np.nan_to_num(depth).astype(np.float32)

            if np.any(np.array(action['EE']) <0.035):
                """
                抓水果的时候，爪子闭合也就0.0301， 这里设置
                """
                EE = np.array([0,0])
            else:
                EE = np.array([0.04,0.04]) 

            #current_EE = np.hstack((np.array(action['Translation']), quat_to_euler_angles(np.array(action['Rotation'])), EE))
            current_EE = np.hstack((np.array(action['Translation']), np.array(action['Rotation']), EE))

            # #visual

            # depth = np.nan_to_num(depth).astype(np.float32)  # 去除NaN值
            # print(np.max(depth), np.min(depth))
            # #depth = np.clip(depth, a_min=0, a_max=65535) 
            # #print(np.max(depth), np.min(depth))

            # depth_map_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            # depth_map_display = depth_map_display.astype(np.uint8)

            # # 应用伪彩色映射到深度图以便更好地可视化
            # depth_map_display = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)
            # # print(image.shape,depth.shape)
            # combined_image = np.hstack((image, depth_map_display))

            # print(action)
            # # 显示拼接后的图像
            # cv2.imshow("combined_image", combined_image)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            seq_image.append(current_image)
            seq_depth.append(current_depth)
            seq_eef_state.append(current_EE)
            seq_action.append(current_EE)

        seq_image = np.stack(seq_image)
        seq_depth = np.stack(seq_depth)
        seq_eef_state = np.stack(seq_eef_state)

        #print('seq_eef_state: ', seq_eef_state)

        # 序列预测action
        for q in range(index - count + 1, index - count + self.action_length + 1):
            if q >= self.num_of_each_group[k_group]:
                next_action = self.all_actions[k_group][-1]
            else:
                next_action = self.all_actions[k_group][q]

            if np.any(np.array(next_action['EE']) <0.035):
                EE = np.array([0,0])
            else:
                EE = np.array([0.04,0.04]) 
            
            '''
            # 转换到相机坐标系下
            
            rotation_matrix = quat_to_rot_matrix(np.array(next_action['Rotation']))
            translation_vector = np.array(next_action['Translation'])
            BaseEE = np.eye(4)
            BaseEE[:3, :3] = rotation_matrix
            BaseEE[:3, 3] = translation_vector

            Base_Translation = translation_vector
            Base_Euler = quat_to_euler_angles(np.array(next_action['Rotation']))
            #print("Base_Translation: ",Base_Translation, "Base_Euler: ",Base_Euler)
            #print("Base2Cam_RT: ",Base2Cam_RT)
            CamEE = np.dot(Base2Cam_RT,BaseEE)
            
            Base2Cam_Orientation = rot_matrix_to_quat(Base2Cam_RT[:3, :3])
            Base2Cam_Translation = Base2Cam_RT[:3, 3]
            Base2Cam_Euler = quat_to_euler_angles(Base2Cam_Orientation)
            #print(Base2Cam_Translation,Base2Cam_Euler)

            Cam_Orientation = rot_matrix_to_quat(CamEE[:3, :3])
            Cam_Translation = CamEE[:3, 3]
            Cam_Euler = quat_to_euler_angles(Cam_Orientation)

            next_EE = np.hstack((Cam_Translation, Cam_Orientation, EE))
            '''

            #next_EE = np.hstack((np.array(next_action['Translation']), quat_to_euler_angles(np.array(next_action['Rotation'])), EE))
            next_EE = np.hstack((np.array(next_action['Translation']), np.array(next_action['Rotation']), EE))
            seq_action.append(next_EE)

            # depth = np.nan_to_num(depth).astype(np.float32)  # 去除NaN值
            # #depth = np.clip(depth, a_min=0, a_max=65535) 
            # #print(np.max(depth), np.min(depth))

            # depth_map_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            # depth_map_display = depth_map_display.astype(np.uint8)

            # # 应用伪彩色映射到深度图以便更好地可视化
            # depth_map_display = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)
            # # print(image.shape,depth.shape)
            # combined_image = np.hstack((image, depth_map_display))

            # # 显示拼接后的图像
            # cv2.imshow("combined_image", combined_image)
            # cv2.waitKey(100000)
            # cv2.destroyAllWindows()

        seq_action = np.stack(seq_action)
        #print('seq_action: ', seq_action)
        seq_depth = np.expand_dims(seq_depth, axis = 1)
        # print(seq_image.shape)
        # print(seq_depth.shape)
        seq_combined_image = np.concatenate((seq_image, seq_depth), axis = 1)
        #print("seq_combined_image: ",seq_combined_image.shape)


        '''
        是否添加初始的图像
        '''
        '''
        init_image = self.transform(self.all_images[k_group][0])
        init_depth = np.expand_dims(np.nan_to_num(self.all_depths[k_group][0]).astype(np.float32), axis = 0)
        init_combined_image = np.concatenate((init_image, init_depth), axis = 0)
        init_combined_image = np.expand_dims(init_combined_image, axis=0)
        init_combined_image = np.tile(init_combined_image, (len(seq_combined_image), 1, 1, 1))
        combined_image_result = np.concatenate((init_combined_image, seq_combined_image), axis=1)
        #print('result: ',result.shape)
        '''

        return seq_combined_image, seq_eef_state, seq_action#, Seq_Trans_Center



            



if __name__=="__main__":
    train_dataset = Franka_Dataset_Memory("/media/jtl/ZJRR8/0224_Franka_Pick_Simple_Fruits", seq_length=8, action_length=8)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers= 1)
    for seq_combine_image, seq_eef_state, seq_action in dataloader:
        print("seq_eef_state: ",seq_eef_state)
        print("seq_action: ",seq_action)
        input()
