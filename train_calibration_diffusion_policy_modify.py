from model.calibration_diffusion.configuration_diffusion import DiffusionConfig
from model.calibration_diffusion.modeling_calibration_diffusion import DiffusionPolicy

from model.diffusion.modeling_diffusion import DiffusionPolicy as BaseDiffusionPolicy
from model.diffusion.configuration_diffusion import DiffusionConfig as BaseDiffusionConfig

import torch
from dataset_center import Franka_Dataset_Memory
from torch.optim.lr_scheduler import StepLR

if __name__=="__main__":
    batch_size = 100
    device = torch.device("cuda:0")

    task_name = "Franka_Arrange_Bottle"
    dataset_path = "/media/jtl/ZJRR8/Free_View_DataSet" + task_name

    base_cfg = BaseDiffusionConfig()
    base_policy = BaseDiffusionPolicy(base_cfg)
    base_policy.to(device)
    base_policy.load_state_dict(torch.load('xxx.pth', 
                                      map_location=device, weights_only=True))

    cfg = DiffusionConfig()
    policy = DiffusionPolicy(cfg)
    policy.to(device)
    
    #optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    #dataset_path = "/media/jtl/ZJRR8/Single_Prespective_Data"

    base_policy_state_dict = base_policy.state_dict()
    policy_state_dict = policy.state_dict()

    for name, param in policy_state_dict.items():

        if name in base_policy_state_dict:
            policy_state_dict[name].copy_(base_policy_state_dict[name])
            
            name_parts = name.split('.')
            if 'unet' in name_parts and name_parts[1] == 'unet' and name_parts[2] != 'diffusion_step_encoder':
                #index = name_parts.index('unet')
                name_parts[2] = 'calibration_' + name_parts[2]
                calibration_name = '.'.join(name_parts)
                policy_state_dict[calibration_name].copy_(base_policy_state_dict[name])

                print(f'copy weight {name} to {calibration_name}')


    # 将模型 B 中与 A 共享的层冻结
    for name, param in policy.named_parameters():  # 对 model_b 模型调用 named_parameters 而不是状态字典
        if name in base_policy_state_dict:
            param.requires_grad = False

    # 打印哪些层被冻结
    for name, param in policy.named_parameters():
        print(f'{name}: freeze_requires_grad = {param.requires_grad}')

    policy.train()
    
    optimizer = torch.optim.Adam([param for name, param in policy.named_parameters() if param.requires_grad], lr=1e-4)
    


    '''
    set_seq_length：Observation sequence length
    action_seq_length：Action sequence length
    '''
    sample_num = -1
    dataset = Franka_Dataset_Memory(dataset_path,seq_length=cfg.set_seq_length, action_length = cfg.action_seq_length,sample_num = sample_num)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle= True, num_workers= 3,pin_memory=True)

    num_epochs = 500
    for epoch in range(num_epochs):
        for image, body, action, Seq_Trans_Center in dataloader:
            image = image.float().to(device)
            body = body.float().to(device)
            action = action.float().to(device)
            Seq_Trans_Center = Seq_Trans_Center.float().to(device)
            optimizer.zero_grad()

            loss = policy.forward(image, body, action, Seq_Trans_Center) * 1000

            loss.backward()
            optimizer.step()
    
        print("epoch {:d} loss: {:.3f}".format(epoch, loss))
        if (epoch + 1) % 100 == 0:
            #policy.save_pretrained('./')
            torch.save(policy.state_dict(), f'0410_{task_name}_control_2_diffusion_finetune_sample_{sample_num}_{epoch + 1}.pth')
