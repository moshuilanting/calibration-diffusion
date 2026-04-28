# from model.class_free_diffusion.configuration_diffusion import DiffusionConfig
# from model.class_free_diffusion.modeling_diffusion import ClassFreeDiffusionPolicy

from model.diffusion.configuration_diffusion import DiffusionConfig
from model.diffusion.modeling_diffusion import DiffusionPolicy
import torch
from dataset import Franka_Dataset_Memory
from torch.optim.lr_scheduler import StepLR

if __name__=="__main__":
    batch_size = 80
    device = torch.device("cuda")
    cfg = DiffusionConfig()
    
    policy = DiffusionPolicy(cfg)
    #policy.load_state_dict(torch.load('0303_normal_diffusion_fruits_400.pth',  map_location=device, weights_only=True))
    
    policy.train()
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    #scheduler = StepLR(optimizer, step_size=800, gamma=0.5)

    task_name = "Franka_Arrange_Bottle"
    dataset_path = "/media/jtl/ZJRR8/Free_View_DataSet" + task_name

    
    '''
    set_seq_length：Observation sequence length
    action_seq_length：Action sequence length
    '''
    sample_num = 300
    dataset = Franka_Dataset_Memory(dataset_path,seq_length=cfg.set_seq_length, action_length = cfg.action_seq_length, sample_num = sample_num)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle= True, num_workers= 3, pin_memory=True)


    num_epochs = 1200
    for epoch in range(num_epochs):
        for image, body, action in dataloader:
            image = image.float().to(device)
            body = body.float().to(device)
            action = action.float().to(device)

            #Seq_Trans_Center = Seq_Trans_Center.float().to(device)
            optimizer.zero_grad()
            #print("action: ", action[:,8:])
            loss = policy.forward(image, body, action) * 1000

            loss.backward()
            optimizer.step()
    
        print("epoch {:d} , loss: {:.3f}".format(epoch, loss))
        #scheduler.step()
        if (epoch + 1) % 400 == 0 :
            #policy.save_pretrained('./')
            torch.save(policy.state_dict(), f'0410_{task_name}_diffusion_sample_{sample_num}_{epoch + 1}.pth')
