import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        ## needs modification (added function) #
        worldgrid2imgcoord_matrices = self.get_worldgrid2imagecoord_matrices(dataset.base.intrinsic_matrices,
                                                                    dataset.base.extrinsic_matrices,
                                                                    dataset.base.worldgrid2worldcoord_mat)

        # modification ends #

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape)) # [1080, 1920] / 4 = [270, 480]  (h, w)
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)  #img_reduce=4
        img_zoom_mat = np.diag(np.append(img_reduce, [1])) #[[4, 0, 0], [0, 4, 0], [0, 0, 1]]
        # map

        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))   #[[1/4, 0, 0], [0, 1/4, 0], [0, 0, 1]]
        # projection matrices: img feat -> map feat

        # needs modification #
        #self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          #for cam in range(self.num_cam)]
        
        #self.proj_mats = [torch.from_numpy(img_zoom_mat @ worldgrid2imgcoord_matrices[cam] @ map_zoom_mat)
        #                  for cam in range(self.num_cam)]
        
        #self.proj_mats = [torch.from_numpy(map_zoom_mat @ worldgrid2imgcoord_matrices[cam] @ img_zoom_mat)
        #                  for cam in range(self.num_cam)]
        
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ worldgrid2imgcoord_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        
        #self.proj_mats = [torch.from_numpy(worldgrid2imgcoord_matrices[cam])
        #                  for cam in range(self.num_cam)]

        # modification ends#

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            self.out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:1')
            self.base_pt2 = base[split:].to('cuda:0')
            self.out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        
        # need modification (done)# 
        #self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
        #                                    nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                    nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        
        #self.map_classifier = #nn.Sequential(nn.Conv2d(self.out_channel*self.num_cam , 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
        #self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 512, 3, padding=1), nn.ReLU(),             
        #                                    nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                    nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        self.map_classifier = nn.Sequential(nn.Conv2d(self.out_channel, 512, 3, padding=1), nn.ReLU(),             
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        # modification ends
        pass

    def forward(self, imgs, visualize=False):
        
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        imgs_result = []
        #world_features = [] 
        world_feature = torch.zeros([B, self.out_channel, self.reducedgrid_shape[0]*self.reducedgrid_shape[1]]).to("cuda:0")  # B C H W # modified 
        
        # create gound grid 
        xi = np.arange(0, self.reducedgrid_shape[0], 1) # 120 
        yi = np.arange(0, self.reducedgrid_shape[1], 1) # 360
         
        world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
        #world_feature = kornia.geometry.transform.warp_perspective(img_feature.to('cuda:0'), proj_mat, self.reducedgrid_shape)
        world_grid = torch.from_numpy(np.concatenate([world_grid, np.ones([1, world_grid.shape[1]])], axis=0)).float().to("cuda:0")
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:1'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')   # upsampling to 270 h, 480 w 
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            imgs_result.append(img_res)

            # need modification # 
           
            #proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            proj_mat = self.proj_mats[cam].repeat([1, 1]).float().to('cuda:0') 

            #map world grid back to image coordinates 
            img_coord = proj_mat @ world_grid 
            img_coord = img_coord[:2, :]/img_coord[2, :] # [2, 43200]
            img_coord = img_coord.long() # convert float coordinates into integers

            #create mask for x, y coordinates  
            mask_x = torch.zeros_like(img_coord[0]) #[43200]    
            mask_y = torch.zeros_like(img_coord[1]) #[43200]
            mask_x[(img_coord[0, :] >= 0) & (img_coord[0, :] < img_feature.shape[-1])]=1  # 0<=x<=480
            mask_y[(img_coord[1, :] >= 0) & (img_coord[1, :] < img_feature.shape[-2])]=1  # 0<=x<=270
            # create a mask which meets both x and y condition
            mask_xy = mask_x * mask_y # [43200]

            # extract x and y separately
            img_coord_1d_x = img_coord[0, (mask_xy>0)]
            img_coord_1d_y = img_coord[1, (mask_xy>0)]

            # the coodinates projected onto 1d feature maps should be  y*480 + x
            img_coord_1d = img_coord_1d_y * img_feature.shape[-1] + img_coord_1d_x # 33713

            img_feature_cp = img_feature.clone().view(B, self.out_channel, -1)
            ground_map = img_feature_cp[:, :, img_coord_1d] 
            #world_feature = torch.zeros([B, self.out_channel, self.reducedgrid_shape[0]*self.reducedgrid_shape[1]]).to("cuda:0")  # B C H W # modified 
            world_feature[:, :, torch.argwhere(mask_xy>0).squeeze()] += ground_map[:, :, :]

            ''' # for verification
            img_coord = img_coord.long().view(2, 120, 360)
            count = 0
            world_feature = torch.zeros([B, self.out_channel, self.reducedgrid_shape[0], self.reducedgrid_shape[1]]).to("cuda:0")  # B C H W # modified 
            for i in range(img_coord.shape[1]):
                for j in range(img_coord.shape[2]):
                    if (img_coord[0, i, j] >= 0) & (img_coord[0, i, j] < 480*4) and (img_coord[1, i, j] >= 0) & (img_coord[1, i, j] < 270*4) :
                        world_feature[:, :, i, j] = img_feature[:, :, img_coord[1, i, j]//4, img_coord[0, i, j]//4]
                        count += 1
            print(count) # 33713
            '''
            if visualize:
            #if True:
                #print("hello")
                if cam == 1:
                    world_feature = world_feature.view(B, self.out_channel, self.reducedgrid_shape[0], self.reducedgrid_shape[1])
                    plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                    plt.show()
                    plt.savefig("img_fea_view"+str(cam+1)+".png")
                    plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                    plt.show()
                    plt.savefig("world_fea_view"+str(cam+1)+".png")
                    
                    plt.imshow(imgs[0,cam].detach().permute(1,2,0).cpu().numpy())
                    plt.show()
                    plt.savefig("org_img"+str(cam+1)+".png")
                    exit()
            #world_features.append(world_feature.reshape(B,  self.out_channel, self.reducedgrid_shape[0],self.reducedgrid_shape[1]).to('cuda:0'))
        #world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        # modification ends #
        
        if visualize:
            plt.imshow(torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            plt.show()
        #exit()
        #map_result = self.map_classifier(world_features.to('cuda:0'))
        world_feature = world_feature.reshape(B, self.out_channel, self.reducedgrid_shape[0], self.reducedgrid_shape[1])/self.num_cam
        map_result = self.map_classifier(world_feature.to("cuda:0"))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        
        #print("done")
        #exit()
        if visualize:
            plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            plt.show()
        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices
    
    #added function 
    def get_worldgrid2imagecoord_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            # step 1 # convert  world grid to world coordinates (# Given augument)
            # step 2 # convert  world coordinates to image coordinates
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            #permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            #projection_matrices[cam] = permutation_mat @ worldgrid2imgcoord_mat
            projection_matrices[cam] = worldgrid2imgcoord_mat
        return projection_matrices 

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    model = PerspTransDetector(dataset)
    map_res, img_res = model(imgs, visualize=True)
    pass


if __name__ == '__main__':
    test()
