import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from multiview_detector.utils import projection
from multiview_detector.datasets.Wildtrack import Wildtrack
import kornia

#added function 
def get_worldgrid2imagecoord_matrices(intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    
    projection_matrices = {}
    #for cam in range(self.num_cam):
    # step 1 # convert  world grid to world coordinates (# Given augument)
    # step 2 # convert  world coordinates to image coordinates
    worldcoord2imgcoord_mat = intrinsic_matrices @ np.delete(extrinsic_matrices, 2, 1)
    # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
    # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
    worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
    #permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    permutation_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    projection_matrices = permutation_mat @ worldgrid2imgcoord_mat
    print(projection_matrices)
    return projection_matrices 


if __name__ == '__main__':
    img = Image.open('/home/zhiming/Data/Wildtrack/Image_subsets/C1/00000000.png')
    #img = Image.open('/home/zhiming/Data/Wildtrack/Image_subsets/C2/00000000.png')
    #img = Image.open('/home/zhiming/Data/Wildtrack/Image_subsets/C7/00000000.png')

    #print(img.size) # w 1920 h 1280
    dataset = Wildtrack('/home/zhiming/Data/Wildtrack')
    xi = np.arange(0, 480, 1) 
    yi = np.arange(0, 1440, 1) 
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
 
    
    world_coord = dataset.get_worldcoord_from_worldgrid(world_grid) 
    #print(world_coord.shape)
    img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[0],
                                                          dataset.extrinsic_matrices[0])
    #print(img_coord[0], img_coord[1])
    #img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[1],
    #                                                      dataset.extrinsic_matrices[1])
    #img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[2],
    #                                                      dataset.extrinsic_matrices[2])
    
    img_coord = img_coord[:, np.where((img_coord[0] >= 0) & (img_coord[1] >= 0) &
                                     (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]

    plt.imshow(img)
    
    img_coord = img_coord.astype(int).transpose()
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #print(img.shape[0])
    for point in img_coord:
        cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img)
    plt.show()
    plt.savefig("view1_grid2img.png")
    
    

    #worldgrid2imgcoord_matrices =  get_worldgrid2imagecoord_matrices(dataset.intrinsic_matrices[0],
    #                                                                dataset.extrinsic_matrices[0],
    #                                                                dataset.worldgrid2worldcoord_mat)
    #print(worldgrid2imgcoord_matrices.shape)
    
    #world_coord = np.concatenate([gt_world_img0, np.ones([1, gt_world_img0.shape[1]])], axis=0)
    #print(world_coord.shape)
    #img_coord_gt_test = worldgrid2imgcoord_matrices @ world_coord 
    #img_coord_gt_test = img_coord_gt_test[:2, :] / img_coord_gt_test[2, :]
    #print(img_coord_gt_test[0], img_coord_gt[0])
    #pass

    #img_coord_gt = img_coord_gt[:, np.where((img_coord_gt[0] > 0) & (img_coord_gt[1] > 0) &
    #                                  (img_coord_gt[0] < 1920) & (img_coord_gt[1] < 1080))[0]]
     
    #img_coord_gt = img_coord_gt.astype(int).transpose() 
    #for point in img_coord_gt:
    #    cv2.circle(img, tuple(point.astype(int)), 5, (255, 0, 0), -1)
        


    #img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #img.save('img_grid_visualize_img0_view1_test.png')
    #plt.imshow(img)
    #plt.show()
    pass
