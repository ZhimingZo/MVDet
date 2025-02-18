import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from multiview_detector.utils import projection
from multiview_detector.datasets.Wildtrack import Wildtrack

if __name__ == '__main__':
    #img = Image.open('/home/zhiming/Data/Wildtrack/Image_subsets/C1/00000000.png')
    #img = Image.open('/home/zhiming/Data/Wildtrack/Image_subsets/C2/00000000.png')
    img = Image.open('/home/zhiming/Data/Wildtrack/Image_subsets/C3/00000000.png')
    dataset = Wildtrack('/home/zhiming/Data/Wildtrack')
    xi = np.arange(0, 480+40, 40)
    yi = np.arange(0, 1440+40, 40)
    world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
    world_coord = dataset.get_worldcoord_from_worldgrid(world_grid)

    #img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[0],
    #                                                      dataset.extrinsic_matrices[0])
    #img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[1],
    #                                                      dataset.extrinsic_matrices[1])
    img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[2],
                                                          dataset.extrinsic_matrices[2])
    img_coord = img_coord[:, np.where((img_coord[0] > 0) & (img_coord[1] > 0) &
                                      (img_coord[0] < 1920) & (img_coord[1] < 1080))[0]]
    
    plt.imshow(img)
    plt.show()
    img_coord = img_coord.astype(int).transpose()
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #for point in img_coord:
    #    cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
    #img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    ## map gt_grid to image coor ## 
    gt_world_img0 = np.load("img0_world_coor.npy") 
    
    world_coord_gt_img0 = dataset.get_worldcoord_from_worldgrid(gt_world_img0)
    
    #img_coord_gt = projection.get_imagecoord_from_worldcoord(world_coord_gt_img0, dataset.intrinsic_matrices[0],
    #                                                      dataset.extrinsic_matrices[0])
    #img_coord_gt = projection.get_imagecoord_from_worldcoord(world_coord_gt_img0, dataset.intrinsic_matrices[1],
    #                                                      dataset.extrinsic_matrices[1])
    img_coord_gt = projection.get_imagecoord_from_worldcoord(world_coord_gt_img0, dataset.intrinsic_matrices[2],
                                                          dataset.extrinsic_matrices[2])
    


    img_coord_gt = img_coord_gt[:, np.where((img_coord_gt[0] > 0) & (img_coord_gt[1] > 0) &
                                      (img_coord_gt[0] < 1920) & (img_coord_gt[1] < 1080))[0]]
     
    img_coord_gt = img_coord_gt.astype(int).transpose() 
    for point in img_coord_gt:
        cv2.circle(img, tuple(point.astype(int)), 5, (255, 0, 0), -1)
        


    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img.save('img_grid_visualize_img0_view3_2.png')
    plt.imshow(img)
    plt.show()
    pass
