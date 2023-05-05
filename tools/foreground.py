import cv2
import numpy as np
import os
path ='./female_large_pose_process_new/anran_dance/imgs/000050.jpg'
img = cv2.imread(path).astype(np.float)/255

parsing_mask = np.load('./female_large_pose_process_new/anran_dance/parsing_SCH_ATR/mask_parsing_000050.npy')
upper_idx = [1, 2, 3, 4,5,8,  11, 16, 17, 14, 15]

foreground = np.zeros(img.shape).astype(np.bool)



for idx in upper_idx:
    foreground |= (parsing_mask == 20)[..., None]

foreground = foreground.astype(np.float)
for ratio in range(1,10):
    alpha = ratio/10.
    pre_img = foreground * img + (1-foreground) * (img * alpha + 1 *(1-alpha))
    pre_img = (255*pre_img).astype(np.uint8)

    print('./pipeline_graph_curve/{:.4f}.png'.format(alpha))
    cv2.imwrite('./pipeline_graph_curve/{:.4f}.png'.format(alpha), pre_img)














