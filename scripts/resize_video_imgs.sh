# resize_imgs(input_imgs, input_mask, joints, size = 1080, save_path, visualized = True):
# anran_run, anran_tic, leyang_jump, leyang_steps
python ./tools/resize_video_imgs.py ./female_large_pose/$1/imgs ./female_large_pose/$1/masks ./female_large_pose/$1/joints2d/2D_pose.json --save_path ./female_large_pose_process/$1
