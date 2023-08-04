# This is the tutorial of data processing of REC-MV.

The data pre-processing part includes img, mask, normal, parsing (garment segmentation), camera, smpl parameters (beta & theta), featurelines, skinning weight.

## Step1
You should make directory to save all processed data, named, to say, xiaoming.
And you turn the video into images:
```
encodepngffmpeg()
{
	# $1: target folder
	# $2: save video name
    ffmpeg -r ${1} -pattern_type glob -i '*.png' -vcodec libx264 -crf 18 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${2}
}


encodepngffmpeg 30 ./xiaoming.mp4
```
Then, your data directory:
```
xiaoming
--imgs
```
## Step2 Normal, Parsing, and Mask
Get the normal map, parsing mask, masks.
```
python prcess_data_all.py --gid <gpu_id> --root <Your data root> --gender <data gender>
# example
python prcess_data_all.py --gid 0 --root /data/xiaoming --gender male 
```

## Step3 SMPL & Camera
To get smpl paramaters (pose and shape), here we use [videoavatar](https://github.com/thmoa/videoavatars):
- Set up the env (**Note it use python2**)
- Prepare keypoints files for each frame in the video and put them under `xiaoming/keypoints`, which I use [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
- Run three python files in videoavatars/prepare_data, you'll get `keypoints.hdf5, masks.hdf5, camera.hdf5.` Or you can just use my script: ```python get_reconstructed_poses.py --root xiaoming --out xiaoming --gender male```
- `bash run_step1.sh`

After you run through videoavatar, you will get `camera.pkl, reconstructed_poses.hdf5`. Put it also under the root(xiaoming).

You can get `smpl_rec.npz, camera.npz` by running:
```
python get_smpl_rec_camera.py --root xiaoming --save_root xiaoming --gender male
```

**Note: You can use any other smpl estimation algorithm, but you should follow the way how smpl_rec.npz save pose, shape, and trans.**

## Step4 Skining Weight
We follow [fite](https://github.com/jsnln/fite) to get the lbs skinning weight to prevent artifacts.

 In fite's readme, you'll get a skining weight cube after finishing 3.Diffused Skinning. Name it `diffused_skinning_weights.npy` and put it under xiaoming.
