import glob
import numpy as np
import os
import os.path as osp
import argparse
import sys
sys.path.append('./')
import shutil



def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path',type=str , help='dataset')
    parser.add_argument('--frame', default='25', type= float, help='select model')


    args = parser.parse_args()
    return args

def main(args):
    input_path = args.input_path
    frame = args.frame
    frame_diff = float(frame / 30)
    assert frame_diff<=1.

    featureline_path = osp.join(input_path, 'featurelines')
    featurelines = glob.glob(osp.join(featureline_path, '*.json'))


    for featureline in sorted(featurelines):

        name = featureline.split('/')[-1]
        cur_frame = float(name.replace('.json', ''))
        target_frame = int(cur_frame * frame_diff)
        target_file = os.path.join(featureline_path, "{:06d}.json".format(target_frame))

        shutil.move(featureline, target_file)
        print('{} -> {}'.format(featureline, target_file))









if __name__ == '__main__':
    args = get_parse()
    main(args)


