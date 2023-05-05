import argparse
import os
import glob
import sys
def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', default='model_path', help='help to generate visualized results')


    args = parser.parse_args()
    return args

def main(args):

    path = args.path


    result_path =  os.path.join(path, 'results')
    os.makedirs(result_path , exist_ok= True)


    colors_path = os.path.join(path, 'colors')
    meshs_path = os.path.join(path, 'meshs')


    os.system('cd {}'.format(colors_path))
    os.system('encodepngffmpeg 30 ../results/animation/colors.mp4')

    # os.system('cd ${}'.format(meshs_path))


    xxxx






    os.system('cd ')


if __name__ == '__main__':

    args = get_parse()

    main(args)

