import numpy as np
import os
import sys
import argparse
import os.path as osp
import glob
import trimesh
sys.path.append('./')
path ={
        'female1': './selfrecon_sythe/female_outfit1/female-1-diffused-skinning',
        'female3': './selfrecon_sythe/female_outfit3/female-3-diffused-skinning',
        'male1':   './selfrecon_sythe/male_outfit1/male-1-diffused-skinning-3',
        'male2':'./selfrecon_sythe/male_outfit2/male-2-diffused-skinning',
        }
def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', default='compute', choices=['male1', 'male2', 'female1',
        'female3'])


    args = parser.parse_args()
    return  args

def main():
    args = get_parse()
    subj = path[args.input]



    meshs_path = sorted(glob.glob(osp.join(subj,'meshs/*.obj')))


    valid = 0

    dis = 0

    for i in range(1, len(meshs_path)-1):
        a = trimesh.load(meshs_path[i-1],process =False)
        b = trimesh.load(meshs_path[i],process = False)
        c = trimesh.load(meshs_path[i+1],process = False)

        if b.vertices.shape[0] != c.vertices.shape[0] or b.vertices.shape[0] != a.vertices.shape[0]:
            continue




        ba = b.vertices - a.vertices
        cb = c.vertices - b.vertices




        dis += np.sqrt(((ba- cb)**2).sum(-1)).sum() / ba.shape[0]
        valid += 1




        print(dis/valid)


    print(subj)
    print(dis/valid)













if __name__ == '__main__':
    main()








