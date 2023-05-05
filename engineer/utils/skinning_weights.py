import numpy as np



def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')
    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors
