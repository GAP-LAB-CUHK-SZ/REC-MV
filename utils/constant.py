import torch
FL_CONSTANT = {
    0 : 'neckline',
    1 : 'right_cuff',
    2 : 'left_cuff',
    3 : 'upper_waist',
    4 : 'lower_waist',
    5 : 'right_knee',
    6 : 'left_knee',
    7 : 'skirt_bottom',
}
FL_NAME=['neckline','right_cuff','left_cuff','upper_waist','lower_waist','right_knee','left_knee','skirt_bottom']
FL_COLOR = {
    'neck':(0, 0, 255),
    'right_cuff': (0, 255, 0),
    'left_cuff':(255, 0, 0),
    'left_pant': (127, 127, 0),
    'right_pant':(0, 127, 127),
    'upper_bottom': (127, 0, 127),
    'bottom_curve':(0, 127, 127),
}
FL_FLIP={
    'right_cuff':'left_cuff',
    'right_knee':'left_knee'
}
# index includes background
FL_CLASSES_FLIP = {
    2:3,
    6:7,
}

RAY_DIRS = torch.tensor([[0,0,1]]).float()
SMPL_DATA_DIR = '../SMPL/'
Z_RAY = torch.tensor([[0, 0, 1.]]).float()
FL_IDX = ['neck', 'right_cuff', 'left_cuff', 'bottom_curve']
TMP_FL_IDX = ['neck_line', 'right_cuff', 'left_cuff', 'upper_waist']

#SNUG CATEGORY SETTING
SNUG_MAP={
        'top00': 'bottom_curve',
        'top01': 'neck',
        'top02': 'right_cuff',
        'top03': 'left_cuff',
        }
RP4D_MAP={
        0: 'neck',
        1: 'right_cuff',
        2: 'left_cuff',
        3: 'bottom_curve',
        }

# using to initalize template garment
GARMENT_FL_MATCH={
        'long_sleeve_upper': ['neck', 'left_cuff', 'right_cuff', 'upper_bottom'],
        'long_pants': ['left_pant', 'right_pant', 'upper_bottom'],
        'short_pants': ['left_pant', 'right_pant', 'upper_bottom'],
        'short_sleeve_upper': ['neck', 'left_cuff', 'right_cuff', 'upper_bottom'],
        'dress': ['neck', 'left_cuff', 'right_cuff', 'bottom_curve'],
        'skirt': ['upper_bottom', 'bottom_curve'],
        'tube': ['neck', 'bottom_curve'],
        'no_sleeve_upper': ['neck', 'left_cuff', 'right_cuff', 'bottom_curve'],
        }
# using to feature line represent, NOTE that upper and bottom has same zone;
# where we need to
FL_EXTRACT = {
        'long_sleeve_upper': ['neck', 'left_cuff', 'right_cuff', 'upper_bottom'],
        'dress': ['neck', 'left_cuff', 'right_cuff', 'bottom_curve'],
        'long_pants': ['left_pant', 'right_pant'], # upper-bottom only use in upper garment
        'short_pants': ['left_pant', 'right_pant'], # upper-bottom only use in upper garment
        'short_sleeve_upper': ['neck', 'left_cuff', 'right_cuff', 'upper_bottom'],
        'tube': ['neck','bottom_curve'],
        'skirt': ['bottom_curve'],
        'no_sleeve_upper': ['neck', 'left_cuff', 'right_cuff', 'bottom_curve'],
}

WHOLE_BODY=[
    'long_pants',
    'long_sleeve_upper',
    ]

template_garment = {
        0:'long_pants',
        1:'long_sleeve_upper',
        2:'no_sleeve_upper',
        3:'short_sleeve_open_upper',
        4:'skirt',
        5:'long_sleeve_open_upper',
        6:'no_sleeve_open_upper',
        7:'short_pants',
        8:'short_sleeve_upper',
        }
TEMPLATE_GARMENT ={
        # 'dance':['long_pants', 'short_sleeve_upper']
        'dance':['short_sleeve_upper'],
        'anran':['short_sleeve_upper', 'skirt'],
        'xiaolin':['no_sleeve_upper'],
        'leyang':['short_sleeve_upper'],
        'tingting':['short_sleeve_upper'],
        # synthetic
        'female_outfit1': ['no_sleeve_upper'],
        'female_outfit3': ['tube'],
        'male_outfit1': ['long_sleeve_upper', 'short_pants'],
        'male_outfit2': ['long_sleeve_upper','long_pants'],

        # female large pose
        'anran_run': ['short_sleeve_upper', 'skirt'],
        'anran_tic': ['short_sleeve_upper', 'skirt'],
        'leyang_jump': ['dress'],
        'leyang_steps': ['dress'],
        'anran_dance': ['short_sleeve_upper', 'skirt'],
        'lingteng_dance': ['short_sleeve_upper', 'short_pants'],

        # people_snapshot_public
        'female-1-casual': ['short_sleeve_upper', 'long_pants'],
        # long pants is covered too much by long_sleeve_upper
        'female-3-casual': ['long_sleeve_upper', 'long_pants'],
        'female-3-sport': ['long_sleeve_upper', 'long_pants'],
        'female-4-casual': ['long_sleeve_upper', 'long_pants'],
        'female-4-sport': ['short_sleeve_upper', 'short_pants'],
        'female-6-plaza': ['long_sleeve_upper', 'long_pants'],
        'female-7-plaza': ['long_sleeve_upper', 'long_pants'],

        'male-1-casual': ['short_sleeve_upper', 'long_pants'],
        'male-1-plaza': ['short_sleeve_upper', 'long_pants'],
        'male-1-sport': ['short_sleeve_upper', 'short_pants'],
        'male-2-casual':['long_sleeve_upper', 'long_pants'],
        'male-2-outdoor': ['long_sleeve_upper', 'long_pants'],
        'male-4-casual': ['long_sleeve_upper', 'long_pants'],
        'male-5-outdoor': ['long_sleeve_upper', 'short_pants'],
        'male-9-plaza': ['long_sleeve_upper', 'long_pants'],
        }

FL_INFOS ={
        # 'dance':['long_pants', 'short_sleeve_upper']
        'dance':['short_sleeve_upper'],
        'anran':['neck', 'left_cuff','right_cuff','upper_bottom', 'bottom_curve'],
        'xiaolin':['neck', 'left_cuff', 'right_cuff', 'bottom_curve'],
        'leyang':['short_sleeve_upper'],
        'tingting':['short_sleeve_upper'],
        # synthetic
        'female_outfit1': ['neck','left_cuff','right_cuff','bottom_curve'],
        'female_outfit3': ['neck','bottom_curve'],
        'male_outfit1': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'male_outfit2': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],

        # large pose
        'anran_run': ['neck','left_cuff','right_cuff','upper_bottom','bottom_curve'],
        'anran_tic': ['neck','left_cuff','right_cuff','upper_bottom','bottom_curve'],
        'leyang_jump': ['neck','left_cuff','right_cuff','bottom_curve'],
        'leyang_steps': ['neck','left_cuff','right_cuff','bottom_curve'],
        'anran_dance': ['neck','left_cuff','right_cuff','upper_bottom','bottom_curve'],
        'lingteng_dance': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],

        # people_snapshot
        'female-3-casual': ['neck','left_cuff','right_cuff','upper_bottom', 'left_pant', 'right_pant'],
        'female-3-sport': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'female-4-casual': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'female-4-sport': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'female-6-plaza': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'female-7-plaza': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],

        'male-1-casual': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'male-1-sport': ['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'male-2-casual':['neck', 'left_cuff', 'right_cuff', 'upper_bottom', 'left_pant', 'right_pant'],
        'male-2-outdoor':['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'male-4-casual':['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'male-5-outdoor':['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
        'male-9-plaza':['neck','left_cuff','right_cuff','upper_bottom','left_pant', 'right_pant'],
    }
PANTS_GARMENT=[
        'long_pants',
        'no_sleeve_upper',
        'long_skirt',
        'short_pants',
        'long_sleeve_dress',
        'short_sleeve_dress',
        'long_sleeve_upper',
        'short_sleeve_upper',
        'no_sleeve_dress',
        'skirt',
]
GARMENT_COLOR_MAP={
        "short_sleeve_upper": dict(back_ground = [125,125,125], left_cuff=[131, 149, 69], right_cuff=[185, 82, 185], upper_bottom = [211, 200, 42], neck=[250, 15, 16]),
        "long_pants": dict(back_ground = [125,125,125], left_pant=[42, 211, 141], right_pant = [67, 42, 211], upper_bottom = [211, 200, 42]),
        "short_pants": dict(back_ground = [125,125,125], left_pant=[42, 211, 141], right_pant = [67, 42, 211], upper_bottom = [211, 200, 42]),
        "long_sleeve_upper": dict(back_ground = [125,125,125], left_cuff=[131, 149, 69], right_cuff=[185, 82, 185], upper_bottom = [211, 200, 42], neck=[250, 15, 16]),
        "skirt": dict(back_ground = [125,125,125], bottom_curve= [155, 126, 151], upper_bottom = [211, 200, 42]),
        "tube": dict(back_ground = [125,125,125], bottom_curve= [155, 126, 151], neck = [211, 200, 42]),
        "no_sleeve_upper": dict(back_ground = [125,125,125], left_cuff=[131, 149, 69], right_cuff=[185, 82, 185], bottom_curve = [211, 200, 42], neck=[250, 15, 16]),
        "dress": dict(back_ground = [125,125,125], left_cuff=[131, 149, 69], right_cuff=[185, 82, 185], bottom_curve = [211, 200, 42], neck=[250, 15, 16]),
 }
#   'atr': {
#        'input_size': [512, 512],
#        'num_classes': 18,
#        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
#                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
#    },

ATR_PARSING = {
    # with head and hand
    'upper':[1, 2, 3, 4,  11, 16, 17, 14, 15],
    #without head
    #  'upper':[1, 2, 3, 4, 11, 16, 17],
    'bottom':[5,6,8],
    'upper_bottom':[1, 2, 3, 4, 5, 7, 8,11, 16, 17, 14, 15,6]
    # w/o hand
    # 'upper_bottom':[4, 5, 7, 16, 17]
}

FL_COLOR = {
    'neck':(0, 0, 255),
    'right_cuff': (0, 255, 0),
    'left_cuff':(255, 0, 0),
    'left_pant': (127, 127, 0),
    'right_pant':(0, 127, 127),
    'upper_bottom': (127, 0, 127),
    'bottom_curve':(0, 127, 127),
}
ZBUF_THRESHOLD={
    'neck': 0.1,
    'right_cuff': 0.05,
    'left_cuff': 0.05,
    'left_pant': 0.05,
    'right_pant': 0.05,
    'upper_bottom': 0.08,
    'bottom_curve': 0.1
        }
CURVE_AWARE={
       'female_outfit1': 'bottom_curve',
       'female_outfit3': 'bottom_curve',
       'anran_dance': 'bottom_curve',
        }


# NOTE that for tube category, neck need to set 2.
INI_FL_SCALE = {
    'neck':1.5,
    'right_cuff': 1.5,
    'left_cuff':1.5,
    'left_pant': 1.5,
    'right_pant':1.5,
    'upper_bottom': 2.,
    'bottom_curve': 2.,
}

SMOOTH_TRANS= {
        'anran':[[116,150], [269,309]],
        'lingteng_dance':[[34,41]],
        'xiaolin':[[]],
        'anran_tic':[[]],
        'anran_run':[[]],
        'leyang_jump':[[]],
        }

RENDER_COLORS={
        'anran':[[255,255, 0 ], [170, 170, 255]],
        'lingteng_dance':[[170,170,127], [72,152,170]],
        'xiaolin':[[193, 210, 240]],
        'anran_tic':[[255, 99, 128],[193, 210, 240]],
        'anran_run':[[255, 99, 128],[193, 210, 240]],
        'leyang_jump':[[193, 210, 240]],
        'female-3-casual':[[255, 99, 128],[193, 210, 240]],
        }

