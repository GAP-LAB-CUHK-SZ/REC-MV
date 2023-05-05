import wandb
from engineer.visualizer.base_visualizer import Base_Visualizer
import datetime
from utils.common_utils import wandb_img,tensor2numpy, inverse_image_normalized, bgr2rgb, resize256, resize512
import pdb

class wandb_visualizer(Base_Visualizer):
    def __init__(self, project_name, exp_name ,resume = False):
        super(wandb_visualizer, self).__init__()
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        wandb.init(project=project_name, name= exp_name+'_'+cur_time, dir='./logs', resume = resume)

    def watch_model(self, model):
        '''
        logger model_parameters
        '''
        wandb.watch(model)
    def add_scalar(self, scalar_dict, step):
        wandb.log(scalar_dict, step)

    def add_image(self, tensor_dict, step, size = 256, rgb = True, normalized = False):
        if normalized:
            tensor_dict = inverse_image_normalized(tensor_dict)
        numpy_dict = tensor2numpy(tensor_dict)
        if rgb:
            numpy_dict = bgr2rgb(numpy_dict)
        if size == 256:
            numpy_dict = resize256(numpy_dict)
        elif size == 512:
            numpy_dict = resize512(numpy_dict)
        else:
            raise NotImplemented

        img_dict = wandb_img(numpy_dict)
        wandb.log(img_dict, step)



    @staticmethod
    def log_images_to_wandb(tensor_dict, column_names, step, size = 256, rgb = True, normalized = False, title=None):

        if title == None:
            title = "this is image table"

        if normalized:
            tensor_dict = inverse_image_normalized(tensor_dict)
        numpy_dict = tensor2numpy(tensor_dict)
        if rgb:
            numpy_dict = bgr2rgb(numpy_dict)
        if size == 256:
            numpy_dict = resize256(numpy_dict)
        elif size == 512:
            numpy_dict = resize512(numpy_dict)
        else:
            raise NotImplemented

        im_data = wandb_img(numpy_dict)

        outputs_table = wandb.Table(data=im_data, columns=column_names)
        wandb.log({f"{title}": outputs_table}, step = step)




