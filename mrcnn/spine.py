Simport os
import sys
import json
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# To find local version of the library
sys.path.append(ROOT_DIR) 
# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

############################################################
#  Configurations
############################################################

# define our module.
class SpineConfig(Config):
    
    """spine config for training."""

    NAME = "spine_segmentation"

    NUM_CLASSES = 1 + 7

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    STEPS_PER_EPOCH = 500
    TRAIN_ROIS_PER_IMAGE = 512

############################################################
#  Dataset
############################################################

class SpineDataset(utils.Dataset):

    # load dataset and define a class.
    def load_dataset(self, dataset_dir):
        self.add_class('dataset', 1, 'catheter')
        self.add_class('dataset', 2, 'guidewire')
        self.add_class('dataset', 3, 'balloon')
        self.add_class('dataset', 4, 'valve')
        self.add_class('dataset', 5, 'spine')
        self.add_class('dataset', 6, 'vertebra')
        self.add_class('dataset', 7, 'intervertebra')

        # find all images
        for i, filename in enumerate(os.listdir(dataset_dir)):
            if '.jpg' in filename:
                self.add_image('dataset', 
                               image_id=i, 
                               path=os.path.join(dataset_dir, filename), 
                               annotation=os.path.join(dataset_dir, filename.replace('.jpg', '.json'))) 
                
    # extract masks from annotation file.            
    def extract_masks(self, filename):
        json_file = os.path.join(filename)
        with open(json_file) as f:
            img_anns = json.load(f)
            
        masks = np.zeros([1024, 1024, len(img_anns['shapes'])], dtype='uint8')
        classes = []
        for i, anno in enumerate(img_anns['shapes']):
            mask = np.zeros([1024, 1024], dtype=np.uint8)
            cv2.fillPoly(mask, np.array([anno['points']], dtype=np.int32), 1)
            masks[:, :, i] = mask
            classes.append(self.class_names.index(anno['label']))
        return masks, classes
 
    # load the masks for an image.
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        masks, classes = self.extract_masks(path)
        return masks, np.asarray(classes, dtype='int32')
    
    # load image reference.
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']