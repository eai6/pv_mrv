import numpy as np
from PIL import Image

# %tensorflow_version 1.x
import tensorflow.compat.v1 as tf

domain = 'tree_trunk'

path_to_model = '/Users/edwardamoah/Documents/GitHub/pv_mrv/dbh_estimation_algorithm_FastAPI/semantic_segmentation_model/tree_trunk_frozen_graph.pb'

#### load DeepLab Model #####
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513

  def __init__(self, file_handle):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None

    graph_def = tf.GraphDef()
    with tf.gfile.GFile(file_handle, 'rb') as fid:
      serialized_graph = fid.read()
      graph_def.ParseFromString(serialized_graph)

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.
    Args:
      image: A PIL.Image object, raw input image.
    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


dbh_MODEL = DeepLabModel(path_to_model) 

def create_tree_trunk_label_colormap():
  """Creates a label colormap for the locusts dataset.
  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((4, 3), dtype=int)
  colormap[0] = [0,0,0]
  colormap[1] = [255,255,255]
  colormap[2] = [0,85,0]
  colormap[3] = [255,150,100]
  return colormap

def label_to_color_image(label, domain):
  """
  Adds color defined by the dataset colormap to the label.
  Args:
    label: A 2D array with integer type, storing the segmentation label.
    domain: A string specifying which label map to use
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.
  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  elif domain == 'tree_trunk':
    colormap = create_tree_trunk_label_colormap()
  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def get_label_names(domain):
  if domain == 'tree_trunk': #dumby labels
    LABEL_NAMES = np.asarray([ "Unlabeled", "Background", "Tree trunk", "Tag"
    ])
  else:
    LABEL_NAMES = 'error'

  return LABEL_NAMES

label_names = get_label_names(domain)
FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP, domain)