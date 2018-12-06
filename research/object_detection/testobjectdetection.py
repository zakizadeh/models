{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "V8-yl-s-WKMG"
   },
   "source": [
    "# Object Detection Demo\n",
    "Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "kFSqkTCdWKMI"
   },
   "source": [
    "# Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "hV4P5gyTWKMI"
   },
   "outputs": [],
   "source": [
    "import collections\n",
    "import numpy as np\n",
    "import os\n",
    "import six.moves.urllib as urllib\n",
    "import sys\n",
    "import tarfile\n",
    "import tensorflow as tf\n",
    "import zipfile\n",
    "import cv2\n",
    "from distutils.version import StrictVersion\n",
    "from collections import defaultdict\n",
    "from io import StringIO\n",
    "from matplotlib import pyplot as plt\n",
    "from PIL import Image\n",
    "\n",
    "\n",
    "# This is needed since the notebook is stored in the object_detection folder.\n",
    "#sys.path.append(\"..\")\n",
    "from object_detection.utils import ops as utils_ops\n",
    "\n",
    "if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):\n",
    "  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Wy72mWwAWKMK"
   },
   "source": [
    "## Env setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(\"video.mp4\")\n",
    "STANDARD_COLORS = [\n",
    "    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',\n",
    "    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',\n",
    "    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',\n",
    "    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',\n",
    "    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',\n",
    "    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',\n",
    "    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',\n",
    "    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',\n",
    "    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',\n",
    "    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',\n",
    "    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',\n",
    "    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',\n",
    "    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',\n",
    "    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',\n",
    "    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',\n",
    "    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',\n",
    "    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',\n",
    "    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',\n",
    "    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',\n",
    "    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',\n",
    "    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',\n",
    "    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',\n",
    "    'WhiteSmoke', 'Yellow', 'YellowGreen'\n",
    "]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "v7m_NY_aWKMK"
   },
   "outputs": [],
   "source": [
    "# This is needed to display the images.\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "r5FNuiRPWKMN"
   },
   "source": [
    "## Object detection imports\n",
    "Here are the imports from the object detection module."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "bm0_uNRnWKMN"
   },
   "outputs": [],
   "source": [
    "from utils import label_map_util\n",
    "\n",
    "from utils import visualization_utils as vis_util"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "cfn_tRFOWKMO"
   },
   "source": [
    "# Model preparation "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "X_sEBLpVWKMQ"
   },
   "source": [
    "## Variables\n",
    "\n",
    "Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  \n",
    "\n",
    "By default we use an \"SSD with Mobilenet\" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "VyPz_t8WWKMQ"
   },
   "outputs": [],
   "source": [
    "# What model to download.\n",
    "CWD_PATH = os.getcwd()\n",
    "MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'\n",
    "PATH_TO_CKPT = os.path.join(CWD_PATH, 'ssd_mobilenet_v1_coco_11_06_2017', 'frozen_inference_graph.pb')\n",
    "\n",
    "# List of the strings that is used to add correct label for each box.\n",
    "PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')\n",
    "\n",
    "NUM_CLASSES = 90\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "7ai8pLZZWKMS"
   },
   "source": [
    "## Download Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "KILYnwR5WKMS"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "YBcB9QHLWKMU"
   },
   "source": [
    "## Load a (frozen) Tensorflow model into memory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "KezjCRVvWKMV"
   },
   "outputs": [],
   "source": [
    "detection_graph = tf.Graph()\n",
    "with detection_graph.as_default():\n",
    "  od_graph_def = tf.GraphDef()\n",
    "  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:\n",
    "    serialized_graph = fid.read()\n",
    "    od_graph_def.ParseFromString(serialized_graph)\n",
    "    tf.import_graph_def(od_graph_def, name='')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "_1MVVTcLWKMW"
   },
   "source": [
    "## Loading label map\n",
    "Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "hDbpHkiWWKMX"
   },
   "outputs": [],
   "source": [
    "category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "EFsoUHvbWKMZ"
   },
   "source": [
    "## Helper code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "aSlYc3JkWKMa"
   },
   "outputs": [],
   "source": [
    "def load_image_into_numpy_array(image):\n",
    "  (im_width, im_height) = image.size\n",
    "  return np.array(image.getdata()).reshape(\n",
    "      (im_height, im_width, 3)).astype(np.uint8)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "H0_1AGhrWKMc"
   },
   "source": [
    "# Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "jG-zn5ykWKMd"
   },
   "outputs": [],
   "source": [
    "# For the sake of simplicity we will use only 2 images:\n",
    "# image1.jpg\n",
    "# image2.jpg\n",
    "# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.\n",
    "PATH_TO_TEST_IMAGES_DIR = 'test_images'\n",
    "TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]\n",
    "\n",
    "# Size, in inches, of the output images.\n",
    "IMAGE_SIZE = (12, 8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "92BHxzcNWKMf"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "person\n",
      "(138.46996307373047, 174.06941413879395, 188.36994171142578, 414.9798583984375)\n",
      "person\n",
      "(140.56546211242676, 177.2208309173584, 190.48442840576172, 429.7618103027344)\n",
      "person\n",
      "(139.54633712768555, 177.3940086364746, 191.27134323120117, 431.29661560058594)\n",
      "person\n",
      "(138.1285572052002, 177.7906036376953, 189.77157592773438, 431.36329650878906)\n",
      "person\n",
      "(137.60026931762695, 177.21871376037598, 189.06538009643555, 430.9849548339844)\n",
      "person\n",
      "(134.67902183532715, 177.75229454040527, 181.5566635131836, 423.2417297363281)\n",
      "person\n",
      "(134.85380172729492, 179.6121597290039, 180.21339416503906, 426.24515533447266)\n",
      "person\n",
      "(134.75769996643066, 186.7038917541504, 171.92852020263672, 432.70069122314453)\n",
      "person\n",
      "(133.75285148620605, 190.76854705810547, 167.00653076171875, 438.6048889160156)\n",
      "person\n",
      "(134.30437088012695, 191.24825477600098, 170.88878631591797, 436.81087493896484)\n",
      "person\n",
      "(134.60482120513916, 190.54683208465576, 176.95823669433594, 436.2327575683594)\n",
      "person\n",
      "(133.4076976776123, 192.15365409851074, 167.59143829345703, 439.2615509033203)\n",
      "person\n",
      "(134.58848476409912, 192.79468059539795, 173.78850936889648, 439.6599578857422)\n",
      "person\n",
      "(134.49151039123535, 193.21964263916016, 176.41212463378906, 438.8939666748047)\n",
      "person\n",
      "(135.27182579040527, 197.60713577270508, 182.0631217956543, 437.56683349609375)\n",
      "person\n",
      "(136.55067443847656, 196.42916679382324, 183.03844451904297, 438.02478790283203)\n",
      "person\n",
      "(136.84171199798584, 196.06820583343506, 186.16756439208984, 441.00086212158203)\n",
      "person\n",
      "(136.7709445953369, 195.72060585021973, 184.5669174194336, 442.0319366455078)\n",
      "person\n",
      "(139.1635036468506, 194.42742347717285, 191.22608184814453, 438.58123779296875)\n",
      "person\n",
      "(145.1575469970703, 194.48555946350098, 192.59849548339844, 437.8240203857422)\n",
      "person\n",
      "(147.16845989227295, 194.05770778656006, 197.93880462646484, 439.09366607666016)\n",
      "person\n",
      "(148.0147647857666, 194.2562484741211, 193.5601806640625, 440.3181457519531)\n",
      "person\n",
      "(149.51565742492676, 194.32053565979004, 195.38330078125, 438.7324905395508)\n",
      "person\n",
      "(152.33263492584229, 196.3723611831665, 190.0066375732422, 446.7324447631836)\n",
      "person\n",
      "(158.68144512176514, 204.66397762298584, 195.78372955322266, 471.84757232666016)\n",
      "person\n",
      "(158.8186740875244, 205.27459144592285, 194.6923828125, 472.02056884765625)\n",
      "person\n",
      "(159.23182010650635, 206.55322551727295, 193.70635986328125, 473.4447479248047)\n",
      "person\n",
      "(159.04250621795654, 207.24027156829834, 190.2798080444336, 475.2030563354492)\n",
      "person\n",
      "(158.65898609161377, 208.1819200515747, 190.04364013671875, 475.4088592529297)\n",
      "person\n",
      "(158.96377086639404, 208.33670139312744, 191.4023208618164, 476.0617446899414)\n",
      "person\n",
      "(158.64446640014648, 208.92648696899414, 191.58443450927734, 476.0679244995117)\n",
      "person\n",
      "(158.9387083053589, 209.08342838287354, 193.760986328125, 473.5187530517578)\n",
      "person\n",
      "(159.12807941436768, 209.20819759368896, 193.38348388671875, 475.5366516113281)\n",
      "person\n",
      "(158.39781761169434, 210.76887130737305, 190.6674575805664, 476.26277923583984)\n",
      "person\n",
      "(158.82827281951904, 212.54440784454346, 192.29944229125977, 475.66314697265625)\n",
      "person\n",
      "(159.00631427764893, 213.81873607635498, 189.81042861938477, 476.61781311035156)\n",
      "person\n",
      "(159.26661014556885, 214.50692653656006, 190.17513275146484, 476.69628143310547)\n",
      "person\n",
      "(159.55830574035645, 215.3609561920166, 190.48038482666016, 478.22444915771484)\n",
      "person\n",
      "(159.92414474487305, 216.02992057800293, 193.3631706237793, 478.3228302001953)\n",
      "person\n",
      "(159.45347785949707, 218.89660835266113, 193.13701629638672, 478.37833404541016)\n",
      "person\n",
      "(159.51517581939697, 220.19163608551025, 191.96369171142578, 485.13500213623047)\n",
      "person\n",
      "(159.8862648010254, 220.9123992919922, 193.96230697631836, 490.8979034423828)\n",
      "person\n",
      "(160.40472507476807, 220.9979009628296, 190.31021118164062, 494.7071838378906)\n",
      "person\n",
      "(161.44620895385742, 220.72586059570312, 196.37409210205078, 490.34473419189453)\n",
      "person\n",
      "(161.35273933410645, 221.5322971343994, 200.63251495361328, 489.78343963623047)\n",
      "person\n",
      "(161.382737159729, 232.9154634475708, 196.31860733032227, 513.5041809082031)\n",
      "person\n",
      "(162.04276084899902, 233.76843452453613, 196.44702911376953, 514.911994934082)\n",
      "person\n",
      "(166.351318359375, 235.74743270874023, 194.71776962280273, 518.612060546875)\n",
      "person\n",
      "(165.46607494354248, 219.62624073028564, 215.72940826416016, 500.06229400634766)\n",
      "person\n",
      "(165.8285093307495, 221.11551761627197, 215.70903778076172, 504.8567581176758)\n",
      "person\n",
      "(168.38027000427246, 233.45272064208984, 197.5588035583496, 524.4388580322266)\n",
      "person\n",
      "(170.25803089141846, 232.79744625091553, 200.9969711303711, 527.4175643920898)\n",
      "person\n",
      "(170.3490114212036, 220.76441287994385, 212.8567886352539, 515.8185195922852)\n",
      "person\n",
      "(178.62533569335938, 231.08436584472656, 215.69087982177734, 510.0748825073242)\n",
      "person\n",
      "(178.96512508392334, 231.5791368484497, 214.1339111328125, 507.4110412597656)\n",
      "person\n",
      "(179.96975898742676, 232.19538688659668, 212.83729553222656, 511.1687469482422)\n",
      "person\n",
      "(180.18476486206055, 232.28307723999023, 214.51229095458984, 513.526496887207)\n",
      "person\n",
      "(180.44302940368652, 232.22408294677734, 212.82630920410156, 516.6832733154297)\n",
      "person\n",
      "(180.7934045791626, 232.41146564483643, 212.08576202392578, 520.7748031616211)\n",
      "person\n",
      "(181.0949420928955, 233.5400676727295, 211.36770248413086, 524.2666625976562)\n",
      "person\n",
      "(181.9429349899292, 233.7013578414917, 213.8637351989746, 520.2216339111328)\n",
      "person\n",
      "(182.34835624694824, 234.47693824768066, 213.65602493286133, 522.6340484619141)\n",
      "person\n",
      "(182.69561290740967, 235.0556230545044, 212.20264434814453, 526.2186050415039)\n",
      "person\n",
      "(182.8741693496704, 235.37376880645752, 213.64269256591797, 527.9063034057617)\n",
      "person\n",
      "(183.0774450302124, 236.1417531967163, 212.27350234985352, 531.7787933349609)\n",
      "person\n",
      "(182.89498329162598, 236.98851585388184, 210.40403366088867, 536.6750335693359)\n",
      "person\n",
      "(183.28552722930908, 236.9884157180786, 209.50639724731445, 540.8551025390625)\n",
      "person\n",
      "(183.35403442382812, 237.68720626831055, 209.59470748901367, 541.0765838623047)\n",
      "person\n",
      "(183.4828805923462, 237.4232053756714, 209.90991592407227, 543.1722259521484)\n",
      "person\n",
      "(183.41134071350098, 237.11740493774414, 211.37651443481445, 540.7353210449219)\n",
      "person\n",
      "(183.39231491088867, 236.67818069458008, 211.4008331298828, 541.8350219726562)\n",
      "person\n",
      "(183.21430206298828, 236.82031631469727, 210.64897537231445, 544.0013885498047)\n",
      "person\n",
      "(183.15622329711914, 236.79508209228516, 212.39307403564453, 543.1049728393555)\n",
      "person\n",
      "(182.4038028717041, 236.80423736572266, 211.77892684936523, 543.0558013916016)\n",
      "person\n",
      "(182.3005771636963, 237.05732345581055, 209.50592041015625, 548.4043884277344)\n",
      "person\n",
      "(181.42476081848145, 236.19194984436035, 213.0257797241211, 534.6596908569336)\n",
      "person\n",
      "(181.5297031402588, 238.24358940124512, 212.70936965942383, 563.3470153808594)\n",
      "person\n",
      "(180.9303903579712, 237.7127981185913, 214.24434661865234, 565.2953720092773)\n",
      "person\n",
      "(180.5423355102539, 237.20023155212402, 216.48426055908203, 564.0586471557617)\n",
      "person\n",
      "(179.9267864227295, 236.22576713562012, 215.60041427612305, 564.537353515625)\n",
      "person\n",
      "(179.6863603591919, 236.57213687896729, 214.15809631347656, 563.4990692138672)\n",
      "person\n",
      "(179.05227184295654, 236.0791254043579, 214.04062271118164, 564.3817138671875)\n",
      "person\n",
      "(179.2196559906006, 235.7332420349121, 214.82656478881836, 567.1253967285156)\n",
      "person\n",
      "(179.0588665008545, 235.93597412109375, 214.39300537109375, 567.1565246582031)\n",
      "person\n",
      "(179.20213222503662, 236.2662935256958, 214.18222427368164, 566.7572021484375)\n",
      "person\n",
      "(178.57206344604492, 235.40817260742188, 212.99240112304688, 568.521728515625)\n",
      "person\n",
      "(178.14897537231445, 235.25854110717773, 213.05042266845703, 568.0769729614258)\n",
      "person\n",
      "(178.2462215423584, 235.25047302246094, 214.13955688476562, 567.8865814208984)\n",
      "person\n",
      "(177.82187461853027, 234.98682975769043, 214.75030899047852, 567.8221893310547)\n",
      "person\n",
      "(177.41397857666016, 234.9524974822998, 215.03061294555664, 569.6189880371094)\n",
      "person\n",
      "(177.01144695281982, 234.93454456329346, 215.159912109375, 570.1509857177734)\n",
      "person\n",
      "(176.8244218826294, 234.72754955291748, 216.30413055419922, 570.7185745239258)\n",
      "person\n",
      "(176.17375373840332, 234.31337356567383, 217.35010147094727, 568.6301422119141)\n",
      "person\n",
      "(175.3582763671875, 234.50689315795898, 218.85669708251953, 566.933479309082)\n",
      "person\n",
      "(175.48683643341064, 235.18445491790771, 219.74016189575195, 565.7184600830078)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "person\n",
      "(175.6784963607788, 236.71181201934814, 220.10520935058594, 562.1432495117188)\n",
      "person\n",
      "(175.75429916381836, 237.87943840026855, 219.65824127197266, 563.0857467651367)\n",
      "person\n",
      "(175.3104829788208, 239.7071886062622, 221.6314697265625, 559.281005859375)\n",
      "person\n",
      "(174.4718885421753, 241.17456436157227, 221.62395477294922, 560.8606338500977)\n",
      "person\n",
      "(172.99827575683594, 243.01274299621582, 228.9689826965332, 561.2654113769531)\n",
      "person\n",
      "(170.7727861404419, 246.5313720703125, 229.64593887329102, 556.81884765625)\n",
      "person\n",
      "(165.09562969207764, 251.77971839904785, 222.53808975219727, 550.0817108154297)\n",
      "person\n",
      "(160.0647497177124, 262.29289054870605, 216.74139022827148, 547.8874969482422)\n",
      "person\n",
      "(160.11765003204346, 273.67481231689453, 213.49647521972656, 548.4091949462891)\n",
      "person\n",
      "(154.61676120758057, 284.7469711303711, 221.3037872314453, 548.6257934570312)\n",
      "person\n",
      "(146.86168670654297, 291.16722106933594, 215.07757186889648, 552.5506591796875)\n",
      "person\n",
      "(140.74387550354004, 289.9705982208252, 206.29274368286133, 559.0477752685547)\n",
      "person\n",
      "(133.9101219177246, 291.6015815734863, 194.09025192260742, 557.7953338623047)\n",
      "person\n",
      "(139.51018810272217, 285.0743865966797, 160.74190139770508, 557.8256225585938)\n",
      "person\n",
      "(145.58927536010742, 280.09331703186035, 143.59405517578125, 559.1187286376953)\n",
      "person\n",
      "(157.23966121673584, 276.5683364868164, 138.70485305786133, 557.2231292724609)\n",
      "person\n",
      "(166.70631408691406, 275.85302352905273, 139.72976684570312, 557.708740234375)\n",
      "person\n",
      "(175.06510734558105, 275.5525875091553, 135.5039405822754, 558.9410400390625)\n",
      "person\n",
      "(180.95449447631836, 275.5842590332031, 146.3994598388672, 556.1788177490234)\n",
      "person\n",
      "(181.0038185119629, 276.7861747741699, 150.17013549804688, 557.0654296875)\n",
      "person\n",
      "(183.07302474975586, 279.65097427368164, 159.76123809814453, 555.0444412231445)\n",
      "person\n",
      "(185.09480953216553, 280.10207176208496, 178.55287551879883, 556.8644714355469)\n",
      "person\n",
      "(184.89598274230957, 285.90986251831055, 169.71446990966797, 554.6123123168945)\n",
      "person\n",
      "(192.6693820953369, 292.92500495910645, 177.83950805664062, 558.27880859375)\n",
      "person\n",
      "(192.75153636932373, 293.62998962402344, 169.71744537353516, 557.707633972168)\n",
      "person\n",
      "(195.2250051498413, 295.80370903015137, 158.64635467529297, 557.9093551635742)\n",
      "person\n",
      "(198.81912231445312, 299.5283889770508, 161.84396743774414, 554.7122955322266)\n",
      "person\n",
      "(200.13639450073242, 305.09817123413086, 166.28070831298828, 549.9496078491211)\n",
      "person\n",
      "(201.63311004638672, 313.4116744995117, 171.21034622192383, 541.7240905761719)\n",
      "person\n",
      "(200.4760980606079, 318.5076427459717, 186.30714416503906, 541.4104461669922)\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "'module' object has no attribute 'destroyallwindows'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-99-7f1bd2b37e6a>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     91\u001b[0m       \u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mimshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'object detection'\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mresize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimage_np\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m800\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m600\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     92\u001b[0m       \u001b[0;32mif\u001b[0m \u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwaitKey\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m25\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m&\u001b[0m \u001b[0;36m0xFF\u001b[0m \u001b[0;34m==\u001b[0m\u001b[0mord\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'q'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 93\u001b[0;31m          \u001b[0mcv2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdestroyallwindows\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     94\u001b[0m          \u001b[0;32mbreak\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'module' object has no attribute 'destroyallwindows'"
     ]
    }
   ],
   "source": [
    "with detection_graph.as_default():\n",
    "  with tf.Session(graph=detection_graph) as sess:\n",
    "    # Definite input and output Tensors for detection_graph\n",
    "    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')\n",
    "    # Each box represents a part of the image where a particular object was detected.\n",
    "    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')\n",
    "    # Each score represent how level of confidence for each of the objects.\n",
    "    # Score is shown on the result image, together with the class label.\n",
    "    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')\n",
    "    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')\n",
    "    num_detections = detection_graph.get_tensor_by_name('num_detections:0')\n",
    "    while True:\n",
    "      ret,image_np=cap.read()\n",
    "      # the array based representation of the image will be used later in order to prepare the\n",
    "      # result image with boxes and labels on it.\n",
    "      #image_np = load_image_into_numpy_array(image)\n",
    "      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]\n",
    "      image_np_expanded = np.expand_dims(image_np, axis=0)\n",
    "      # Actual detection.\n",
    "      (boxes, scores, classes, num) = sess.run(\n",
    "          [detection_boxes, detection_scores, detection_classes, num_detections],\n",
    "          feed_dict={image_tensor: image_np_expanded})\n",
    "\n",
    "      # Create a display string (and color) for every box location, group any boxes\n",
    "      # that correspond to the same location.\n",
    "      image=image_np\n",
    "      im_width, im_height, channels = image.shape\n",
    "      boxes=np.squeeze(boxes)\n",
    "      classes=np.squeeze(classes).astype(np.int32)\n",
    "      scores=np.squeeze(scores)\n",
    "      category_index=category_index\n",
    "      instance_masks=None\n",
    "      keypoints=None\n",
    "      use_normalized_coordinates=False\n",
    "      max_boxes_to_draw=20\n",
    "      min_score_thresh=.5\n",
    "      agnostic_mode=False\n",
    "      line_thickness=4\n",
    "      box_to_display_str_map = collections.defaultdict(list)\n",
    "      box_to_color_map = collections.defaultdict(str)\n",
    "      box_to_instance_masks_map = {}\n",
    "      box_to_keypoints_map = collections.defaultdict(list)\n",
    "      if not max_boxes_to_draw:\n",
    "        max_boxes_to_draw = boxes.shape[0]\n",
    "      for i in range(min(max_boxes_to_draw, boxes.shape[0])):\n",
    "        if scores is None or scores[i] > min_score_thresh:\n",
    "          box = tuple(boxes[i].tolist())\n",
    "          if instance_masks is not None:\n",
    "            box_to_instance_masks_map[box] = instance_masks[i]\n",
    "          if keypoints is not None:\n",
    "            box_to_keypoints_map[box].extend(keypoints[i])\n",
    "          if scores is None:\n",
    "            box_to_color_map[box] = 'black'\n",
    "          else:\n",
    "            if not agnostic_mode:\n",
    "              if classes[i] in category_index.keys():\n",
    "                class_name = category_index[classes[i]]['name']\n",
    "              else:\n",
    "                class_name = 'N/A'\n",
    "              display_str = '{}'.format(class_name)\n",
    "            else:\n",
    "                 display_str = 'score: {}%'.format(int(100 * scores[i]))\n",
    "              #display_str = 'score: {}%'.format(int(100 * scores[i]))\n",
    "\n",
    "\n",
    "            if display_str =='person':\n",
    "                print(display_str)\n",
    "                box_to_display_str_map[box].append(display_str)\n",
    "                if agnostic_mode:\n",
    "                  box_to_color_map[box] = 'DarkOrange'\n",
    "                else:\n",
    "                  box_to_color_map[box] = STANDARD_COLORS[\n",
    "                      classes[i] % len(STANDARD_COLORS)]\n",
    "      # Draw all boxes onto image.\n",
    "      for box, color in box_to_color_map.items():\n",
    "         ymin, xmin, ymax, xmax = box\n",
    "         #print(box)\n",
    "         (xminn, xmaxx, yminn, ymaxx) =(xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)\n",
    "         print(xminn,xmaxx,yminn,ymaxx)\n",
    "\n",
    "      vis_util.visualize_boxes_and_labels_on_image_array(\n",
    "          image_np,\n",
    "          np.squeeze(boxes),\n",
    "          np.squeeze(classes).astype(np.int32),\n",
    "          np.squeeze(scores),\n",
    "          category_index,\n",
    "          use_normalized_coordinates=True,\n",
    "          max_boxes_to_draw=20,\n",
    "          min_score_thresh=.5,\n",
    "          line_thickness=8)\n",
    "      cv2.imshow('object detection',cv2.resize(image_np,(800,600)))\n",
    "      if cv2.waitKey(25) & 0xFF ==ord('q'):\n",
    "      \t cv2.destroyallwindows()\n",
    "         break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "3a5wMHN8WKMh"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "autoexec": {
      "startup": false,
      "wait_interval": 0
     }
    },
    "colab_type": "code",
    "id": "LQSEnEsPWKMj"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "default_view": {},
   "name": "object_detection_tutorial.ipynb?workspaceId=ronnyvotel:python_inference::citc",
   "provenance": [],
   "version": "0.3.2",
   "views": {}
  },
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15rc1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
