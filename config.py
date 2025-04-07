#数据集相关配置
#------------------------------

#图像路径配置
TEST_IMAGE_DIR = './data/test'

TRAIN_IMAGE_DIR = './data/train'

VALID_IMAGE_DIR = './data/valid'

#标签路径配置
TRAIN_MASK_DIR = './data/train'

VALID_MASK_DIR = '.data/valid'
#-------------------------------



#标签预处理参数
#-------------------------------

#由于标签图相中的像素值可能不是纯0或者255，
#需要阙值对灰度图进行二值化处理
BINARIZATION_THRESHOLD = 128

#-------------------------------



#图像相关配置
#-------------------------------

#DeepGlobe 数据集中图像尺寸为1024*1024, RGB格式
IMAGE_SIZE = (1024,1024)

INPUT_CHANNELS = 3 #RGB图像有3个通道
OUTPUT_CHANNELS = 1 #分割任务输出为单通道二值mask
MUM_CLASSES = 2 #类别数： 背景和通道

#-------------------------------



#训练相关配置
#-------------------------------

BATCH_SIZE = 8 #每个batch的样本数，根据显存大小调整
NUM_EPOCHS = 50 #训练轮数
LEARNING_RATE = 0.001 #学习率
WEIGHT_DECAY = 1e-4 #权重衰减（正则化）
MOMENTUM = 0.9 #动量参数

#-------------------------------



#模型保存与日志配置
#------------------------------

CHECKPOINT_DIR = '.checkpoints' #模型权重保存目录
MODEL_NAME = 'unet_basic.pth'
LOG_INTERVAL = 10 #每个多少个batch输出一次日志

#------------------------------



#其他配置
#------------------------------

#随机种子，可用于实验复现
SEED = 42