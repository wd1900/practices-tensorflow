import os
import tensorflow as tf
import argparse
from PIL import Image

'''
class1 -- 1.jpg
       -- 2.jpg
       -- ...

class2 -- 1.jpg
       -- 2.jpg
       -- ...
...

'''
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="where to get input images")
parser.add_argument("--output_dir", required=True, help="where to put output tfrecords")
parser.add_argument("--img_width", required=True, type=int, help="the width of image")
parser.add_argument("--img_height", required=True, type=int, help="the height of image")

a = parser.parse_args()


classes = ["dance", "nodance"]

writer = tf.python_io.TFRecordWriter(a.output_dir)
for name in os.listdir(a.input_dir):
    class_path = a.input_dir + "/" + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize((a.img_width, a.img_height))
        for i in [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270, 'none']:
        # for i in ['none']:
            img_tran = img
            if i != 'none':
                img_tran = img_tran.transpose(i)
                
            img_raw = img_tran.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
                'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
writer.close()
