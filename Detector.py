# -*- coding:utf-8 -*-
import time
import numpy as np
from PIL import Image
import onnxruntime
from DataLoader import getDataset


##################################################
# change input data format
##################################################
def preprocess(img_file, w, h):
    # convert input data into [1,3,w,h]
    img = Image.open(img_file)
    img = img.resize((w, h), Image.BILINEAR)
    # convert the input data into the float32 input
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data.astype('float32'), np.array(img)


if __name__ == "__main__":
    datapath = r'D:\datapath'  # image data path
    _, _, data_dict = getDataset(datapath)
    img_file = r'D:\1.png'  # test image
    input_data, raw_data = preprocess(img_file, 224, 224)
    session = onnxruntime.InferenceSession('output/mymodel.onnx')
    session.get_modelmeta()
    startt = time.time()
    results = session.run(None, {"input.1": input_data})
    print("inference time :%0.6f" % (time.time() - startt))

    print('predict label :', data_dict[np.argmax(results)])
