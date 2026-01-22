import os
import argparse
import queue

from imgseg_config import *
from models import *
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=CUDA_VISIBLE_DEVICES, type=str)
parser.add_argument('--port', default=Url.port, type=int)
parser.add_argument('--default_model_id', default=DEFAULT_RESNET_MODEL_ID, type=str)
args = parser.parse_args()
# resnet默认模型id
DEFAULT_RESNET_MODEL_ID = args.default_model_id

# 指定显卡号运行
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import base64
import json
from collections import Counter
import io
import threading
import time
import gc
import uuid
from datetime import datetime

import pynvml
import torchvision.transforms as transforms
import requests
import uvicorn

from fastapi import Response, FastAPI
from typing import Optional
from PIL import Image
from io import BytesIO
import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader
import cv2
import numpy as np
from starlette.background import BackgroundTasks
import easyocr

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from data_loader import splitData, WeatherDataSet
from config import Common, Train
from model import model as Model
from train import train, val

from paddleocr import PaddleOCR, draw_ocr

import re
app = FastAPI()



# 获取当前py文件所在的文件夹路径
rootpath = os.path.dirname(os.path.abspath(__file__))


train_info_dir_path = os.path.join(rootpath, TRAIN_INFO)       # resnet训练状态信息所产生文件的存放路径
model_dir_path = os.path.join(rootpath, RESNET_MODEL)        # resnet训练产生的模型的存放路径
data_path = os.path.join(rootpath, RESNET_DATA)        # resnet训练数据集存放路径


if not os.path.exists(train_info_dir_path):
    os.mkdir(train_info_dir_path)
if not os.path.exists(model_dir_path):
    os.mkdir(model_dir_path)
if not os.path.exists(data_path):
    os.mkdir(data_path)

# 初始化ocr模型
# reader = easyocr.Reader(['ch_sim', 'en'])
reader = PaddleOCR(use_angle_cls=False, lang="ch")
# sam_checkpoint = r"../sam_pth/sam_vit_b_01ec64.pth"
# model_type = "vit_b"

# 加载sam模型
sam_checkpoint = r"../sam_pth/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

# 加载resnet默认模型
default_model_path = os.path.join(model_dir_path, DEFAULT_RESNET_MODEL_ID)
default_checkpoint = torch.load(default_model_path + '.pth')
# default_model = default_checkpoint['model']
# default_model.load_state_dict(default_checkpoint['model_state_dict'])

# sam推理中 线程消耗大量显存时的全局锁
samLock = threading.Lock()
# ocr的全局锁
ocrLock = threading.Lock()
# 加载resnet模型时的全局锁
resnetLock = threading.Lock()
# 修改is_train全局变量时的全局锁
trainLock = threading.Lock()
# 用来限制同一时间只有一个训练任务的Flag全局变量
is_train = False



# 根据uuid4获取唯一的task_id
def get_task_id(name: str):
    return name + str(uuid.uuid4())


def url_update(original_string):
    # 使用正则表达式匹配地址部分，并替换为新地址
    pattern = r"(?<=://)(.*?)/minio"
    pattern1 = r"(?<=://)(.*?)/lowcode"
    new_string = ''
    print(original_string)
    if "43.247.90.34" in original_string.lower():
        new_string = re.sub(pattern, "10.100.14.77:9000", original_string)
        new_string = re.sub(pattern1, "10.100.14.77:9000/lowcode", original_string)
    elif "163.53.168.57" in original_string.lower():
        new_string = re.sub(pattern, "10.100.13.184:9000", original_string)
        new_string = re.sub(pattern1, "10.100.13.184:9000/lowcode", original_string)
    else:
        new_string = original_string
    # 去掉 "/minio" 部分
    # new_string = new_string.replace("/minio", "")
    new_string = new_string.replace("https", "http")
    return new_string
# 获取url上的图像,如果出现异常将异常返回
def get_image(url):
    url = url_update(url)
    try:
        response = requests.get(url, verify=False)
        # print("response.status_code",response.status_code)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            response.close()
            print(f"{url}图片已成功读取")
            print(f"读取图片已完成，时间:",datetime.now())
            return image
        else:
            print(f"无法获取{url}图片内容")
            raise Exception(f"无法获取{url}图片内容，response状态码: {response.status_code}")
    except Exception as e:
        print(f"读取过程出错，错误原因：{e}")
        return e
def get_image_numpy(url):
    url = url_update(url)
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()  # 检查是否有错误发生

        # 将图像数据转换为NumPy数组
        image_array = np.asarray(bytearray(response.content), dtype="uint8")

        # 使用OpenCV解码图像
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        print(f"{url}图片已成功读取")
        print(f"读取图片已完成，时间:", datetime.now())
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        response.close()
        return image
    except Exception as e:
        print(f"读取过程出错，错误原因：: {e}")
        return e


def add_border(image, border_width=20, border_color=(255, 255, 255)):
    """
    在图像外围添加边框

    Parameters:
    - image: NumPy数组，表示输入图像
    - border_width: 整数，边框宽度，默认为20像素
    - border_color: 元组，表示边框颜色，默认为白色 (255, 255, 255)

    Returns:
    - 带有边框的新图像
    """
    # 确保输入图像是3通道的彩色图像
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    # 转换为三通道
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    # 复制图像以防止修改原始数据
    image_with_border = np.copy(image)
    # 获取图像的高度和宽度
    height, width, _ = image_with_border.shape

    # 创建新的带有边框的图像
    new_height = height + 2 * border_width
    new_width = width + 2 * border_width

    # 在新图像中复制原始图像
    image_with_border = np.full((new_height, new_width, 3), border_color, dtype=np.uint8)
    image_with_border[border_width:height + border_width, border_width:width + border_width, :] = image

    return image_with_border

# 求所给图像的rgb平均色取整
def rgb_mean(image):
    means = cv2.mean(image)
    means = list(map(int, means))
    return means


# 根据掩码求得最小包络框，返回包络框的左上角坐标、宽高和包络框图像
def img_crop(mask, image):
    min_row = float('inf')  # 初始化最小行为正无穷大
    max_row = float('-inf')  # 初始化最大行为负无穷大
    min_col = float('inf')  # 初始化最小列为正无穷大
    max_col = float('-inf')  # 初始化最大列为负无穷大
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                # 更新最小行、最大行、最小列和最大列的值
                min_col = min(min_col, i)
                max_col = max(max_col, i)
                min_row = min(min_row, j)
                max_row = max(max_row, j)
    if min_col <= max_col and min_row <= max_row:
        width = max_row - min_row
        height = max_col - min_col
        img_component = image[min_col:max_col, min_row:max_row]
    # 掩码全部为0 图片就是原图片
    else:
        height, width, _ = image.shape
        img_component = image
        min_row = 0
        min_col = 0
    return min_row, min_col, width, height, img_component

# 根据掩码求得最小包络框，返回包络框的左上角坐标、宽高和轮廓图像
def img_crop_1(mask, image):
    mask_copy = mask
    mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    # 创建一个黑色背景图像，与输入图像相同大小
    segmented_image = np.zeros_like(image)
    # 使用掩码提取输入图像中对应位置为True的像素
    segmented_image[mask] = image[mask]
    height, width, channels = image.shape
    new_img = np.ones((height, width, 4)) * 255
    new_img[:, :, :3] = segmented_image
    # for i in range(height):
    #     for j in range(width):
    #         if new_img[i, j, :3].tolist() == [0.0, 0.0, 0.0]:
    #             new_img[i, j, :] = np.array([0.0, 0.0, 0.0, 0])
    # 将图像中RGB值为[0.0, 0.0, 0.0]的像素点的alpha通道设置为0
    zero_pixels = (new_img[:, :, :3] == [0.0, 0.0, 0.0]).all(axis=2)
    new_img[zero_pixels, 3] = 0
    #根据坐标切割图像
    min_row = float('inf')  # 初始化最小行为正无穷大
    max_row = float('-inf')  # 初始化最大行为负无穷大
    min_col = float('inf')  # 初始化最小列为正无穷大
    max_col = float('-inf')  # 初始化最大列为负无穷大
    for i in range(len(mask_copy)):
        for j in range(len(mask_copy[i])):
            if mask_copy[i][j]:
                # 更新最小行、最大行、最小列和最大列的值
                min_col = min(min_col, i)
                max_col = max(max_col, i)
                min_row = min(min_row, j)
                max_row = max(max_row, j)
    if min_col <= max_col and min_row <= max_row:
        width = max_row - min_row
        height = max_col - min_col
        img_component = new_img[min_col:max_col, min_row:max_row]
    # 掩码全部为0 图片就是原图片
    else:
        height, width, _ = new_img.shape
        img_component = new_img
        min_row = 0
        min_col = 0
    return min_row, min_col, width, height, img_component
# 校验输入的prompt，格式正确返回True，错误返回False
def samPromptVerify(input_point, input_label, input_box, select_box):
    point_num = len(input_point)
    label_num = len(input_label)
    box_num = len(input_box)
    # 判断提示点和标签数量是否相等，不相等返回False（不符合输入prompt规则）
    if label_num != point_num:
        return False
    # 判断提示框数量大于1时，如果有提示点返回False（不符合输入prompt规则）
    if box_num > 1 and label_num > 0:
        return False
    # 判断提示点是否是二维格式
    for point in input_point:
        if len(point) != 2:
            return False
    # 判断提示框是否是四维格式
    for box in input_box:
        if len(box) != 4:
            return False
        # x1>x2或y1>y2 说明输入的框不符合做左上角坐标和右下角坐标规则
        if box[0] > box[2] or box[1] > box[3]:
            return False
    if len(select_box) != 4:
        return False
    return True


# 判断对应任务id下的任务状态是否被用户终止（标记为500：TaskStatus.USER_TERMINATED）
# def stopJudge(task_id, open_file_path):
#     with open(open_file_path, 'r') as file:
#         content = json.load(file)
#     return content[task_id]["status"] == TaskStatus.USER_TERMINATED


# 更新json文件，choice:1:初始化一对key和value  0：更新对应key上的value
# def updateJson(task_id, open_file_path, key, value, choice):
#     with open(open_file_path, 'r') as file:
#         content = json.load(file)
#     if choice == 1:
#         content[task_id] = {key: value}
#     if choice == 0:
#         content[task_id][key] = value
#     with open(open_file_path, 'w') as file:
#         json.dump(content, file, ensure_ascii=False, indent=4)


# 利用传入的resnet模型推理,（sam推理中调用的函数）
def detect_label(image, checkpoint):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # 进行缩放
    # 转换成RGB图像
    image = image.convert('RGB')
    image = image.resize(Common.imageSize)
    model = checkpoint['model']
    labels = checkpoint['labels']
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    model = model.to(Common.device)
    # 转为tensor张量
    transform = transforms.ToTensor()
    x = transform(image)
    x = torch.unsqueeze(x, 0)  # 升维
    x = x.to(Common.device)
    # 传入模型
    output = model(x)
    # 使用argmax选出最有可能的结果
    output = torch.argmax(output)
    del model
    del checkpoint
    del image
    gc.collect()
    torch.cuda.empty_cache()
    # 返回预测的组件类型名称
    return labels[output.item()]


# resnet推理函数
def predictResnet(image_url_list, modelId, input_component_type):
    # 初始化组件类型字典 url地址:组件类型名
    component_type = {}
    # 初始化预测结果字典
    predict_result = {}
    # 加载模型
    if modelId != DEFAULT_RESNET_MODEL_ID:
        with resnetLock:
            model_path = os.path.join(model_dir_path, modelId)
            checkpoint = torch.load(model_path+'.pth')
    else:
        checkpoint = default_checkpoint
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    labels = checkpoint['labels']
    # 冻结模型参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    # 转换为评估模式
    model.eval()
    model = model.to(Common.device)
    for image_url in image_url_list:
        # 获取图像
        image = get_image(image_url)
        # 如果读取图片过程出现了异常,修改任务状态码为异常
        if isinstance(image, Exception):
            component_type[image_url] = "Failed to read this url image"
            continue
        # 转换成RGB图像
        image = image.convert('RGB')
        # 进行缩放
        image = image.resize(Common.imageSize)

        # 转为tensor张量
        transform = transforms.ToTensor()
        x = transform(image)
        x = torch.unsqueeze(x, 0)  # 升维
        x = x.to(Common.device)
        # 传入模型
        output = model(x)

        # 使用argmax选出最有可能的结果
        output = torch.argmax(output)
        # 写入组件类型
        component_type[image_url] = labels[output.item()]
        if component_type[image_url] in predict_result:
            predict_result[component_type[image_url]] += 1
        else:
            predict_result[component_type[image_url]] = 1
    del model
    del checkpoint

    image_num = len(image_url_list)
    if input_component_type:
        # 识别准确率初始化为0
        acc = 0
        for key in predict_result:
            if key == input_component_type:
                predict_result[key] = predict_result[key]/image_num
                acc = predict_result[key]
            else:
                predict_result[key] = predict_result[key]/image_num
        predict_result["acc"] = acc
    gc.collect()
    torch.cuda.empty_cache()
    return component_type, predict_result, image_num

# SAM推理函数
def predictSam(image_url, input_point, input_label, input_box, is_predict_type, resnet_model_id, read_text_component_list, cut_outline_flag):

    image = get_image(image_url)
    # 如果读取图片过程出现了异常
    if isinstance(image, Exception):
        return ErrorCode.READ_URL_FAIL
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 没有提示调用SamAutomaticMaskGenerator类直接对图像进行分割生成多个掩码
    if len(input_point) == 0 and len(input_box) == 0:
        # mask_generator = SamAutomaticMaskGenerator(sam)
        masks = auto_mask_generator(image)
        # 产生掩码过程发生了异常，（超时或爆显存）
        if masks is True:
            torch.cuda.empty_cache()
            gc.collect()
            # 先都视为超时异常
            return ErrorCode.TIMEOUT
        keys = ['left_top_coord', 'width', 'height', 'rgb_mean', 'img_component_base64']
        values = []
        # 对多个掩码进行处理
        for mask in masks:
            min_row = int(mask["bbox"][0])
            min_col = int(mask["bbox"][1])
            width = int(mask["bbox"][2])
            height = int(mask["bbox"][3])
            img_component = image[min_col:min_col + height, min_row:min_row + width]
            mean_color = rgb_mean(img_component)
            # 将图像编码为base64格式
            img_component_uint8 = img_component.astype(np.uint8)
            try:
                img_component = Image.fromarray(img_component_uint8)
            # 发生异常，说明img_component有一个维度里没有元素，无法转换为图像
            except Exception:
                continue
            img_io = BytesIO()
            img_component.save(img_io, 'PNG')

            img_byte = img_io.getvalue()
            img_base64 = base64.b64encode(img_byte).decode('utf-8')
            values.append([[min_row, min_col], width, height, mean_color[0:3], img_base64])
        del masks
        component_info = [dict(zip(keys, value)) for value in values]

    # 有提示 调用SamPredictor类预先对图像进行分割预测
    else:
        # predictor = SamPredictor(sam)
        with samLock:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.device))  # 指定显卡号
            flag = False  # 空闲显存是否足够分配和是否超时的标志
            start_time = time.time()
            while True:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if time.time()-start_time > 150:   # 超过150s按超时异常处理
                    flag = True
                    break
                if meminfo.free/1024**3 > 10:   # 空余显存大于10G则分配
                    try:
                        predictor.set_image(image)
                    except RuntimeError as e:
                        # print(f"显存不够分配", {e})
                        flag = True
                    finally:
                        break
                time.sleep(0.1)
        if flag:
            gc.collect()
            torch.cuda.empty_cache()
            # 先都视为超时异常
            return ErrorCode.TIMEOUT
        # 提示框数量大于1，对多个提示框中的图像切割生成多个掩码
        if len(input_box) > 1:
            # 转换成torch张量
            input_boxes = torch.tensor(input_box, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            # 进行掩码预测
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            keys = ['left_top_coord', 'width', 'height', 'rgb_mean', 'img_component_base64']
            values = []
            # 对多个掩码进行处理
            for mask in masks:
                # 前端传入True 确定需要返回轮廓图
                if cut_outline_flag:
                    min_row, min_col, width, height, img_component_numpy = img_crop_1(mask[0].cpu().numpy(), image)
                # 否则默认返回矩形图
                else:
                    min_row, min_col, width, height, img_component_numpy = img_crop(mask[0].cpu().numpy(), image)
                mean_color = rgb_mean(img_component_numpy)
                # 将图像编码为base64格式
                img_component_uint8 = img_component_numpy.astype(np.uint8)
                try:
                    img_component = Image.fromarray(img_component_uint8)
                # 发生异常，说明img_component有一个维度里没有元素，无法转换为图像
                except Exception:
                    continue
                img_io = BytesIO()
                img_component.save(img_io, 'PNG')
                img_byte = img_io.getvalue()
                img_base64 = base64.b64encode(img_byte).decode('utf-8')
                values.append([[min_row, min_col], width, height, mean_color[0:3], img_base64])
            del masks
            component_info = [dict(zip(keys, value)) for value in values]
            if is_predict_type:
                component_info = get_component_info(component_info, resnet_model_id, read_text_component_list,1)
            else:
                for info in component_info:
                    info['component_type'] = 'Image'
        # 提示框的数量是0或1,根据提示点和提示框生成一个掩码
        else:
            if len(input_point):
                input_point = np.array(input_point, dtype=int)
                input_label = np.array(input_label, dtype=int)
            else:
                input_point = None
                input_label = None
            if len(input_box):
                input_box = np.array(input_box, dtype=int)
            else:
                input_box = None
            mask, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            )
            # 前端传入True 确定需要返回轮廓图
            if cut_outline_flag:
                min_row, min_col, width, height, img_component_numpy = img_crop_1(mask[0], image)
            # 否则默认返回矩形图
            else:
                min_row, min_col, width, height, img_component_numpy = img_crop(mask[0], image)
            mean_color = rgb_mean(img_component_numpy)
            # 将图像编码为base64格式
            img_component_uint8 = img_component_numpy.astype(np.uint8)
            img_component = Image.fromarray(img_component_uint8)
            img_io = BytesIO()
            img_component.save(img_io, 'PNG')
            img_byte = img_io.getvalue()
            img_base64 = base64.b64encode(img_byte).decode('utf-8')

            keys = ['left_top_coord', 'width', 'height', 'rgb_mean', 'img_component_base64']
            values = [[[min_row, min_col], width, height, mean_color[0:3], img_base64]]
            component_info = [dict(zip(keys, value)) for value in values]
            if is_predict_type:
                component_info = get_component_info(component_info, resnet_model_id, read_text_component_list,1)
            else:
                for info in component_info:
                    info['component_type'] = 'Image'
    gc.collect()
    torch.cuda.empty_cache()
    print("结束predictsam接口时间",datetime.now())
    return component_info


# 根据resnet预测的不同组件类型调用对应的组件信息获取函数
def get_component_info(new_image_info_list, resnet_model_id, read_text_component_list, sam_predict_id):
    # 确定resnet模型
    if resnet_model_id != DEFAULT_RESNET_MODEL_ID:
        with resnetLock:
            model_path = os.path.join(model_dir_path, resnet_model_id)
            checkpoint = torch.load(model_path + '.pth')
    else:
        checkpoint = default_checkpoint

    # 两次循环 第一次用resnet预测组件类型，第二次根据组件类型作进一步处理（不能一次循环的原因：如Input需要知道所有组件中是否有Icon.Required决定是否进一步处理）
    for image_info in new_image_info_list:
        component_image = convert_base64_to_Image(image_info['img_component_base64'])
        component_type = detect_label(component_image, checkpoint)
        image_info['component_type'] = component_type
    # 第二次循环，根据组件类型调用对应函数
    for image_info in new_image_info_list:
        if "text" in image_info:
            # sampredict2接口中在筛选的过程中已经提前获得了文字信息
            if sam_predict_id == 2:
                textResult = image_info['text']
                # 获取行高和对齐方向
                textResult = get_line_height(textResult)
                image_info['text'] = textResult
                continue
        component_image = convert_base64_to_Image(image_info['img_component_base64'])
        component_type = image_info['component_type']
        # if component_type == "Input":
        #     formMsg = get_form_input_info(reader, component_image)
        #     # 表单信息
        #     image_info['formMsg'] = formMsg
        if component_type in read_text_component_list:
            textResult = get_text_info(reader, component_image)
            # 获取文字行高和对齐方向
            textResult = get_line_height(textResult)
            image_info['text'] = textResult
        if component_type == "Radio.Group":
            textResult = get_radio_coord_width_height_content_list(reader, component_image)
            image_info['text'] = textResult
        if component_type == "Table":
            textResult = get_table_header_info(reader, component_image)
            # other_info = get_table_other_info(reader, component_image)
            image_info['table_header_info'] = textResult
            # image_info['table_header_other_info'] = other_info
        # 识别组件列表：开关、滑块等颜色
        if component_type in NEED_COLOR_COMPONENT:
            colorResult = get_component_color(component_image)
            image_info['component_color'] = colorResult
            image_info['background_color'] = get_background_color(component_image)
        # 识别组件是否有圆角
        if component_type in NEED_ROUND_CORNER_FLAG:
            round_corner_flag, round_corner_height = has_rounded_corners(component_image)
            image_info['round_corner_flag'] = round_corner_flag
            if round_corner_flag:
                image_info['round_corner_height'] = round_corner_height
        # 识别组件rgb边框色
        if component_type in NEED_BOADER_COLOR:
            boader_color = get_avg_border_color(component_image)
            image_info['boader_color'] = boader_color
        # 识别组件横竖方向
        if component_type in NEED_HORIZONTAL_VERTICAL:
            direction = get_direction(component_image)
            image_info['direction'] = direction
        # 识别日期选择器的文本块信息和选择类型
        if component_type == "DatePicker":
            textResult = get_text_info(reader, component_image)
            image_info['text'] = textResult
            # 判断是否是日期区间
            date_interval_flag = get_is_date_picker(textResult)
            image_info['isPicker'] = date_interval_flag
            # 判断区间的模式 年、季度、月、周、日
            mode = get_datepicker_mode(textResult)
            image_info['mode'] = mode
        if component_type == "TimePicker":
            textResult = get_text_info(reader, component_image)
            image_info['text'] = textResult
            # 判断是否时间范围选择器
            time_range_flag = get_is_time_picker(textResult)
            image_info['isPicker'] = time_range_flag
    return new_image_info_list



# 保留筛选框中的文字框图片并获得文字框里的文字信息（内容、颜色、坐标）
def get_crop_text_image(content_coord_width_height_list, image_numpy, box):
    filter_coord_width_height_list = []
    box_left_top_x, box_left_top_y, box_right_bottom_x, box_right_bottom_y = box
    keys = ['left_top_coord', 'width', 'height', 'rgb_mean', 'img_component_base64', 'text', 'component_type']
    values = []
    for content_coord_width_height in content_coord_width_height_list:
        # 获取文字框里的左上角坐标和右下角坐标
        left_top_coord = content_coord_width_height[0]
        image_left_top_x = left_top_coord[0]
        image_left_top_y = left_top_coord[1]
        width = content_coord_width_height[1]
        height = content_coord_width_height[2]
        right_bottom_coord = (left_top_coord[0] + width, left_top_coord[1]+height)
        image_right_bottom_x = right_bottom_coord[0]
        image_right_bottom_y = right_bottom_coord[1]
        text_result = []
        text_info = {}
        # 判断是否在筛选框内
        if (image_left_top_x >= box_left_top_x and image_left_top_y >= box_left_top_y and
                image_right_bottom_x <= box_right_bottom_x and image_right_bottom_y <= box_right_bottom_y):
            try:
                text_image = image_numpy[image_left_top_y:image_right_bottom_y, image_left_top_x:image_right_bottom_x]
            except Exception:
                continue
            mean_color = rgb_mean(text_image)
            img_component_uint8 = text_image.astype(np.uint8)
            try:
                text_image = Image.fromarray(img_component_uint8)
            # 发生异常，说明img_component有一个维度里没有元素，无法转换为图像
            except Exception:
                continue
            img_io = BytesIO()
            text_image.save(img_io, 'PNG')
            img_byte = img_io.getvalue()
            img_base64 = base64.b64encode(img_byte).decode('utf-8')
            text_info['left_top_coord'] = [0, 0]
            text_info['width'] = width
            text_info['height'] = height
            text_info['msg'] = content_coord_width_height[3]

            # text_image = text_image.convert("RGB")
            # bg_color = get_background_color(text_image)
            bg_color = get_edge_colors(text_image)
            text_color = get_text_color(bg_color, text_image)
            text_info['backgroundColor'] = bg_color
            text_info['color'] = text_color
            text_result.append(text_info)
            values.append([[image_left_top_x, image_left_top_y], width, height, mean_color[0:3], img_base64, text_result, "Text"])
            filter_coord_width_height_list.append(content_coord_width_height[:3])
    crop_text_image_list = [dict(zip(keys, value)) for value in values]
    return crop_text_image_list, filter_coord_width_height_list


def auto_mask_generator(image):
    # mask_generator = SamAutomaticMaskGenerator(sam)
    with samLock:  # mask_generator.generate(image)需要分配显存，上线程锁以防资源竞争导致爆显存
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.device))  # 指定显卡号
        flag = False  # 空闲显存是否足够分配和是否超时的标志
        start_time = time.time()
        while True:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            end_time = time.time()
            if end_time - start_time > 150:  # 超过150s按超时异常处理
                flag = True
                break
            if meminfo.free / 1024 ** 3 > 20:  # 空余显存大于20G则分配
                try:
                    masks = mask_generator.generate(image)
                except RuntimeError as e:
                    print(f"显存不够分配:{e}")
                    flag = True
                    del masks
                finally:
                    break
            time.sleep(0.1)
    if flag:
        torch.cuda.empty_cache()
        gc.collect()
        return flag
    return masks


# 对提示点生成一个掩码、筛选框作为提示框生成一个掩码作为父节点
def predict_mask(image, input_point, input_label, input_box):
    # predictor = SamPredictor(sam)
    box_mask = None
    point_mask = None
    with samLock:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(args.device))  # 指定显卡号
        flag = False  # 空闲显存是否足够分配和是否超时的标志
        start_time = time.time()
        while True:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if time.time() - start_time > 150:  # 超过150s按超时异常处理
                flag = True
                break
            if meminfo.free / 1024 ** 3 > 10:  # 空余显存大于10G则分配
                try:
                    predictor.set_image(image)
                except RuntimeError as e:
                    # print(f"显存不够分配", {e})
                    flag = True
                finally:
                    break
            time.sleep(0.1)
    if flag:
        torch.cuda.empty_cache()
        gc.collect()
        return flag, flag
    if len(input_point):
        input_point = np.array(input_point, dtype=int)
        input_label = np.array(input_label, dtype=int)
        point_mask, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )
    if len(input_box):
        input_box = np.array([input_box], dtype=int)
        box_mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
    return box_mask, point_mask


# 对自动分割出的掩码进行处理，筛选掉选择框外的掩码图片和文字框内的掩码图片
def processMasks(image, masks, select_box, filter_coord_width_height_list):
    box_left_top_x, box_left_top_y, box_right_bottom_x, box_right_bottom_y = select_box
    keys = ['left_top_coord', 'width', 'height', 'rgb_mean', 'img_component_base64']
    values = []
    # 对多个掩码进行处理
    for mask in masks:
        text_box_flag = False
        min_row = int(mask["bbox"][0])
        min_col = int(mask["bbox"][1])
        width = int(mask["bbox"][2])
        height = int(mask["bbox"][3])
        max_row = min_row + width
        max_col = min_col + height
        # 判断掩码图片是否在select_box内
        select_box_flag = (min_row > box_left_top_x and min_col > box_left_top_y and max_row < box_right_bottom_x and max_col < box_right_bottom_y)
        # 掩码图片不在select_box中，排除
        if not select_box_flag:
            continue
        for filter_coord_width_height in filter_coord_width_height_list:
            text_box_left_top_x, text_box_left_top_y = filter_coord_width_height[0]
            text_box_right_bottom_x = text_box_left_top_x + filter_coord_width_height[1]
            text_box_right_bottom_y = text_box_left_top_y + filter_coord_width_height[2]
            # 掩码图片在文字框图片中，需要排除
            if (min_row >= text_box_left_top_x and min_col >= text_box_left_top_y and max_row <= text_box_right_bottom_x and max_col <= text_box_right_bottom_y):
                text_box_flag = True
                break
        if text_box_flag:
            continue
        img_component = image[min_col:max_col, min_row:max_row]
        mean_color = rgb_mean(img_component)

        img_component_uint8 = img_component.astype(np.uint8)
        try:
            img_component = Image.fromarray(img_component_uint8)
        # 如果发生异常，说明img_component有一个维度里没有元素，无法转换为图像
        except Exception:
            continue
        img_io = BytesIO()
        img_component.save(img_io, 'PNG')
        img_byte = img_io.getvalue()
        # 将图像编码为base64格式
        img_base64 = base64.b64encode(img_byte).decode('utf-8')
        values.append([[min_row, min_col], width, height, mean_color[0:3], img_base64])
    crop_image_info_list = [dict(zip(keys, value)) for value in values]
    return crop_image_info_list


# 对输入提示获得的一个掩码进行处理
def processMask(image, mask, cut_outline_flag=False):
    keys = ['left_top_coord', 'width', 'height', 'rgb_mean', 'img_component_base64']
    if cut_outline_flag:
        min_row, min_col, width, height, img_component_numpy = img_crop_1(mask, image)
    else:
        min_row, min_col, width, height, img_component_numpy = img_crop(mask, image)
    mean_color = rgb_mean(img_component_numpy)
    # 将图像编码为base64格式
    img_component_uint8 = img_component_numpy.astype(np.uint8)
    img_component = Image.fromarray(img_component_uint8)
    img_io = BytesIO()
    img_component.save(img_io, 'PNG')
    img_byte = img_io.getvalue()
    img_base64 = base64.b64encode(img_byte).decode('utf-8')
    values = [[[min_row, min_col], width, height, mean_color[0:3], img_base64]]
    single_image_info_list = [dict(zip(keys, value)) for value in values]
    return single_image_info_list


# 根据有提示生成的掩码图片去掉一部分分割图片
def filter_crop_image(image_info, prompt_image_info):
    image_left_top_corrd = image_info['left_top_coord']
    image_left_top_x = image_left_top_corrd[0]
    image_left_top_y = image_left_top_corrd[1]
    image_width = image_info['width']
    image_height = image_info['height']
    image_right_bottom_x = image_left_top_x + image_width
    image_right_bottom_y = image_left_top_y + image_height

    prompt_image_left_top_coord = prompt_image_info[0]['left_top_coord']
    prompt_image_left_top_x = prompt_image_left_top_coord[0]
    prompt_image_left_top_y = prompt_image_left_top_coord[1]
    prompt_image_width = prompt_image_info[0]['width']
    prompt_image_height = prompt_image_info[0]['height']
    prompt_image_right_bottom_x = prompt_image_left_top_x + prompt_image_width
    prompt_image_right_bottom_y = image_left_top_y + prompt_image_height
    # 分割图片是否完全在根据提示生成的图片中的标志
    return not (image_left_top_x > prompt_image_left_top_x-10 and image_left_top_y > prompt_image_left_top_y-10 and
     image_right_bottom_x < prompt_image_right_bottom_x+10 and image_right_bottom_y < prompt_image_right_bottom_y+10)



# 求相对于筛选框的坐标
def coord_to_select_box(father_left_top_coord, new_image_info_list):
    x = father_left_top_coord[0]
    y = father_left_top_coord[1]
    for image_info in new_image_info_list:
        image_info['left_top_coord'] = [image_info['left_top_coord'][0]-x, image_info['left_top_coord'][1]-y]
    return new_image_info_list


# 将base64编码格式的图片信息转化为Image图像对象
def convert_base64_to_Image(base64_code):
    # 解码Base64数据
    image_data = base64.b64decode(base64_code)
    # 将解码后的数据转换为PIL图像
    return Image.open(BytesIO(image_data))


# SAM推理函数
def predictSam2(image_url, input_point, input_label, select_box, is_predict_type, resnet_model_id, read_text_component_list,cut_outline_flag):
    image = get_image(image_url)
    # 如果读取图片过程出现了异常
    if isinstance(image, Exception):
        return ErrorCode.READ_URL_FAIL
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 获得选择框内的分割掩码
    masks = auto_mask_generator(image)
    # 说明返回了flag
    if masks is True:
        # 先都视为超时异常
        return ErrorCode.TIMEOUT
    # 获取输入图片中所有文字框的内容、左上角坐标与宽高组成的信息列表
    content_coord_width_height_list = get_text_content(reader, image)
    # 输入图片中有文字框
    if not(content_coord_width_height_list is None):
        # 过滤掉筛选框中的掩码图片和文字框图片
        crop_text_image_list, filter_coord_width_height_list = get_crop_text_image(content_coord_width_height_list, image, select_box)
        # 过滤文字框中的掩码得到掩码图片信息
        crop_image_info_list = processMasks(image, masks, select_box, filter_coord_width_height_list)
        # 图片信息列表 = 筛选框中的文字框图片+筛选框中的掩码图片
        image_info_list = crop_text_image_list + crop_image_info_list
    # 输入图片中没有文字框
    else:
        crop_text_image_list = []
        image_info_list = processMasks(image, masks, select_box, [])
    # 如果有提示点和筛选框
    if len(input_point) > 0:
        box_mask, point_mask = predict_mask(image, input_point, input_label, select_box)
        # 说明返回的是flag发生了异常
        if point_mask is True:
            # 先都视为超时异常
            return ErrorCode.TIMEOUT
        box_prompt_image_info = processMask(image, box_mask[0])
        point_prompt_image_info = processMask(image, point_mask[0], cut_outline_flag)
        #
        new_image_info_list = [image_info for image_info in image_info_list if
                               filter_crop_image(image_info, point_prompt_image_info)]
    # 只有筛选框没有提示点
    else:
        box_mask, _ = predict_mask(image, [], [], select_box)
        if box_mask is True:
            # 先都视为超时异常
            return ErrorCode.TIMEOUT
        box_prompt_image_info = processMask(image, box_mask[0])
        point_prompt_image_info = []
        new_image_info_list = image_info_list
    del masks
    new_image_info_list = point_prompt_image_info + new_image_info_list
    if is_predict_type:
        new_image_info_list = get_component_info(new_image_info_list, resnet_model_id, read_text_component_list, 2)

    father_image_info = box_prompt_image_info[:]
    # 更新子节点相对于父节点的左上角坐标
    new_image_info_list = coord_to_select_box(father_image_info[0]['left_top_coord'], new_image_info_list)
    father_image = convert_base64_to_Image(father_image_info[0]['img_component_base64'])
    father_image_info[0]['backgroundColor'] = get_background_color(father_image, new_image_info_list)
    father_image_info[0]['children'] = new_image_info_list

    gc.collect()
    torch.cuda.empty_cache()
    return father_image_info

def points2Box(points):
    x_min = points[0][0]
    y_min = points[0][1]
    x_max = points[0][0]
    y_max = points[0][1]
    for point in points:
        if point[0] < x_min:
            x_min = point[0]
        elif point[0] > x_max:
            x_max = point[0]
        if point[1] < y_min:
            y_min = point[1]
        elif point[1] > y_max:
            y_max = point[1]
    box = [[x_min, y_min, x_max, y_max]]
    return box



@app.post(Url.PREDICT_RESNET)
def predictResnetStart(item: ResnetStartItem):
    image_url_list = item.image_url_list
    topic_id = item.topic_id
    user_id = item.user_id
    task_id = get_task_id("resnetPredict_")
    # resnet模型存放在同级目录下的model文件夹
    modelIdList = os.listdir(model_dir_path)
    # 检查前端传入的resnet模型ID是否存在，不存在返回错误信息
    if str(item.model_id) + ".pth" not in modelIdList:
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.MODEL_ID_ERROR,  # 模型id输入错误（不存在）
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    component_type, predict_result, image_num = predictResnet(image_url_list, item.model_id ,item.input_component_type)
    return {
        "code": ResponseCode.SUCCESS_GENERAL,
        "status": StatusCode.OK,
        "message": Message.SUCCESS,
        "payload": {
            "task_id": task_id,
            "task_status": TaskStatus.NORMAL_END,  # 推理正常结束
            "component_type": component_type,  # 返回对应任务id中的组件类型
            "predict_result": predict_result,  # 返回对应任务id中的统计结果
            "image_num": image_num,  # 返回对应任务id中的输入预测的图片数量
            "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
            "user_id": user_id
        }
    }






@app.post(Url.PREDICT_SAM)
def predictSamStart(item: SamStartItem):
    print("********************************************************************************************")
    print("进入samstart接口时间",datetime.now())
    print("samStart接口接收前端数据：", item)
    print("********************************************************************************************")
    image_url = item.image_url
    topic_id = item.topic_id
    user_id = item.user_id
    input_point = item.prompt['input_point']
    input_label = item.prompt['input_label']
    input_box = item.prompt['input_box']
    # 判断输入的prompt格式是否正确
    if not samPromptVerify(input_point, input_label, input_box, [0,0,0,0]):
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.DATA_FORMAT_ERROR,  # 输入prompt数据格式有误
                "topic_id": topic_id,
                "user_id": user_id
            }
        }
    # resnet模型存放在同级目录下的model文件夹
    modelIdList = os.listdir(model_dir_path)
    # 检查前端传入的resnet模型ID是否存在，不存在返回错误信息
    if str(item.resnet_model_id) + ".pth" not in modelIdList:
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.MODEL_ID_ERROR,  # 模型id输入错误（不存在）
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    task_id = get_task_id("samPredict_")
    print("***************")
    print(f"task_id:{task_id}")
    print("***************")
    component_info = predictSam(image_url, input_point, input_label, input_box,item.is_predict_type, item.resnet_model_id, item.read_text_component_list, item.cut_outline_flag)
    if component_info == ErrorCode.READ_URL_FAIL:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "err": ErrorCode.READ_URL_FAIL,
                "task_id": task_id,
                "task_status": TaskStatus.ABNORMAL_TERMINATION,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    elif component_info == ErrorCode.TIMEOUT:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "err": ErrorCode.TIMEOUT,
                "task_id": task_id,
                "task_status": TaskStatus.ABNORMAL_TERMINATION,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    else:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "task_id": task_id,
                "task_status": TaskStatus.NORMAL_END,  # 正在推理
                "component_info": component_info,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }


@app.post(Url.PREDICT_SAM_2)
def predictSamStart2(item: SamStartItem2):
    print("********************************************************************************************")
    print(datetime.now())
    print("samStart2接口接收前端数据：", item)
    print("********************************************************************************************")
    image_url = item.image_url
    topic_id = item.topic_id
    user_id = item.user_id
    input_point = item.prompt['input_point']
    input_label = item.prompt['input_label']
    select_box = item.select_box

    # 判断输入的prompt格式是否正确
    if not samPromptVerify(input_point, input_label, [], select_box):
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.DATA_FORMAT_ERROR,  # 输入prompt数据格式有误
                "topic_id": topic_id,
                "user_id": user_id
            }
        }

    # resnet模型存放在同级目录下的model文件夹
    modelIdList = os.listdir(model_dir_path)
    # 检查前端传入的resnet模型ID是否存在，不存在返回错误信息
    if str(item.resnet_model_id) + ".pth" not in modelIdList:
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.MODEL_ID_ERROR,  # 模型id输入错误（不存在）
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }

    task_id = get_task_id("samPredict_")
    print("***************")
    print(f"task_id:{task_id}")
    print("***************")
    component_info = predictSam2(image_url, input_point, input_label, select_box, item.is_predict_type, item.resnet_model_id, item.read_text_component_list, item.cut_outline_flag)
    if component_info == ErrorCode.READ_URL_FAIL:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "err": ErrorCode.READ_URL_FAIL,
                "task_id": task_id,
                "task_status": TaskStatus.ABNORMAL_TERMINATION,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    elif component_info == ErrorCode.TIMEOUT:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "err": ErrorCode.TIMEOUT,
                "task_id": task_id,
                "task_status": TaskStatus.ABNORMAL_TERMINATION,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    else:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "task_id": task_id,
                "task_status": TaskStatus.NORMAL_END,  # 正在推理
                "component_info": component_info,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }


@app.post(Url.PREDICT_SAM_3)
def predictSamStart3(item: SamStartItem3):
    print("********************************************************************************************")
    print(datetime.now())
    print("samStart3接口接收前端数据：", item)
    print("********************************************************************************************")
    image_url = item.image_url
    topic_id = item.topic_id
    user_id = item.user_id
    points = item.points
    input_box = points2Box(points)
    input_point = item.prompt['input_point']
    input_label = item.prompt['input_label']
    # input_box = item.prompt['input_box']
    # 判断输入的prompt格式是否正确
    if not samPromptVerify(input_point, input_label, input_box, [0,0,0,0]):
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.DATA_FORMAT_ERROR,  # 输入prompt数据格式有误
                "topic_id": topic_id,
                "user_id": user_id
            }
        }
    # resnet模型存放在同级目录下的model文件夹
    modelIdList = os.listdir(model_dir_path)
    # 检查前端传入的resnet模型ID是否存在，不存在返回错误信息
    if str(item.resnet_model_id) + ".pth" not in modelIdList:
        return {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": "失败",
            "payload": {
                "err": ErrorCode.MODEL_ID_ERROR,  # 模型id输入错误（不存在）
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    task_id = get_task_id("samPredict_")
    print("***************")
    print(f"task_id:{task_id}")
    print("***************")
    component_info = predictSam(image_url, input_point, input_label, input_box,item.is_predict_type, item.resnet_model_id, item.read_text_component_list, True)
    if component_info == ErrorCode.READ_URL_FAIL:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "err": ErrorCode.READ_URL_FAIL,
                "task_id": task_id,
                "task_status": TaskStatus.ABNORMAL_TERMINATION,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    elif component_info == ErrorCode.TIMEOUT:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "err": ErrorCode.TIMEOUT,
                "task_id": task_id,
                "task_status": TaskStatus.ABNORMAL_TERMINATION,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }
    else:
        return {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "task_id": task_id,
                "task_status": TaskStatus.NORMAL_END,  # 正在推理
                "component_info": component_info,
                "topic_id": topic_id,  # 返回前端传入的topic_id和user_id
                "user_id": user_id
            }
        }



def training(epochs, batch_size, lr, train_task_id):
    train_task_path = os.path.join(train_info_dir_path,train_task_id)
    # 根据输入的数据集获取类别数
    labels = os.listdir(data_path)
    classNum = len(labels)
    # 1. 获取模型
    model = Model(classNum)
    model.to(Common.device)
    # 2. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 3. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    maxAcc = 0.0
    minLoss = float('inf')
    checkpointName = train_task_id
    # 获取数据
    # 1. 创建并分割数据集，获取数据集数据量
    dataset = WeatherDataSet(data_path)
    # validation_dataset = WeatherDataSet("./test_data")
    train_dataset, validation_dataset = splitData(dataset)
    # train_dataset = dataset

    dataset_num = len(dataset)
    # 2. 训练数据集加载器
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=Train.num_workers)
    # 3. 验证集数据加载器
    valLoader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=Train.num_workers)
    # 开始训练
    for epoch in range(1, epochs + 1):
        print("-----------------")
        trainAcc = train(epoch, trainLoader, model, criterion, optimizer)
        valAcc, singleAcc, valLoss = val(epoch, valLoader, model, criterion, labels)
        with open(f"{train_task_path}.json", "r", encoding='utf-8') as f:
            read_json = json.load(f)
        stop = read_json['task_status']
        if stop == TaskStatus.USER_TERMINATED:
            global is_train
            is_train = False
            return
        read_json['epoch'] = epoch
        read_json['valAccList'].append(valAcc)
        read_json['valLossList'].append(valLoss)
        with open(f"{train_task_path}.json", "w", encoding='utf-8') as f:
            json.dump(read_json, f, indent=4)
        if valAcc >= maxAcc:
            if valAcc == maxAcc and valLoss > minLoss:
                continue
            maxAcc = valAcc
            minLoss = valLoss
            model_path = os.path.join(model_dir_path, checkpointName+".pth")
            # 保存最大模型
            torch.save({
                'model': model,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': valAcc,
                'loss': valLoss,
                'labels': labels,
                'singleAcc': singleAcc,
                'dataset_num': dataset_num
            }
                , model_path)
    # 标记模型正常结束训练
    print("训练结束")
    with open(f"{train_task_path}.json", "r", encoding='utf-8') as f:
        read_json = json.load(f)
    read_json['task_status'] = TaskStatus.NORMAL_END
    with open(f"{train_task_path}.json", "w", encoding='utf-8') as f:
        json.dump(read_json, f, indent=4)
    is_train = False
    gc.collect()
    torch.cuda.empty_cache()


# 训练任务开始api
@app.post(Url.resnetTrainStart)
def trainStart(background_tasks: BackgroundTasks, input: TrainStartItem):
    # 获取参数
    epochs = input.train_params['epochs']
    batch_size = input.train_params['batch_size']
    lr = input.train_params['learning_rate']

    epochs = int(epochs)
    batch_size = int(batch_size)
    lr = float(lr)

    task_type = input.task_type
    user_id = input.user_id

    global is_train
    with trainLock:
        # 如果没有训练任务
        if not is_train:
            is_train = True   # 标记为有训练任务
        else:  # 有训练任务 返回开始训练失败
            return {
                "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
                "status": StatusCode.NOT_FOUND,
                "message": Message.FAILURE,
                "payload": {
                    "err": ErrorCode.CREATE_TRAIN_FAIL,  # 有训练任务正在进行 返回创建训练任务失败
                    "user_id": user_id,
                    "task_type": task_type
                }
            }

    train_task_id = "Train-imgSeg-" + str(uuid.uuid4())  # 生成训练任务ID
    train_task_path = os.path.join(train_info_dir_path,train_task_id)
    background_tasks.add_task(training, epochs, batch_size, lr, train_task_id)

    content = {
        'train_task_id': train_task_id,
        'task_type': task_type,
        'user_id': user_id,
        'task_status': TaskStatus.IN_PROGRESS,
        'epoch': 0,
        'valAccList': [],
        'valLossList': []
    }

    # 将任务信息写入JSON文件
    with open(f"{train_task_path}.json", "w", encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

    status_response_data = {
        'code': ResponseCode.SUCCESS_GENERAL,
        'status': StatusCode.OK,
        'message': Message.SUCCESS,
        'payload': content
    }
    return status_response_data


# 训练信息查询api
@app.post(Url.resnetTrainProgress)
def statusInfo(input: TrainProcessSkipItem):
    train_task_id = input.train_task_id
    user_id = input.user_id
    task_type = input.task_type

    train_task_path = os.path.join(train_info_dir_path,train_task_id)
    # 检查训练任务文件是否存在
    if not os.path.exists(f"{train_task_path}.json"):
        status_response_data = {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.NOT_FOUND,
            "message": Message.FAILURE,
            "payload": {
                "err": ErrorCode.TASK_ID_NOT_FOUND,
                "train_task_id": train_task_id,
                "user_id": user_id,
                "task_type": task_type
            }
        }
        return status_response_data
    modelIdList = os.listdir(model_dir_path)
    modelId = train_task_id

    try:
        # 读取训练任务文件内容
        with open(f"{train_task_path}.json", "r", encoding='utf-8') as f:
            read_json = json.load(f)

        task_status = read_json.get("task_status")  # 获取任务状态
        epoch = read_json.get("epoch")
        valAccList = read_json.get("valAccList")
        valLossList = read_json.get("valLossList")
        # 检查前端传入的resnet模型ID是否保存，没有保存先不返回训练信息acc，只返回训练状态
        if modelId + ".pth" not in modelIdList:
            status_response_data = {
                "code": ResponseCode.SUCCESS_GENERAL,
                "status": StatusCode.OK,
                "message": Message.SUCCESS,
                "payload": {
                    "train_task_id": train_task_id,
                    "user_id": user_id,
                    "task_type": task_type,
                    "task_status": task_status,
                    "epoch": epoch,
                    "valAccList": valAccList,
                    "valLossList": valLossList
                }
            }
        else:
            if modelId != DEFAULT_RESNET_MODEL_ID:
                with resnetLock:
                    model_path = os.path.join(model_dir_path, modelId)
                    checkpoint = torch.load(model_path + '.pth')
            else:
                checkpoint = default_checkpoint
            acc = checkpoint['acc']
            loss = checkpoint['loss']
            labels = checkpoint['labels']
            singleAcc = checkpoint['singleAcc']
            modelEpoch = checkpoint['epoch']
            # 之前保存的模型没有dataset_num键值
            try:
                dataset_num = checkpoint['dataset_num']
            except Exception:
                dataset_num = 0
            status_response_data = {
                "code": ResponseCode.SUCCESS_GENERAL,
                "status": StatusCode.OK,
                "message": Message.SUCCESS,
                "payload": {
                    "train_task_id": train_task_id,
                    "user_id": user_id,
                    "task_type": task_type,
                    "task_status": task_status,
                    "acc": acc,
                    "loss": loss,
                    "epoch": epoch,             # 当前训练的轮数
                    "modelEpoch": modelEpoch,   # 模型是在多少轮保存下来的
                    "labels": labels,
                    "singleAcc": singleAcc,
                    "dataset_num": dataset_num,
                    "valAccList": valAccList,
                    "valLossList": valLossList
                }
            }
    except Exception as err:
        print(err)
        status_response_data = {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.NOT_FOUND,
            "message": Message.FAILURE,
            "payload": {
                "err": ErrorCode.UNKNOWN_EXCEPTION,
                "train_task_id": train_task_id,
                "user_id": user_id,
                "task_type": task_type
            }
        }
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        return status_response_data


# 训练任务终止api
@app.post(Url.resnetTrainSkip)
def stopTrain(input: TrainProcessSkipItem):
    train_task_id = input.train_task_id
    user_id = input.user_id
    task_type = input.task_type
    train_task_path = os.path.join(train_info_dir_path, train_task_id)
    # 检查训练任务文件是否存在
    if not os.path.exists(f"{train_task_path}.json"):
        status_response_data = {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.NOT_FOUND,
            "message": Message.FAILURE,
            "payload": {
                "err": ErrorCode.TASK_ID_NOT_FOUND,
                "train_task_id": train_task_id,
                "user_id": user_id,
                "task_type": task_type
            }
        }
        return status_response_data
    try:
        # 读取训练任务文件内容
        with open(f"{train_task_path}.json", "r", encoding='utf-8') as f:
            read_json = json.load(f)
        # 如果训练任务还没有正常结束
        if read_json["task_status"] != TaskStatus.NORMAL_END:
            read_json["task_status"] = TaskStatus.USER_TERMINATED  # 更新任务状态为用户终止
            # 写入更新后的训练任务文件内容
            with open(f"{train_task_path}.json", "w", encoding='utf-8') as f:
                json.dump(read_json, f, ensure_ascii=False, indent=4)

        status_response_data = {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "train_task_id": train_task_id,
                "user_id": user_id,
                "task_type": task_type,
                "task_status": read_json["task_status"]
            }
        }
    except Exception as err:
        print(err)
        status_response_data = {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.NOT_FOUND,
            "message": Message.FAILURE,
            "payload": {
                "err": ErrorCode.UNKNOWN_EXCEPTION,
                "train_task_id": train_task_id,
                "user_id": user_id,
                "task_type": task_type
            }
        }
    finally:
        return status_response_data


# 单条数据校验（图像地址和图像组件类型一对数据），返回ValidStatus码或READ_URL_FAIL错误码
def singleVal(url: str, component_type: str):
    type_list = os.listdir(data_path)
    image = get_image(url)
    # 如果读取url上的图像失败（get_image返回了异常）返回READ_URL_FAIL错误码
    if isinstance(image, Exception):
        return ErrorCode.READ_URL_FAIL
    width, height = image.size
    # 判断组件类型是否在组件列表中、图像尺寸是否在规定范围中
    if (component_type not in type_list) and (width > ValidDataParam.MAX_TRAIN_IMAGE_WIDTH or
                                                                       height > ValidDataParam.MAX_TRAIN_IMAGE_HEIGHT or
                                                                       width < ValidDataParam.MIN_TRAIN_IMAGE_WIDTH or
                                                                       height < ValidDataParam.MIN_TRAIN_IMAGE_HEIGHT):
        return ValidStatus.BOTH_NON_COMPLIANT           # 图像尺寸和组件类型名均不合规

    elif component_type not in type_list and ValidDataParam.MAX_TRAIN_IMAGE_WIDTH > width > \
            ValidDataParam.MIN_TRAIN_IMAGE_WIDTH and ValidDataParam.MAX_TRAIN_IMAGE_HEIGHT > height > \
            ValidDataParam.MIN_TRAIN_IMAGE_HEIGHT:
        return ValidStatus.TYPE_NON_COMPLIANT           # 组件类型名不合规

    elif (component_type in type_list) and (width > ValidDataParam.MAX_TRAIN_IMAGE_WIDTH or
                                                                     height > ValidDataParam.MAX_TRAIN_IMAGE_HEIGHT or
                                                                     width < ValidDataParam.MIN_TRAIN_IMAGE_WIDTH or
                                                                     height < ValidDataParam.MIN_TRAIN_IMAGE_HEIGHT):
        return ValidStatus.URL_NON_COMPLIANT            # 图像尺寸不合格

    else:
        return ValidStatus.BOTH_COMPLIANT               # 合规数据


# 读取数据并保存
def getDataset(dataset: dict):
    getinfo = {}
    count = 0
    for data in dataset:
        # 读取url和type
        try:
            image_url = dataset[data]['url']
            component_type = dataset[data]['component_type']
        except Exception:
            getinfo[data] = GetTrainDataStatus.URL_OR_TYPE_ERR  #url或component_type键名有误
            continue
        # 如果读取url图像出现异常，跳过
        image = get_image(image_url)
        if isinstance(image, Exception):
            getinfo[data] = GetTrainDataStatus.READ_URL_ERR     #读取url图像失败
            continue
        valStatus = singleVal(image_url, component_type)
        # 根据type保存图像
        if valStatus == ValidStatus.BOTH_COMPLIANT:
            getinfo[data] = GetTrainDataStatus.SUCCESS
            type_path = os.path.join(data_path,component_type)
            # dataPath = f"{rootpath}/{Train.dataDir}{component_type}"
            image = image.convert('RGB')
            image_id = component_type + '_' + str(uuid.uuid4())
            image.save(type_path + f"/{image_id}.jpg", "JPEG")
        else:
            getinfo[data] = valStatus
    # 判断是否全部训练数据都保存失败
    for i in getinfo:
        if getinfo[i] == GetTrainDataStatus.SUCCESS:
            count = count + 1
    return getinfo, count


@app.post(Url.getTrainData)
def getTrainData(item: GetTrainData):
    user_id = item.user_id
    dataset = item.dataset
    getinfo, count = getDataset(dataset)
    # 有数据新增成功，即返回成功
    if count > 0:
        get_data_response = {
            "code": ResponseCode.SUCCESS_GENERAL,
            "status": StatusCode.OK,
            "message": Message.SUCCESS,
            "payload": {
                "user_id": user_id,
                "get_info": getinfo
            }
        }
    # 没有数据新增成功，即返回失败
    else:
        get_data_response = {
            "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
            "status": StatusCode.BAD_REQUEST,
            "message": Message.FAILURE,
            "payload": {
                "user_id": user_id,
                "get_info": getinfo
            }
        }
    return get_data_response


@app.post(Url.predictText)
def predictText(item: PredictTextInput):
    text = {}
    for image_url in item.image_url_list:
        image = get_image_numpy(image_url)
        if isinstance(image, Exception):
            return {
                "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
                "status": StatusCode.BAD_REQUEST,
                "message": Message.FAILURE,
                "payload": {
                    "err": ErrorCode.READ_URL_FAIL,
                    "user_id": item.user_id
                }
            }
        # text[image_url] = get_text_info(reader, image)
        text[image_url] = get_line_height(get_text_info(reader, image))
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "code": ResponseCode.SUCCESS_GENERAL,
        "status": StatusCode.OK,
        "message": Message.SUCCESS,
        "payload": {
            "user_id": item.user_id,
            "text": text
        }
    }
# 获取文字行高
def get_line_height(text_info_list):
    # 存放封装了行高的最终返回结果
    new_text_info_list = []

    # 没有文字的空白图片直接返回
    if len(text_info_list) == 0:
        return new_text_info_list

    # 存放所有的段左上角坐标x
    left_top_coord_x_list = []
    # 存放所有段右下角坐标x
    right_bottom_coord_x_list = []
    left_top_coord_x = 9999
    right_bottom_coord_x = -1

    if len(text_info_list) > 0:
        # 存放暂时还不能确定行高的元素
        que = queue.Queue()
        # 记录前一个段的左上角坐标
        pre_left_top_y = -1
        # 记录前一个段的字高
        pre_height = -1
        # 记录前一个段的行高
        pre_line_height = -1
        for text_info in text_info_list:
            if que.empty():
                que.put(text_info)
                pre_left_top_y = text_info['left_top_coord'][1]
                pre_height = text_info['height']
            elif abs(text_info['left_top_coord'][1] - pre_left_top_y) <= pre_height:
                que.put(text_info)
            else:
                while not que.empty():
                    new_text_info = que.get()
                    new_text_info['line_height'] = text_info['left_top_coord'][1] - pre_left_top_y
                    pre_line_height = new_text_info['line_height']
                    new_text_info_list.append(new_text_info)

                    left_top_coord_x = min(left_top_coord_x, new_text_info['left_top_coord'][0])
                    right_bottom_coord_x = max(right_bottom_coord_x, new_text_info['left_top_coord'][0] + new_text_info['width'])

                left_top_coord_x_list.append(left_top_coord_x)
                right_bottom_coord_x_list.append(right_bottom_coord_x)
                left_top_coord_x = 9999
                right_bottom_coord_x = -1
                que.put(text_info)
                pre_left_top_y = text_info['left_top_coord'][1]
                pre_height = text_info['height']

    # 最后一行单独处理
    while not que.empty():
        new_text_info = que.get()
        if -1 == pre_line_height:
            new_text_info['line_height'] = new_text_info['height']
        else:
            new_text_info['line_height'] = pre_line_height
        new_text_info_list.append(new_text_info)

        left_top_coord_x = min(left_top_coord_x, new_text_info['left_top_coord'][0])
        right_bottom_coord_x = max(right_bottom_coord_x, new_text_info['left_top_coord'][0] + new_text_info['width'])

    left_top_coord_x_list.append(left_top_coord_x)
    right_bottom_coord_x_list.append(right_bottom_coord_x)

    sort_left_top_coord_x_list = sorted(left_top_coord_x_list)
    sort_right_bottom_coord_y_list = sorted(right_bottom_coord_x_list)

    align = {}
    if len(sort_left_top_coord_x_list) > 1:
        align_left = abs(sort_left_top_coord_x_list[-1] - sort_left_top_coord_x_list[0])
        align_right = abs(sort_right_bottom_coord_y_list[-1] - sort_right_bottom_coord_y_list[0])
        if align_left > 10 and align_right > 10:
            align['align'] = "center"
        elif align_left < align_right:
            align['align'] = "left"
        else:
            align['align'] = "right"
    else:
        align['align'] = "center"

    new_text_info_list.append(align)

    return new_text_info_list
# 获取图片中文字所有信息
def get_text_info(reader, image):
    text_info_list = []
    # 传入的image是Image对象
    if isinstance(image, Image.Image):
        image_numpy = np.asarray(image)
    # 传入的image是numpy对象
    if isinstance(image, np.ndarray):
        image_numpy = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    # 返回文字内容和文字区域的左上角坐标宽高
    content_coord_width_height_list = get_text_content(reader, image_numpy)
    if not content_coord_width_height_list:
        return text_info_list
    for info in content_coord_width_height_list:
        text_info = {}
        left_top_coord = info[0]
        text_info['left_top_coord'] = info[0]
        text_info['width'] = info[1]
        text_info['height'] = info[2]
        text_info['msg'] = info[3]
        right_down_coord = [left_top_coord[0] + info[1], left_top_coord[1] + info[2]]
        box = (left_top_coord[0], left_top_coord[1], right_down_coord[0], right_down_coord[1])
        image_crop = image.crop(box)
        # bg_color = get_background_color(image_crop)
        bg_color = get_edge_colors(image_crop)
        text_color = get_text_color(bg_color, image_crop)
        text_info['backgroundColor'] = bg_color
        text_info['color'] = text_color
        text_info_list.append(text_info)
    return text_info_list


# 获取文字的内容及区域信息
def get_text_content(reader, image_numpy):
    # 给图片加上宽为20像素的边框 增加ocr识别的准确率
    add_border_image_numpy = add_border(image_numpy)
    with ocrLock:
        # results = reader.readtext(image_numpy)

        # result = reader.ocr(image_numpy, cls=True)
        result = reader.ocr(add_border_image_numpy, cls=True)
        results = result[0]
    image_height, image_width, _ = image_numpy.shape
    # 文字内容、文字区域左上角坐标、区域宽高列表
    coord_width_height_content_list = []
    if results is not None:
        for result in results:
            # 获取四个角坐标
            coords = result[0]
            # 获取左上角和右下角坐标
            left_top_coords = coords[0]
            right_bottom_coords = coords[2]
            # 识别的文字框不是矩形框 排除
            # if left_top_coords[1] != coords[1][1]:
            #     continue
            # 需要将numpy.int32转化为int类型，否则在fastapi返回时会出现问题
            left_top_coords = [int(item) for item in left_top_coords]
            right_bottom_coords = [int(item) for item in right_bottom_coords]
            # 排除掉ocr识别效果差的坐标
            if left_top_coords[0] >= right_bottom_coords[0] or left_top_coords[1] >= right_bottom_coords[1] or \
                    left_top_coords[0] < 0 or left_top_coords[1] < 0:
                continue
            # 获取文字内容
            # text_content = result[1]
            text_content = result[1][0]
            # 使用列表推导式对每个值减去20
            left_top_coords = [max(x - 20, 0) for x in left_top_coords]
            right_bottom_coords = [max(x - 20, 0) for x in right_bottom_coords]
            # 求文字区域的宽和高。由于之前加了20像素边框可能导致右下角坐标偏大。在这里需要对宽高进行适当的调整，如宽高本身和左上角坐标加宽高不能超过原图的宽高
            width = min(image_width, int(right_bottom_coords[0] - left_top_coords[0]))
            height = min(image_width, int(right_bottom_coords[1] - left_top_coords[1]))
            if left_top_coords[0] + width > image_width:
                width = image_width - left_top_coords[0]
            if left_top_coords[1] + height > image_height:
                height = image_height - left_top_coords[1]
            coord_width_height_content_list.append((left_top_coords, width, height, text_content))
        return coord_width_height_content_list
    else:
        return None


def is_point_inside_rectangle(x, y, rect_x, rect_y, rect_width, rect_height):
    """
    判断点(x, y)是否在矩形框内。

    参数：
    x (float): 点的 x 坐标。
    y (float): 点的 y 坐标。
    rect_x (float): 矩形框的左上角 x 坐标。
    rect_y (float): 矩形框的左上角 y 坐标。
    rect_width (float): 矩形框的宽度。
    rect_height (float): 矩形框的高度。

    返回：
    bool: 如果点在矩形框内，返回True；否则返回False。
    """
    # 判断点是否在矩形框内
    if rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height:
        return True
    else:
        return False


# 获取输入Image对象图片的背景颜色，第二个参数为图片中子节点的信息，不统计子节点区域的像素颜色
def get_background_color(image, children_image_info=None):
    image = image.convert("RGB")
    if children_image_info is None:
        children_image_info = []
    # 获取图片的宽度和高度
    width, height = image.size
    # 统计每个像素点的颜色
    pixel_count = {}
    pixel_count_standby = {}
    # 判断是否需要跳过统计标志
    flag = False
    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if pixel in pixel_count_standby:
                pixel_count_standby[pixel] += 1
            else:
                pixel_count_standby[pixel] = 1
            for value in children_image_info:
                rect_x, rect_y = value['left_top_coord']
                rect_width = value['width']
                rect_height = value['height']
                if is_point_inside_rectangle(x, y, rect_x, rect_y, rect_width, rect_height):
                    flag = True
                    break
            if flag:
                flag = False
                continue
            if pixel in pixel_count:
                pixel_count[pixel] += 1
            else:
                pixel_count[pixel] = 1
    if pixel_count:
        # 找出现次数最多的颜色
        most_common_color = max(pixel_count, key=pixel_count.get)
    else:
        most_common_color = max(pixel_count_standby, key=pixel_count_standby.get)
    return most_common_color[0:3]

# 获取输入带有文字的图片边缘颜色作为文字背景颜色。
def get_edge_colors(image, edge_width=5):
    image = image.convert("RGB")
    # 转换为NumPy数组
    img_array = np.array(image)

    # 获取图像的形状
    height, width, _ = img_array.shape

    # 定义边缘的范围
    edge_range_x = range(0, edge_width)  # 左边缘
    edge_range_y_top = range(0, edge_width)  # 顶部边缘
    edge_range_y_bottom = range(max(0, height - edge_width), height)  # 底部边缘
    edge_range_x_right = range(max(0, width - edge_width), width)  # 右边缘
    # 获取边缘像素的RGB颜色
    edge_colors = [tuple(img_array[y, x]) for x in edge_range_x for y in range(height)] + \
                  [tuple(img_array[y, x]) for x in range(width) for y in edge_range_y_top] + \
                  [tuple(img_array[y, x]) for x in range(width) for y in edge_range_y_bottom] + \
                  [tuple(img_array[y, x]) for x in edge_range_x_right for y in range(height)]
    # 使用Counter计算RGB元组的出现次数
    color_counter = Counter(edge_colors)
    # 找到出现次数最多的RGB元组
    most_common_color = color_counter.most_common(1)[0][0]
    most_common_color = [int(item) for item in most_common_color]
    return most_common_color[0:3]
# 获取图片中文字的rgb颜色
def get_text_color(BG_color, image):
    image = image.convert("RGB")
    image_np = np.array(image)
    gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray_BG=0.2989*BG_color[0]+0.5870*BG_color[1]+0.1140*BG_color[2]
    # cv2.imshow("",gray_image_np)
    # cv2.waitKey(0)
    retVal, a_img = cv2.threshold(gray_image_np, 0, 255, cv2.THRESH_OTSU)
    # print("使用opencv函数的方法：" + str(retVal))
    # cv2.imshow("a_img", a_img)
    # cv2.waitKey(0)
    if gray_BG > retVal:
        wz_color = 0
        # print("灰度图中文字为黑色")
    else:
        wz_color = 255
        # print("灰度图中文字为白色")
    # 初始化一个空列表
    wz_pixels = []
    # 遍历所有像素点并存储文字像素点的坐标
    for i in range(gray_image_np.shape[0]):
        for j in range(gray_image_np.shape[1]):
            if a_img[i, j] == wz_color:
                wz_pixels.append((j, i))
    # 输出文字像素点的坐标列表
    # 统计每个像素点的颜色
    pixel_count = {}
    for coord in wz_pixels:
        x = coord[0]
        y = coord[1]
        pixel = image.getpixel((x, y))
        if pixel in pixel_count:
            pixel_count[pixel] += 1
        else:
            pixel_count[pixel] = 1
    most_common_color = max(pixel_count, key=pixel_count.get)
    return most_common_color[0:3]

def get_direction(image):
    image_numpy = np.array(image)
    height, width, _ = image_numpy.shape
    if height > width:
        return "vertical"
    else:
        return "horizontal"


# 组件颜色b
def get_component_color(image):
    bg_color = get_background_color(image)
    component_color = get_text_color(bg_color, image)
    return component_color


def get_radio_coord_width_height_content_list(reader, image):
    image_numpy = np.array(image)
    # 给图片加上宽为20像素的边框 增加ocr识别的准确率
    add_border_image_numpy = add_border(image_numpy)
    with ocrLock:
        #  results = reader.readtext(image_numpy)
        # result = reader.ocr(image_numpy, cls=True)
        result = reader.ocr(add_border_image_numpy, cls=True)
        results = result[0]
    # 每个元素为选项文字区域的左上角坐标、宽高和文字内容()
    group_radio_info_list = []
    # 右下角x坐标
    right_bottom_x = 0
    # 右下角y坐标
    right_bottom_y = 0
    # 文字类容
    final_text_content = ""
    if results is not None:
        left_top_coords = [int(item) for item in results[0][0][0]]
        for i in range(len(results)):
            group_radio_info = {}
            coords = results[i][0]
            # single_text_content = results[i][1]
            single_text_content = results[i][1][0]
            if i + 1 < len(results):
                # next_single_text_content = results[i + 1][1]
                next_single_text_content = results[i + 1][1][0]

            if 'A' <= single_text_content[0] and single_text_content[0] <= 'Z':
                left_top_coords = [int(item) for item in coords[0]]
                final_text_content = single_text_content
            elif not ('A' <= single_text_content[0] and single_text_content[0] <= 'Z'):
                if right_bottom_x < int(coords[2][0]):
                    right_bottom_x = int(coords[2][0])

                if right_bottom_y < int(coords[2][1]):
                    right_bottom_y = int(coords[2][1])

                final_text_content = final_text_content + " " + single_text_content

            if (i + 1 == len(results)) or ('A' <= next_single_text_content[0] and next_single_text_content[0] <= 'Z'):
                width = right_bottom_x - left_top_coords[0]
                height = right_bottom_y - left_top_coords[1]
                right_bottom_x = -1
                right_bottom_y = -1
                if width < 0 or height < 0:
                    return None
                # 使用列表推导式对每个值减去20
                left_top_coords = [max(x - 20, 0) for x in left_top_coords]
                group_radio_info['left_top_coord'] = left_top_coords
                group_radio_info['width'] = width
                group_radio_info['height'] = height
                group_radio_info['msg'] = final_text_content
                group_radio_info_list.append(group_radio_info)

    return group_radio_info_list


@app.post(Url.GET_RADIO_TEXT_INFO)
def getRadioCoordWidthHeightContentList(item: PredictTextInput):
    radio_text_info = {}
    for image_url in item.image_url_list:
        image = get_image(image_url)
        if isinstance(image, Exception):
            return {
                "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
                "status": StatusCode.BAD_REQUEST,
                "message": Message.FAILURE,
                "payload": {
                    "err": ErrorCode.READ_URL_FAIL,
                    "user_id": item.user_id,
                    "err_url": image_url
                }
            }
        radio_text_info[image_url] = get_radio_coord_width_height_content_list(reader, image)
    return {
        "code": ResponseCode.SUCCESS_GENERAL,
        "status": StatusCode.OK,
        "message": Message.SUCCESS,
        "payload": {
            "user_id": item.user_id,
            "text": radio_text_info
        }
    }


# 获取表头（第一行）文字信息，包括文字框坐标宽高、字段列长度、颜色等
def get_table_header_info(reader, image):
    table_header_info_list = []
    image_numpy = np.array(image)
    # 给图片加上宽为20像素的边框 增加ocr识别的准确率
    add_border_image_numpy = add_border(image_numpy)
    with ocrLock:
        # results = reader.readtext(image_numpy)
        # result = reader.ocr(image_numpy, cls=True)
        result = reader.ocr(add_border_image_numpy, cls=True)
        results = result[0]
    image_height, image_width, _ = image_numpy.shape
    # 加上20 便于计算行列宽（由于给图片加了边框，得到的坐标普遍加了20的值）
    image_width = image_width + 20
    if results is not None:
        # 获取左上角和右下角坐标
        left_top_coord = results[0][0][0]
        right_bottom_coord = results[0][0][2]
        header_height = int(right_bottom_coord[1]-left_top_coord[1])
        # print(header_height)
        for i in range(len(results)):
            # 第一行（表头）是否结束的标志
            flag = False
            table_header_info = {}
            # text_content = results[i][1]
            text_content = results[i][1][0]
            # 获取文字框的四个角的坐标
            coords = results[i][0]
            # 获取文字框的左上角坐标和右下角坐标
            left_top_coord = coords[0]
            left_top_coord = [int(item) for item in left_top_coord]
            right_bottom_coord = coords[2]
            right_bottom_coord = [int(item) for item in right_bottom_coord]
            if i+1 < len(results):
                next_coords = results[i+1][0]
                next_left_top_coord = next_coords[0]
                if next_left_top_coord[1] - left_top_coord[1] < header_height:
                    field_width = int(next_left_top_coord[0] - left_top_coord[0])
                else:
                    field_width = image_width - left_top_coord[0]
                    flag = True
            else:
                field_width = image_width - left_top_coord[0]


            # 使用列表推导式对每个值减去20
            left_top_coord = [max(x - 20, 0) for x in left_top_coord]
            right_bottom_coord = [max(x - 20, 0) for x in right_bottom_coord]
            table_header_info['left_top_coord'] = left_top_coord
            table_header_info['width'] = min(image_width, int(coords[2][0]-coords[0][0]))
            table_header_info['height'] = min(image_height, int(coords[2][1]-coords[0][1]))
            table_header_info['field_width'] = field_width
            table_header_info['msg'] = text_content
            # 分割出文字框的图片准备求颜色
            box = (left_top_coord[0], left_top_coord[1], right_bottom_coord[0], right_bottom_coord[1])
            image_crop = image.crop(box)
            # bg_color = get_background_color(image_crop)
            bg_color = get_edge_colors(image_crop)
            text_color = get_text_color(bg_color, image_crop)
            table_header_info['backgroundColor'] = bg_color
            table_header_info['color'] = text_color
            table_header_info_list.append(table_header_info)
            if flag:
                break
    return table_header_info_list




@app.post(Url.GET_TABLE_HEADER_INFO)
def getTableHeaderInfo(item: PredictTextInput):
    table_header_text_info = {}
    # 是否排序和是否有单、复选框
    table_header_other_info = {}
    for image_url in item.image_url_list:
        image = get_image(image_url)
        if isinstance(image, Exception):
            return {
                "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
                "status": StatusCode.BAD_REQUEST,
                "message": Message.FAILURE,
                "payload": {
                    "err": ErrorCode.READ_URL_FAIL,
                    "user_id": item.user_id,
                    "err_url": image_url
                }
            }
        table_header_text_info[image_url] = get_table_header_info(reader, image)
        # table_header_other_info[image_url] = get_table_other_info(reader, image)
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "code": ResponseCode.SUCCESS_GENERAL,
        "status": StatusCode.OK,
        "message": Message.SUCCESS,
        "payload": {
            "user_id": item.user_id,
            "table_header_info": table_header_text_info,
            # "other_info": table_header_other_info
        }
    }



# 根据向量计算角度
def calculate_angle(vertex1, vertex2, vertex3):
    # 计算两个向量
    vector1 = np.array([vertex1[0] - vertex2[0], vertex1[1] - vertex2[1]])
    vector2 = np.array([vertex3[0] - vertex2[0], vertex3[1] - vertex2[1]])

    # 计算两个向量的长度
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)

    # 计算两个向量的夹角（弧度）
    dot_product = np.dot(vector1, vector2)
    angle_rad = np.arccos(dot_product / (length1 * length2))

    # 将弧度转换为角度
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# 检测图片是否有圆角
def has_rounded_corners(image, epsilon_threshold=0.0105, angle_threshold=90):
    has_flag = False
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为灰度图
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height,width,_ = image.shape
    # 边缘检测
    edges = cv2.Canny(gray, 10, 100)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到具有最大周长的轮廓
    max_perimeter = 0
    max_contour = None

    for contour in contours:
        # 求周长
        perimeter = cv2.arcLength(contour, True)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            max_contour = contour
    # 图片无轮廓返回无圆角
    if max_contour is None:
        return False, 0
    # 多边形逼近
    epsilon = epsilon_threshold * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    num_vertices = len(approx)
    # 初始化圆角高度
    round_corner_height = 0
    for i in range(num_vertices):
        prev_vertex = approx[i][0]  # 前一个顶点
        curr_vertex = approx[(i + 1) % num_vertices][0]  # 当前顶点
        next_vertex = approx[(i + 2) % num_vertices][0]  # 后一个顶点
        # 计算角度并进行判断
        angle = calculate_angle(prev_vertex, curr_vertex, next_vertex)
        if angle > angle_threshold + 10:
            has_flag = True
            round_corner_height = get_round_corner_height(approx)
            break
    return has_flag, round_corner_height
def get_round_corner_height(approx):
    point_list = [point[0] for point in approx]
    # 初始化最左、最上、最下和最右的点
    leftmost_point = point_list[0]
    topmost_point = point_list[0]
    bottommost_point = point_list[0]
    rightmost_point = point_list[0]
    # 遍历坐标列表，找到最左、最上、最下和最右的点
    for point in point_list:
        x, y = point

        # 更新最左的点
        if x < leftmost_point[0]:
            leftmost_point = point

        # 更新最上的点
        if y < topmost_point[1]:
            topmost_point = point

        # 更新最下的点
        if y > bottommost_point[1]:
            bottommost_point = point

        # 更新最右的点
        if x > rightmost_point[0]:
            rightmost_point = point
    # 圆角高度取最左点y与最高最低点y的差值中的较小值
    height = min(abs(leftmost_point[1] - topmost_point[1]), abs(leftmost_point[1] - bottommost_point[1]))
    return int(height)


def get_avg_border_color(image):
    if not isinstance(image, np.ndarray):
        # 如果图像不是NumPy数组，将其转换为NumPy数组
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 高斯模糊，用于降低噪声
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny边缘检测
    edged = cv2.Canny(gray, 10, 100)

    # 查找边缘检测后的图像中的轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    r = 0
    g = 0
    b = 0

    # 求周长
    max_perimeter = 0
    max_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            max_contour = contour
    # 图片无轮廓返回空
    if max_contour is None:
        return None
    for row in max_contour:
        for element in row:
            if 0 == r and 0 == g and 0 == b:
                b, g, r = image[element[1], element[0]]
            else:
                tmp_b, tmp_g, tmp_r = image[element[1], element[0]]
                b = int(b / 2 + tmp_b / 2)
                g = int(g / 2 + tmp_g / 2)
                r = int(r / 2 + tmp_r / 2)
    rgb_list = [r, g, b]
    # 需要将numpy.int32转化为int类型，否则在fastapi返回时会出现问题
    rgb_list = [int(item) for item in rgb_list]
    return rgb_list


@app.post(Url.GET_BORDER_COLOR_AND_ROUND_HEIGHT)
def getBorderColorAndRoundHeight(item: PredictTextInput):
    border_color = {}
    round_corner_flag_and_height = {}
    for image_url in item.image_url_list:
        image = get_image_numpy(image_url)
        if isinstance(image, Exception):
            return {
                "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
                "status": StatusCode.BAD_REQUEST,
                "message": Message.FAILURE,
                "payload": {
                    "err": ErrorCode.READ_URL_FAIL,
                    "user_id": item.user_id
                }
            }
        # text[image_url] = get_text_info(reader, image)
        border_color[image_url] = get_avg_border_color(image)
        round_corner_flag, round_corner_height = has_rounded_corners(image)
        round_corner_flag_and_height[image_url] = {"round_corner_flag": round_corner_flag, "round_corner_height": round_corner_height}
    return {
        "code": ResponseCode.SUCCESS_GENERAL,
        "status": StatusCode.OK,
        "message": Message.SUCCESS,
        "payload": {
            "user_id": item.user_id,
            "border_color": border_color,
            "round_corner_flag_and_height": round_corner_flag_and_height
        }
    }


@app.post(Url.GET_COMPONENT_COLOR)
def getComponentColor(item: PredictTextInput):
    component_color = {}
    background_color = {}
    for image_url in item.image_url_list:
        image = get_image(image_url)
        if isinstance(image, Exception):
            return {
                "code": ResponseCode.REQUEST_PARAMS_NOT_MATCH,
                "status": StatusCode.BAD_REQUEST,
                "message": Message.FAILURE,
                "payload": {
                    "err": ErrorCode.READ_URL_FAIL,
                    "user_id": item.user_id
                }
            }
        # text[image_url] = get_text_info(reader, image)
        component_color[image_url] = get_component_color(image)
        background_color[image_url] = get_background_color(image)
    return {
        "code": ResponseCode.SUCCESS_GENERAL,
        "status": StatusCode.OK,
        "message": Message.SUCCESS,
        "payload": {
            "user_id": item.user_id,
            "component_color": component_color,
            "background_color": background_color,
        }
    }


# 获取日期选择器的类型
def get_datepicker_mode(text_infos):
    mode = ""
    # 文字内容
    msg = ""
    if len(text_infos) > 0:
        for text_info in text_infos:
            msg = msg + text_info['msg']
        if "年" in msg:
            mode = "year"
        if "季度" in msg:
            mode = "quarter"
        if "月" in msg:
            mode = "month"
        if "周" in msg:
            mode = "week"
        if "日" in msg:
            mode = "day"
    return mode
# 判断是否是时间范围选择器
def get_is_time_picker(text_info_list):
    # 文本识别结果不是两段, 认为不是范围时间选择器
    if 2 != len(text_info_list):
        return False

    for text_info in text_info_list:
        str_list = text_info['msg'].split(':')
        if 1 == len(str_list):
            str_list = text_info['msg'].split(".")
        for s in str_list:
            if 2 != len(s) or (not s.isdigit()):
                return False

    return True

# 判断是否是日期区间
def get_is_date_picker(text_info_list):
    if len(text_info_list) != 2:
        return False

    start_flag = 0
    end_flag = 0
    for text_info in text_info_list:
        if "开始" in text_info['msg']:
            start_flag = start_flag + 1
        elif "结束" in text_info['msg']:
            end_flag = end_flag + 1

    if 1 == start_flag and 1 == end_flag:
        return True
    else:
        return False


if __name__ == '__main__':
    uvicorn.run(app, host=Url.ip, port=args.port)
