import json
import os
import shutil
import uuid
from io import BytesIO
from zipfile import ZipFile
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import requests
from imgseg_config import *
from config import Train
import matplotlib.pyplot as plt

TRAIN_API_URL_START = Url.TRAIN_API_URL_START
TRAIN_API_URL_PROGRESS = Url.TRAIN_API_URL_PROGRESS
TRAIN_API_URL_SKIP = Url.TRAIN_API_URL_SKIP
ADD_SINGLE_DATA = Url.ADD_SINGLE_DATA
rootpath = os.getcwd()


datasetPath = os.path.join(rootpath, Train.dataDir)
trainInfoPath = os.path.join(rootpath, "trainInfo")
def taskStatus(task_status):
    if task_status == TaskStatus.NORMAL_END:
        return "训练正常结束"
    elif task_status == TaskStatus.IN_PROGRESS:
        return "正在训练中..."
    elif task_status == TaskStatus.USER_TERMINATED:
        return "训练被用户手动终止"
    else:
        return "训练异常终止!"


def errcodeToText(err):
    if err == ErrorCode.CREATE_TRAIN_FAIL:
        return "当前有训练任务正在训练,创建训练任务失败"
    elif err == ErrorCode.TASK_ID_NOT_FOUND:
        return "任务ID不存在"


def plotBarImage(x, y):
    labels = x
    singleAcc = y
    if len(labels) <= 3:
        plt.figure(figsize=(3, 8))
    else:
        plt.figure(figsize=(len(labels), 10))
    plt.bar(labels, singleAcc)
    plt.xlabel('label')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('acc')
    for i in range(len(labels)):
        plt.text(i, singleAcc[i] + 0.01, f'{singleAcc[i]:.2f}', ha='center', va='bottom')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_variable = buffer.read()
    plt.close()
    # 将 image_variable 转换为 PIL Image 对象
    Bar_image = Image.open(BytesIO(image_variable))
    return Bar_image


def plotLineImage(x, y1, y2):
    epoch = x
    valAccList = y1
    valLossList = y2
    epochs = list(i for i in range(1, epoch+1))
    plt.plot(epochs, valAccList, label='valAcc')  # 绘制第一条折线
    plt.plot(epochs, valLossList, label='valLoss')  # 绘制第二条折线
    plt.xlabel('epoch')
    plt.ylabel('acc and loss')
    plt.legend()  # 显示图例
    # 将Matplotlib图像转换为PIL Image对象
    buffer = BytesIO()  # 创建一个BytesIO对象
    plt.savefig(buffer, format='png')  # 将图像保存到BytesIO对象
    buffer.seek(0)  # 将文件指针移至开头
    Line_image = Image.open(buffer)  # 通过PIL打开图像数据
    # 关闭图表
    plt.close()
    return Line_image
def train_start(epochs, batch_size, learning_rate):
    headers = {"Content-Type": "application/json"}
    data = {
        "user_id": "admin",
        "task_type": "train",
        "train_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    }
    response = requests.post(TRAIN_API_URL_START, headers=headers, data=json.dumps(data), timeout=600)
    output_data = response.json()
    data = output_data.get("payload")
    train_task_id = data.get("train_task_id")
    task_status = data.get("task_status", "创建训练任务失败!")
    if task_status == TaskStatus.IN_PROGRESS:
        task_status = '创建训练任务成功！'
    # print(task_status)
    err = data.get("err", "")
    if "err" in data.keys():
        err = errcodeToText(err)
    return train_task_id, task_status, err, gr.update(
                choices=[task_id.split(".")[0] for task_id in os.listdir(trainInfoPath)]), gr.update(
                choices=[task_id.split(".")[0] for task_id in os.listdir(trainInfoPath)])


def train_progress(train_task_id):
    headers = {"Content-Type": "application/json"}
    data = {
        "train_task_id": train_task_id,
        "user_id": "admin",
        "task_type": "train"
    }
    response = requests.post(TRAIN_API_URL_PROGRESS, headers=headers, data=json.dumps(data), timeout=600)
    output_data = response.json()
    data = output_data.get("payload")
    task_status = data.get("task_status", "查询训练进度失败!")
    if "task_status" in data.keys():
        task_status = taskStatus(task_status)
    epoch = data.get("epoch", "")
    modelEpoch = data.get("modelEpoch", "")
    acc = data.get("acc", 0)
    loss = data.get("loss", "无穷大")
    err = data.get("err", "")
    valAccList = data.get("valAccList", [])
    valLossList = data.get("valLossList", [])
    labels = data.get("labels", [])
    singleAcc = data.get("singleAcc", [])
    if "err" in data.keys():
        err = errcodeToText(err)
    dataset_num = data.get("dataset_num", 0)
    trainInfo = {
        "task_status": task_status,
        "err": err,
        "epoch": epoch,
    }
    trainResult = {
        "dataset_num": dataset_num,
        "acc": acc,
        "loss": loss,
        "model_epoch": modelEpoch
    }
    if len(labels) > 0:
        Bar_image = plotBarImage(labels, singleAcc)
    else:
        Bar_image = None
    if len(valAccList) > 0:
        Line_image = plotLineImage(epoch, valAccList, valLossList)
    else:
        Line_image = None
    return trainInfo, trainResult, Bar_image, Line_image
    # return task_status, err, dataset_num, epoch, acc, loss, barPath


# 训练终止
def train_skip(train_task_id):
    headers = {"Content-Type": "application/json"}
    data = {
        "train_task_id": train_task_id,
        "user_id": "admin",
        "task_type": "train"
    }
    response = requests.post(TRAIN_API_URL_SKIP, headers=headers, data=json.dumps(data), timeout=600)
    output_data = response.json()
    data = output_data.get("payload")
    task_status = data.get("task_status", "终止训练任务失败!")
    if task_status == TaskStatus.NORMAL_END:
        task_status = "训练已正常结束"
    elif task_status == TaskStatus.USER_TERMINATED:
        task_status = "终止训练任务成功！"
    err = data.get("err", "")
    if "err" in data.keys():
        err = errcodeToText(err)
    return task_status, err





def updateTypeDir():
    return gr.Dropdown.update(choices=os.listdir(datasetPath), label="组件文件夹", value="请选择组件文件夹",
                              min_width=400)

def updateImageList(type_id):
    image_path = os.path.join(Train.dataDir, type_id)
    image_list = os.listdir(image_path)
    return gr.Dropdown.update(choices=image_list, label=type_id + "图片文件夹", value="请选择图片ID"), \
        gr.Textbox.update(value=len(image_list), label=type_id + "图片数量"), None, gr.Button.update(value="下载该组件文件夹", visible=True), \
        gr.Button.update(value="删除该组件文件夹", visible=True), gr.update(visible=False)


def downloadFiles(type_id):
    if type_id not in os.listdir(datasetPath):
        return None, None, gr.update(value="下载失败！请选择文件夹后再点击")
    image_path = os.path.join(Train.dataDir, type_id)
    image_list = os.listdir(image_path)
    filesPath = []
    for image_id in image_list:
        file = os.path.join(image_path, image_id)
        filesPath.append(file)
    with ZipFile("tmp.zip", "w") as zipObj:
        for filePath in filesPath:
            zipObj.write(filePath, os.path.basename(filePath))
    return "tmp.zip", gr.File.update(visible=True), gr.update(value="下载成功！选择文件夹后可继续下载")


def show_image(type_id, image_id):
    if type_id not in os.listdir(datasetPath):
        return
    imagePath = os.path.join(datasetPath, type_id, image_id)
    try:
        image = Image.open(imagePath)
        if image.height >=500:
            height = 500
        else:
            height =image.height
        return imagePath, gr.Image.update(height=height, width=image.width+150), gr.Button.update(value="删除该图片", visible=True)
    except Exception:
        return None, gr.update(height=200), gr.Button.update(value="该文件不是图片需要删除，点击删除该文件", visible=True)


def deleteDir(type_id):
    typePath = os.path.join(datasetPath, type_id)
    shutil.rmtree(typePath)
    return gr.Dropdown.update(choices=os.listdir(datasetPath), label="组件文件夹", value="请选择组件文件夹",
                              min_width=400), \
        gr.Dropdown.update(choices=[], label="图片文件夹", value="请选择图片ID", min_width=400), gr.Textbox.update(
        label="图片数量", value=None), gr.Button.update(value="删除成功！选择文件夹后可继续删除", visible=False), gr.Button.update(visible=False), gr.Button.update(visible=False)


def deleteSingleImage(type_id, image_id):
    delete_path = os.path.join(datasetPath, type_id, image_id)
    try:
        os.remove(delete_path)
    except Exception:
        shutil.rmtree(delete_path)
    image_path = os.path.join(datasetPath, type_id)
    image_list = os.listdir(image_path)
    return gr.Dropdown.update(choices=image_list, label=type_id + "图片文件夹", value="请选择图片ID"), \
        gr.Textbox.update(value=len(image_list), label=type_id + "图片数量"), None, gr.Button.update(visible=False), gr.Button.update(visible=False), gr.Button.update(visible=False)




def confirmDelete():
    return confirmButton.update(visible=True), cancelButton.update(visible=True), deleteSingleButton.update(visible=False)

def cancelDelete():
    return confirmButton.update(visible=False), cancelButton.update(visible=False), deleteSingleButton.update(visible=True)

def confirmDirDelete():
    return confirmDirButton.update(visible=True), cancelDirButton.update(visible=True), deleteDirButton.update(visible=False)


def cancelDirDelete():
    return confirmDirButton.update(visible=False), cancelDirButton.update(visible=False), deleteDirButton.update(visible=True)
def add1(dataJsonStr):
    if not dataJsonStr:
        return None, gr.update(value="新增数据失败！请输入数据集后再点击")
    headers = {"Content-Type": "application/json"}
    try:
        datasetInfo = json.loads(dataJsonStr)
    except Exception:
        return None, gr.update(value="新增数据失败！请根据示例输入正确的数据集")
    data = {
        "dataset": datasetInfo
    }
    response = requests.post(ADD_SINGLE_DATA, headers=headers, data=json.dumps(data),
                             timeout=600)
    output_data = response.json()
    print(output_data)
    payload = output_data.get("payload", {})
    try:
        add_info = payload['get_info']
    except Exception:
        return None, gr.update(value="新增数据失败！请根据示例输入正确的数据集")
    for info in add_info:
        if add_info[info] == GetTrainDataStatus.SUCCESS:
            add_info[info] = "新增成功"
        elif add_info[info] == GetTrainDataStatus.READ_URL_ERR:
            add_info[info] = "读取url上的图片失败"
        elif add_info[info] == GetTrainDataStatus.URL_OR_TYPE_ERR:
            add_info[info] = "dataset中url或component_type键名称错误"
        elif add_info[info] == ValidStatus.URL_NON_COMPLIANT:
            add_info[info] = "图片过大或过小"
        elif add_info[info] == ValidStatus.TYPE_NON_COMPLIANT:
            add_info[info] = "component_type名称不合规，训练数据集中没有此component_type名称"
        else:
            add_info[info] = "图片过大或过小且component_type名称不合规"
    if output_data['message'] == Message.SUCCESS:
        return add_info, gr.Button.update("新增成功！")
    else:
        return add_info, gr.Button.update("新增失败！")


def add2(add_image, add_image_type):
    if add_image is None:
        return gr.update(value="新增数据失败！请上传图片后再点击")
    if not add_image_type:
        return gr.update(value="新增数据失败！请选择上传图片的组件类型后再点击")
    image_id = add_image_type + '_' + str(uuid.uuid4()) + '.jpg'
    type_path = os.path.join(datasetPath, add_image_type)
    image_path = os.path.join(type_path, image_id)
    # 需要转换RGB通道否则保存的图片为变色
    add_image = cv2.cvtColor(add_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, add_image)
    return gr.Button.update(value=f"新增成功！id为{image_id}")

def add3(add_image_zip, add_image_type):
    file_type_list = ['zip']
    if add_image_zip is None:
        return gr.update(value="新增数据失败！请上传图片zip后再点击")
    if not add_image_type:
        return gr.update(value="新增数据失败！请选择上传图片的组件类型后再点击")
    file_type = add_image_zip.name.split('.')[-1]
    if file_type not in file_type_list:
        return gr.update(value="新增数据失败！请上传正确的压缩包类型")
    type_path = os.path.join(datasetPath, add_image_type)
    flag = False
    if file_type == "zip":
        try:
            with ZipFile(add_image_zip.name) as zfile:
                for info in zfile.infolist():
                    if '/' not in info.filename and (info.filename.endswith('.jpg') or info.filename.endswith('.png') or info.filename.endswith('.jpeg') or info.filename.endswith('.webp') or info.filename.endswith('.JPG') or info.filename.endswith('.PNG') or info.filename.endswith('.JPEG') or info.filename.endswith('.WEBQ')):
                        zfile.extract(info, type_path)
                        flag = True
        except Exception:
            return gr.Button.update(value="新增数据失败！解压发生了错误")
    if not flag:
        return gr.Button.update(value="新增失败！请上传正确的文件夹结构（zip中没有文件夹，仅包含图片）")
    return gr.Button.update(value="新增成功！")

def addTypeDir(type_id):
    if not type_id:
        return gr.Button.update("请先输入需要新增的组件名称")
    addpath = os.path.join(datasetPath, type_id)
    try:
        os.mkdir(addpath)
        return gr.Button.update("新增成功！")
    except Exception:
        return gr.Button.update("文件夹已存在！")




with gr.Blocks() as demo:
    gr.Markdown("# 图分割后台界面")
    with gr.Tab("数据管理"):
        gr.Markdown("#### 训练数据管理界面")
        with gr.Tab("数据查询和删除"):
            with gr.Row():  # 并行显示，可开多列
                with gr.Column():  # 并列显示，可开多行
                    checkStart = gr.Button("更新组件文件夹列表开始查询")
                    with gr.Row():
                        type_dir = gr.Dropdown(choices=[], label="组件文件夹", value="请选择组件文件夹", min_width=300)
                        downloadButton = gr.Button("下载该组件文件夹", min_width=100)
                        deleteDirButton = gr.Button("删除该组件文件夹", min_width=100, visible=False)
                        confirmDirButton = gr.Button("确认删除", min_width=100, visible=False)
                        cancelDirButton = gr.Button("取消删除", min_width=100, visible=False)
                    with gr.Row():
                        image_dir = gr.Dropdown(choices=[], label="图片文件夹", value="请选择图片ID", min_width=400)
                        image_num = gr.Textbox(label="图片数量", min_width=50)
                with gr.Column():
                    image_zip = gr.File(label="组件文件夹zip压缩包", visible=False)
                    image = gr.Image(label="图片")
                    deleteSingleButton = gr.Button("删除该图片", visible=False, min_width=100)
                    with gr.Row():
                        confirmButton = gr.Button("确认", visible=False, min_width=100)
                        cancelButton = gr.Button("取消", visible=False, min_width=100)
            checkStart.click(fn=updateTypeDir, outputs=type_dir)
            type_dir.select(fn=updateImageList, inputs=type_dir,
                            outputs=[image_dir, image_num, image, downloadButton, deleteDirButton, deleteSingleButton])
            image_dir.select(fn=show_image, inputs=[type_dir, image_dir], outputs=[image, image, deleteSingleButton])
            downloadButton.click(fn=downloadFiles, inputs=type_dir, outputs=[image_zip, image_zip, downloadButton])

            deleteDirButton.click(fn=confirmDirDelete, inputs=[], outputs=[confirmDirButton, cancelDirButton, deleteDirButton])
            confirmDirButton.click(fn=deleteDir, inputs=type_dir,
                                  outputs=[type_dir, image_dir, image_num, deleteDirButton, confirmDirButton, cancelDirButton])
            cancelDirButton.click(fn=cancelDirDelete, inputs=[], outputs=[confirmDirButton, cancelDirButton, deleteDirButton])
            deleteSingleButton.click(fn=confirmDelete, inputs=[], outputs=[confirmButton, cancelButton, deleteSingleButton])
            confirmButton.click(fn=deleteSingleImage, inputs=[type_dir, image_dir],
                                     outputs=[image_dir, image_num, image, deleteSingleButton, confirmButton, cancelButton])
            cancelButton.click(fn=cancelDelete, inputs=[], outputs=[confirmButton, cancelButton, deleteSingleButton])

            with gr.Row():
                with gr.Tab("新增训练数据"):
                    with gr.Tab("输入数据集字典新增训练数据"):
                        with gr.Row():
                            with gr.Column():
                                addDataExample = gr.Textbox(lines=2, label="输入数据集格式示例", interactive=False,
                                                            show_copy_button=True, value=
                                                            """{
"data_1":{"url": "url1","component_type": "type1"},
"data_2": {"url": "url2","component_type": "type2"}
}""")
                                addDataJson = gr.Textbox(lines=10, label="数据集字典")
                                add1_button = gr.Button("新增")
                            with gr.Column():
                                addInfo1 = gr.JSON(label="新增数据情况")
                    addDataJson.change(fn=lambda: gr.Button.update("新增"), outputs=add1_button)
                    add1_button.click(fn=add1, inputs=addDataJson, outputs=[addInfo1, add1_button])
                    with gr.Tab("上传单个图片增加到训练数据集"):
                        with gr.Row():
                            with gr.Column():
                                addImage = gr.Image(label="图片")
                                with gr.Row():
                                    addImageType = gr.Dropdown(choices=[], label="上传图片的组件类型", min_width=500)
                                    getList1 = gr.Button("获取组件类型列表")
                                add2_button = gr.Button("新增")
                    getList1.click(fn=lambda: gr.update(choices=os.listdir(datasetPath)), outputs=addImageType)
                    addImage.clear(fn=lambda: gr.Button.update("新增"), outputs=add2_button)
                    add2_button.click(fn=add2, inputs=[addImage, addImageType], outputs=add2_button)
                    with gr.Tab("上传zip文件压缩包增加训练数据"):
                        with gr.Row():
                            with gr.Column():
                                addImageZip = gr.File(label="图片zip压缩包")
                                with gr.Row():
                                    addImageZipType = gr.Dropdown(choices=[], label="上传图片的组件类型", min_width=500)
                                    getList2 = gr.Button("获取组件类型列表")
                                add3_button = gr.Button("新增")
                    getList2.click(fn=lambda: gr.update(choices=os.listdir(datasetPath)), outputs=addImageZipType)
                    addImageZip.clear(fn=lambda: gr.Button.update("新增"), outputs=add3_button)
                    add3_button.click(fn=add3, inputs=[addImageZip, addImageZipType], outputs=add3_button)
                with gr.Tab("新增组件类型文件夹"):
                    with gr.Row():
                        addType = gr.Textbox(label="需要新增的组件类型名称")
                        addTypeButton = gr.Button("新增")
                addTypeButton.click(fn=addTypeDir, inputs=addType, outputs=addTypeButton)
                addType.change(fn=lambda: gr.Button.update("新增"), outputs=addTypeButton)
    with gr.Tab("模型训练"):
        gr.Markdown("#### 模型训练界面")
        with gr.Tab("开始训练"):
            with gr.Row():  # 并行显示，可开多列
                with gr.Column():  # 并列显示，可开多行
                    input1 = [gr.Slider(label="epochs", minimum=1, maximum=500, step=1, value=50,
                                        info="训练轮数,默认值50,取值范围1~500"),
                              gr.Slider(label="batch_size", minimum=1, maximum=64, step=1, value=8,
                                        info="模型训练的batch_size,推荐取2的n次方,默认值8,取值范围1~64"),
                              gr.Slider(label="learning_rate", minimum=0.000001, maximum=0.01, value=0.001,
                                        step=0.000001,
                                        info="模型训练的学习率,默认值0.001,取值范围0.000001~0.01")]
                    bottom1 = gr.Button("开始训练")
                with gr.Column():
                    startOut1 = gr.Textbox(lines=1, label="train_task_id", show_copy_button=True, info="创建的训练任务ID")
                    startOut2 = gr.Textbox(lines=1, label="create_info", info="是否成功创建训练任务")
                    startOut3 = gr.Textbox(lines=1, label="err", info="创建任务失败的原因，成功则为空。")

            with gr.Tab("训练进度查询"):
                with gr.Row(equal_height=False):  # 并行显示，可开多列
                    with gr.Column():  # 并列显示，可开多行
                        # input2 = [gr.Textbox(lines=1, label="train_task_id", info="输入需要查询的训练任务ID")]
                        with gr.Row():
                            input2 = gr.Dropdown(choices=[], label="train_task_id", info="选择需要查询的训练任务ID",
                                                 min_width=500)
                            getTaskIdListButton1 = gr.Button("获取任务id列表")
                        bottom2 = gr.Button("查询训练进度")
                        singleAccBar = gr.Image(label="当前保存模型验证集中的各组件准确率柱状图")
                    with gr.Column():
                        trainInfo = gr.JSON(label="训练状态信息(显示训练状态、错误信息和当前训练轮数)")
                        trainResult = gr.JSON(label="训练结果信息(显示数据集总量、当前模型的验证集准确率和平均损失以及当前模型是第几轮训练的模型)")
                        accLossLine = gr.Image(label="训练过程中验证集的准确率和平均损失折线图")

            with gr.Tab("训练任务终止"):
                with gr.Row(equal_height=False):  # 并行显示，可开多列
                    with gr.Column():  # 并列显示，可开多行
                        with gr.Row():
                            input3 = gr.Dropdown(choices=[], label="train_task_id", info="选择需要终止的训练任务ID",
                                                 min_width=500)
                            getTaskIdListButton2 = gr.Button("获取任务id列表")
                        bottom3 = gr.Button("终止训练任务")

                    with gr.Column():
                        out3 = [
                            gr.Textbox(lines=1, label="task_status", info="该训练任务所处的状态"),
                            gr.Textbox(lines=1, label="err", info="终止训练任务失败的原因，成功则为空")]

            bottom1.click(train_start, inputs=input1, outputs=[startOut1, startOut2, startOut3, input2, input3])
            # 获取trainInfo里的去掉json后缀的任务id
            getTaskIdListButton1.click(fn=lambda: gr.update(
                choices=[task_id.split(".")[0] for task_id in os.listdir(trainInfoPath)] if os.listdir(
                    trainInfoPath) else ["没有任务ID"]), outputs=input2)

            getTaskIdListButton2.click(fn=lambda: gr.update(
                choices=[task_id.split(".")[0] for task_id in os.listdir(trainInfoPath)] if os.listdir(
                    trainInfoPath) else ["没有任务ID"]), outputs=input3)
            bottom2.click(train_progress, inputs=input2, outputs=[trainInfo, trainResult, singleAccBar, accLossLine])
            bottom3.click(train_skip, inputs=input3, outputs=out3)

            input2.select(fn=lambda: [gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)], outputs=[trainInfo, trainResult, singleAccBar, accLossLine])
            input3.select(fn=lambda: [gr.update(value=None), gr.update(value=None)], outputs=out3)

demo.launch(server_name=Url.ip, server_port=Url.port_gradio)
# demo.launch()
