from pydantic import BaseModel
from typing import Optional


# 开始ResNet推理接口输入类
class ResnetStartItem(BaseModel):
    image_url_list: list
    model_id: Optional[str] = None
    topic_id: str
    user_id: str
    input_component_type: Optional[str] = ""


# 开始SAM推理输入类
class SamStartItem(BaseModel):
    image_url: str
    prompt: dict = {
        "input_point": [],
        "input_label": [],
        "input_box": []
    }
    topic_id: str
    user_id: str
    is_predict_type: Optional[bool] = False
    resnet_model_id: Optional[str] = None
    read_text_component_list: Optional[list] = []
    cut_outline_flag: Optional[bool] = False


# 开始SAM推理输入类
class SamStartItem2(BaseModel):
    image_url: str
    prompt: dict = {
        "input_point": [],
        "input_label": [],
    }
    select_box: Optional[list] = []
    topic_id: str
    user_id: str
    is_predict_type: Optional[bool] = False
    resnet_model_id: Optional[str] = None
    read_text_component_list: Optional[list] = []
    cut_outline_flag: Optional[bool] = False


# 开始SAM推理输入类
class SamStartItem3(BaseModel):
    image_url: str
    points: list
    prompt: dict = {
        "input_point": [],
        "input_label": [],
        #"input_box": []
    }
    topic_id: str
    user_id: str
    is_predict_type: Optional[bool] = False
    resnet_model_id: Optional[str] = None
    read_text_component_list: Optional[list] = []


# （SAM和ResNet）推理状态查询和终止的输入类
class ProcessSKipItem(BaseModel):
    task_id: str
    topic_id: str
    user_id: str


# 训练开始接口
class TrainStartItem(BaseModel):
    user_id: str
    task_type: Optional[str] = "train"
    train_params: dict = {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001
    }


# 训练过程跳过接口
class TrainProcessSkipItem(BaseModel):
    train_task_id: str
    user_id: str
    task_type: Optional[str] = "train"


# 获得训练数据接口
class GetTrainData(BaseModel):
    user_id: Optional[str] = "admin"
    dataset: dict


# 预测文本输入接口
class PredictTextInput(BaseModel):
    user_id: Optional[str] = "admin"
    image_url_list: list
