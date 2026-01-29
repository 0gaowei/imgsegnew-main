import requests
import json
def remove_base64_from_json(obj):
    """
    Recursively remove large base64 strings or keys named 'img_component_base64'
    to avoid printing huge base64 payloads in tests.
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if k == 'img_component_base64':
                new[k] = '<BASE64_REMOVED>'
            else:
                new[k] = remove_base64_from_json(v)
        return new
    if isinstance(obj, list):
        return [remove_base64_from_json(x) for x in obj]
    if isinstance(obj, str):
        # Heuristic: very long string consisting of base64 chars -> remove
        if len(obj) > 200 and all(c.isalnum() or c in '+/=\n\r' for c in obj):
            return '<BASE64_REMOVED>'
        return obj
    return obj

# 测试ResNet推理接口
def test_resnet_predict():
    url = "http://localhost:9002/imgSeg/v1/predictResnet"
    data = {
        "image_url_list": [
            "https://img.zcool.cn/community/0181445f0194b9a801215aa00b682d.png?x-oss-process=image/auto-orient,1/resize,m_lfit,w_1280,limit_1/sharpen,100"
        ],
        "model_id": "Train-imgSeg-52ddfff5-6e6e-4e83-955a-94783d83613a",
        "topic_id": "test123",
        "user_id": "user_test_01"
    }

    print("测试ResNet推理接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试SAM推理接口
def test_sam1_predict():
    url = "http://localhost:9002/imgSeg/v1/predictSam"
    data = {
        "image_url": "https://img.zcool.cn/community/0181445f0194b9a801215aa00b682d.png?x-oss-process=image/auto-orient,1/resize,m_lfit,w_1280,limit_1/sharpen,100",
        "prompt": {
            "input_point": [],
            "input_label": [],
            "input_box": []
        },
        "topic_id": "test123",
        "user_id": "user_test_01",
        "is_predict_type": True,
        "resnet_model_id": "Train-imgSeg-52ddfff5-6e6e-4e83-955a-94783d83613a",
        "read_text_component_list": ["Button", "Input"]
    }

    print("\n测试SAM推理接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试OCR文字识别接口
def test_ocr_predict():
    url = "http://localhost:9002/imgSeg/v1/predictText"
    data = {
        "image_url_list": [
            "https://img.zcool.cn/community/0181445f0194b9a801215aa00b682d.png?x-oss-process=image/auto-orient,1/resize,m_lfit,w_1280,limit_1/sharpen,100"
        ],
        "user_id": "user_test_01"
    }

    print("\n测试OCR文字识别接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试增加训练数据接口
def test_add_train_data():
    url = "http://localhost:9002/imgSeg/v1/getTrainData"
    data = {
        "user_id": "user_test_01",
        "dataset": {
            "imgseg_test_1": {
                "url": "https://img.zcool.cn/community/0181445f0194b9a801215aa00b682d.png?x-oss-process=image/auto-orient,1/resize,m_lfit,w_1280,limit_1/sharpen,100",
                "component_type": "Table"
            },
            "imgseg_test_2": {
                "url": "https://img1.baidu.com/it/u=3848756526,1882598275&fm=253&fmt=auto&app=138&f=JPEG?w=888&h=500",
                "component_type": "Button"
            }
        }
    }

    print("\n测试增加训练数据接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试启动训练接口
def test_train_start():
    url = "http://localhost:9002/imgSeg/v1/resnetTrainStart"
    data = {
        "user_id": "user_test_01",
        "task_type": "train",
        "train_params": {
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 0.001
        }
    }

    print("\n测试启动训练接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试查询训练进度接口
def test_train_progress():
    url = "http://localhost:9002/imgSeg/v1/resnetTrainProgress"
    data = {
        "train_task_id": "Train-imgSeg-1208b6d9-0102-4323-a090-83e897d116a4",
        "user_id": "user_test_01",
        "task_type": "train"
    }

    print("\n测试查询训练进度接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试终止训练接口
def test_train_stop():
    url = "http://localhost:9002/imgSeg/v1/resnetTrainSkip"
    data = {
        "train_task_id": "Train-imgSeg-1208b6d9-0102-4323-a090-83e897d116a4",
        "user_id": "user_test_01",
        "task_type": "train"
    }

    print("\n测试终止训练接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试SAM自动分割接口
def test_sam2_predict():
    url = "http://localhost:9002/imgSeg/v1/predictSam2"
    data = {
        "image_url": "https://img.zcool.cn/community/0181445f0194b9a801215aa00b682d.png?x-oss-process=image/auto-orient,1/resize,m_lfit,w_1280,limit_1/sharpen,100",
        "select_box": [350, 500, 630, 830],
        "prompt": {
            "input_point": [],
            "input_label": []
        },
        "topic_id": "test123",
        "user_id": "user_test_01",
        "resnet_model_id": "Train-imgSeg-52ddfff5-6e6e-4e83-955a-94783d83613a",
        "is_predict_type": True
    }

    print("\n测试SAM自动分割接口...")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=data)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
        return resp_json
    except Exception as e:
        print(f"请求失败: {e}")
        return None

# 测试ingest接口（使用 image_url 参数）
def test_ingest():
    url = "http://localhost:9002/imgSeg/v1/ingest"
    params = {
        "image_url": "https://img.zcool.cn/community/0181445f0194b9a801215aa00b682d.png?x-oss-process=image/auto-orient,1/resize,m_lfit,w_1280,limit_1/sharpen,100",
        "user_id": "user_test_01"
    }

    print("\n测试ingest接口...")
    print(f"请求URL: {url}")
    print(f"请求参数: {json.dumps(params, indent=2, ensure_ascii=False)}")

    try:
        # 使用 params 将 image_url 作为查询参数发送，ingest 接口接受上传或 image_url
        response = requests.post(url, params=params)
        resp_json = response.json()
        safe_json = remove_base64_from_json(resp_json)
        print(f"响应状态码: {response.status_code}")
        try:
            print(f"响应内容: {json.dumps(safe_json, indent=2, ensure_ascii=False)}")
            return resp_json
        except Exception:
            print(f"响应非JSON: {response.text}")
            return None
    except Exception as e:
        print(f"请求失败: {e}")
        return None

if __name__ == "__main__":
    # 只测试还未测试过的接口
    print("开始测试剩余接口...")
    # test_resnet_predict()  # 已测试 ✓
    # test_sam1_predict()    # 已测试 ✓
    # test_ocr_predict()     # 已测试 ✓
    # test_add_train_data()  # 已测试 ✓
    # test_train_start()     # 已测试 ✓
    # test_train_progress()  # 已测试 ✓
    # test_train_stop()      # 已测试 ✓
    # test_ingest()        # 已测试 ✓
    test_sam2_predict()  # 测试超时
