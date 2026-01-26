class Url:
    ip = '0.0.0.0'
    port = 9002                                   # ImgSegAPI文件默认运行端口（图分割推理和训练接口）
    port_gradio = 9006                            # imgseg_gradio默认运行端口（resnet训练后台界面）
    # 推理
    PREDICT_RESNET = "/imgSeg/v1/predictResnet"  # Resnet推理
    # predictResnetProcess = "/imgSeg/v1/predictResnetProcess"  # 用于获取Resnet推理状态
    # predictResnetSkip = "/imgSeg/v1/predictResnetSkip"  # 用于终止Resnet推理

    PREDICT_SAM = "/imgSeg/v1/predictSam"  # sam推理(无提示、点、框提示)
    PREDICT_SAM_2 = "/imgSeg/v1/predictSam2"  # sam推理第2接口(框选混合)
    PREDICT_SAM_3 = "/imgSeg/v1/predictSam3"  # sam推理第3接口(套索)
    # predictSamProcess = "/imgSeg/v1/predictSamProcess"  # 用于获取sam推理状态
    # predictSamSkip = "/imgSeg/v1/predictSamSkip"  # 用于终止sam推理

    predictText = "/imgSeg/v1/predictText"
    # 训练
    resnetTrainStart = "/imgSeg/v1/resnetTrainStart"    # 用于开始训练
    resnetTrainProgress = "/imgSeg/v1/resnetTrainProgress"  # 用于训练任务进度查询
    resnetTrainSkip = "/imgSeg/v1/resnetTrainSkip"     # 用于训练任务终止

    getTrainData = "/imgSeg/v1/getTrainData"

    # TRAIN_API_URL_START = "http://43.247.90.210:19002/imgSeg/v1/resnetTrainStart"
    # TRAIN_API_URL_PROGRESS = "http://43.247.90.210:19002/imgSeg/v1/resnetTrainProgress"
    # TRAIN_API_URL_SKIP = "http://43.247.90.210:19002/imgSeg/v1/resnetTrainSkip"
    # ADD_SINGLE_DATA = "http://43.247.90.210:19002/imgSeg/v1/getTrainData"
    TRAIN_API_URL_START = f"http://0.0.0.0:{port}/imgSeg/v1/resnetTrainStart"
    TRAIN_API_URL_PROGRESS = f"http://0.0.0.0:{port}/imgSeg/v1/resnetTrainProgress"
    TRAIN_API_URL_SKIP = f"http://0.0.0.0:{port}/imgSeg/v1/resnetTrainSkip"
    ADD_SINGLE_DATA = f"http://0.0.0.0:{port}/imgSeg/v1/getTrainData"

    GET_RADIO_TEXT_INFO = "/imgSeg/v1/getRadioCoordWidthHeightContentList"
    GET_TABLE_HEADER_INFO = "/imgSeg/v1/getTableHeaderInfo"
    GET_FORM_INPUT_INFO = "/imgSeg/v1/getFormInputInfo"
    GET_BORDER_COLOR_AND_ROUND_HEIGHT = "/imgSeg/v1/getBorderColorAndRoundHeight"
    GET_COMPONENT_COLOR = "/imgSeg/v1/getComponentColor"
    
    # ingest
    INGEST = "/imgSeg/v1/ingest"
    INGEST_STATUS = "/imgSeg/v1/ingest/status"
    INGEST_RESULT = "/imgSeg/v1/ingest/result"

# return中的“code”
class ResponseCode:
    """
    定义返回码
    """

    # 成功返回码
    SUCCESS = 200  # ("OK") 用于一般性的成功返回，不可用于请求错误返回
    CREATED = 201  # ("Created") 资源被创建

    # 一般成功返回码
    SUCCESS_GENERAL = 100  # ("SUCCESS") 成功

    # 用户相关返回码
    USERNAME_OR_PASSWORD_ERROR = -1001  # ("USERNAME_OR_PASSWORD_ERROR ") 用户名或密码错误
    USER_NOT_FOUND = -1002  # ("USER_NOT_FOUND ") 用户不存在
    USER_NOT_LOGIN = -1003  # ("USER_NOT_LOGIN") 用户未登录

    # 数据库相关返回码
    DATABASE_ID_NOT_SET = -2000  # ("DATABASE_ID_NOT_SET") 数据库ID系统自动生成，不能自己指定
    DATABASE_ID_NOT_FOUND = -2001  # ("DATABASE_ID_NOT_FOUND") 数据库主键ID不存在
    DATABASE_UPDATE_ID_NOT_SET = -2002  # ("DATABASE_UPDATE_ID_NOT_SET") 数据库主键ID不存在
    DATABASE_UPDATE_ID_NOT_REPEAT = -2003  # ("DATABASE_UPDATE_ID_NOT_REPEAT") 数据已存在，不允许重复

    # 请求参数相关返回码
    REQUEST_PARAMS_NOT_MATCH = -3000  # ("REQUEST_PARAMS_NOT_MATCH") 参数传递错误

    # 文件相关返回码
    FILE_UPLOAD_ERROR = -10000  # ("FILE_UPLOAD_ERROR") 文件上传错误
    FILE_NOT_FOUND = -10001  # ("FILE_NOT_FOUND") 找不到文件
    CHARACTER_ENCODING_ERROR = -10002  # ("CHARACTER_ENCODING_ERROR") 转码异常
    FILE_IO_EXCEPTION = -10003  # ("FILE_IO_EXCEPTION") 文件IO异常

    # 异常相关返回码
    SQL_EXCEPTION = -10004  # ("SQL_EXCEPTION") SQL执行异常
    NULL_POINTER_EXCEPTION = -10005  # ("NULL_POINTER_EXCEPTION") 空指针异常

    # 其他返回码
    PARAMS_NOT_ENOUGH = -10006  # ("PARAMS_NOT_ENOUGH ") 参数传递不够，请参照模板
    SAVE_ERROR = -10013  # ("SAVE_ERROR") 保存错误
    OPERATOR_ERROR = -10014  # ("OPERATOR_ERROR") 操作错误
    BPM_ERROR = -10015  # ("BPM_ERROR") 工作流异常


# return中的"status"
class StatusCode:
    """
    定义状态码
    """

    # 成功状态码
    OK = 200  # ("OK") 用于一般性的成功返回，不可用于请求错误返回
    CREATED = 201  # ("Created") 资源被创建
    ACCEPTED = 202  # ("Accepted") 用于Controller控制类资源异步处理的返回，仅表示请求已经收到
    NO_CONTENT = 204  # ("No Content") 此状态可能会出现在PUT、POST、DELETE的请求中，表示资源存在，但消息体中不会返回任何资源相关的状态或信息

    # 重定向状态码
    MOVED_PERMANENTLY = 301  # ("Moved Permanently") 资源的URI被转移，需要使用新的URI访问
    FOUND = 302  # ("Found") 不推荐使用，应该只在GET/HEAD方法下使用，客户端才能根据Location执行自动跳转
    SEE_OTHER = 303  # ("See Other") 返回一个资源地址URI的引用，但不强制要求客户端获取该地址的状态
    NOT_MODIFIED = 304  # ("Not Modified") 资源与客户端最近访问的资源版本一致，不返回资源消息体
    TEMPORARY_REDIRECT = 307  # ("Temporary Redirect") URI临时性重定向到另外一个URI

    # 客户端错误状态码
    BAD_REQUEST = 400  # ("Bad Request") 客户端一般性错误返回，具体错误信息可以放在body中
    UNAUTHORIZED = 401  # ("Unauthorized") 访问需要验证的资源时，验证错误
    FORBIDDEN = 403  # ("Forbidden") 非验证性资源访问被禁止
    NOT_FOUND = 404  # ("Not Found") 找不到URI对应的资源
    METHOD_NOT_ALLOWED = 405  # ("Method Not Allowed") HTTP的方法不支持，必须声明该URI所支持的方法
    NOT_ACCEPTABLE = 406  # ("Not Acceptable") 客户端所请求的资源数据格式类型不被支持
    CONFLICT = 409  # ("Conflict") 资源状态冲突
    PRECONDITION_FAILED = 412  # ("Precondition Failed") 有条件的操作不被满足时
    UNSUPPORTED_MEDIA_TYPE = 415  # ("Unsupported Media Type") 客户端支持的数据类型，服务端无法满足

    # 服务器端错误状态码
    INTERNAL_SERVER_ERROR = 500  # ("Internal Server Error") 服务器端接口错误，与客户端无关



# 定义任务状态 （推理和训练）
class TaskStatus:
    """
    定义任务状态码
    """
    NORMAL_END = 200  # 正常结束
    IN_PROGRESS = 300  # 正在进行中
    ABNORMAL_TERMINATION = 400  # 异常终止
    USER_TERMINATED = 500  # 被用户手动终止


# 定义错误码 return中的“err”
class ErrorCode:
    """
    定义错误码的类
    """
    CREATE_TRAIN_FAIL = 800  # 当前有训练任务正在进行，创建训练任务失败
    READ_URL_FAIL = 801  # 读取url图像数据失败
    TIMEOUT = 802  # 超时
    TASK_ID_NOT_FOUND = 803  # 任务ID不存在
    USER_ID_NOT_FOUND = 804  # 用户ID不存在
    MODEL_ID_ERROR = 805  # 模型ID输入错误 (resnet模型id)
    DATA_FORMAT_ERROR = 806  # 数据格式错误 如输入prompt格式有误
    INVALID_PARAMETER = 807  # 参数不合法
    UNKNOWN_EXCEPTION = 808  # 未知异常


# 定义训练数据校验规则参数
class ValidDataParam:
    """
    定义ValidTrainDataParam参数含义
    """
    MIN_TRAIN_IMAGE_WIDTH = 30
    MAX_TRAIN_IMAGE_WIDTH = 740
    MIN_TRAIN_IMAGE_HEIGHT = 30
    MAX_TRAIN_IMAGE_HEIGHT = 740



# 定义训练数据校验结果码
class ValidStatus:
    """
    定义valid_status参数含义
    """

    BOTH_COMPLIANT = 600  # url图片数据和类型type均为合规数据
    URL_NON_COMPLIANT = 601  # url图片数据不合规
    TYPE_NON_COMPLIANT = 602  # type不合规
    BOTH_NON_COMPLIANT = 603  # url图片数据和type均为不合规数据


# 定义训练数据校验任务状态码
class VerifyStatus:
    """
    定义verify_status含义
    """

    STARTED = 700  # 开始校验
    IN_PROGRESS = 701  # 校验中
    COMPLETED = 702  # 校验完成
    TERMINATED = 704  # 校验终止

#定义新增训练数据单条数据结果码
class GetTrainDataStatus:
    SUCCESS = 900 #单条数据新增成功
    URL_OR_TYPE_ERR = 901 #单条数据url或component_type键名称错误
    READ_URL_ERR = 902 #单条数据url图像读取失败


# return中的“message”
class Message:
    SUCCESS = "成功"
    FAILURE = "失败"


# 指定运行的显卡号
CUDA_VISIBLE_DEVICES = "0"

# resnet默认推理模型
DEFAULT_RESNET_MODEL_ID = "Train-imgSeg-52ddfff5-6e6e-4e83-955a-94783d83613a"


TRAIN_INFO = 'trainInfo'     # resnet训练状态信息存放的文件夹名称
RESNET_MODEL = 'model'     # resnet模型存放的文件夹名称
RESNET_DATA = 'data'      # resnet训练数据集存放的文件夹名称
INGEST_INFO = 'ingestInfo'     # ingest 任务信息存放的文件夹名称
UPLOADS_DIR = 'uploads'        # 上传文件存放目录


# 需要识别颜色的组件类型list
NEED_COLOR_COMPONENT = [
    'Radio',  # 单选框
    'Checkbox',  # 多选框
    'Switch',  # 开关
    'Slider',  # 滑块
    'Rate',  # 评分
    'Progress'  # 进度条
    'Divider',  # 分割线
    'NavMenu',  # 导航菜单
    ]
# 需要进行横竖方向判断的组件类型list
NEED_HORIZONTAL_VERTICAL = [
    'Divider',  # 分割线
    'NavMenu',  # 导航菜单
    'Steps',   # 步骤条
]
# 需要识别是否有圆角的组件类型list
NEED_ROUND_CORNER_FLAG = [
    'Button',  # 按钮
    'Input',  # 输入框
    'InputNumber',  # 计数器
    'Select',  # 选择器
    'Cascader',  # 级联选择器
]
# 需要识别边框色的组件类型list
NEED_BOADER_COLOR = [
    'Button',  # 按钮
    'Input',  # 输入框
    'Select',  # 选择器
    'Cascader',  # 级联选择器
    'TimePicker',  # 时间选择器
    'DatePicker',  # 日期选择器
]
