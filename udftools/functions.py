import pickle
import cv2,time
import numpy as np

# import onnxruntime

# class ONNXModel():
#     def __init__(self, onnx_path):
#         """
#         :param onnx_path:
#         """
#         self.onnx_session = onnxruntime.InferenceSession(onnx_path)
#         self.input_name = self.get_input_name(self.onnx_session)
#         self.output_name = self.get_output_name(self.onnx_session)
#         print("input_name:{}".format(self.input_name))
#         print("output_name:{}".format(self.output_name))
#
#     def get_output_name(self, onnx_session):
#         """
#         output_name = onnx_session.get_outputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         output_name = []
#         for node in onnx_session.get_outputs():
#             output_name.append(node.name)
#         return output_name
#
#     def get_input_name(self, onnx_session):
#         """
#         input_name = onnx_session.get_inputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         input_name = []
#         for node in onnx_session.get_inputs():
#             input_name.append(node.name)
#         return input_name
#
#     def get_input_feed(self, input_name, image_tensor):
#         """
#         input_feed={self.input_name: image_tensor}
#         :param input_name:
#         :param image_tensor:
#         :return:
#         """
#         input_feed = {}
#         for name in input_name:
#             input_feed[name] = image_tensor
#         return input_feed
#
#     def forward(self, image_tensor):
#         '''
#         image_tensor = image.transpose(2, 0, 1)
#         image_tensor = image_tensor[np.newaxis, :]
#         onnx_session.run([output_name], {input_name: x})
#         :param image_tensor:
#         :return:
#         '''
#         # 输入数据的类型必须与模型一致,以下三种写法都是可以的
#         # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
#         # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
#         input_feed = self.get_input_feed(self.input_name, image_tensor)
#         output = self.onnx_session.run(self.output_name, input_feed=input_feed)
#         return output

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def face_normalize(img_cv2):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    mean = 127.5
    std = 127.5
    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = img_data - mean
    img_data = img_data / std
    img_data = img_data.astype(np.float32)
    return img_data

def face_format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format

def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)

def smallest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n least indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, array.shape)


class timer(object):
    def __init__(self,it:str='',display=True):
        self._start = 0
        self._end = 0
        self._name = it
        self._display = display
        pass

    def start(self,it:str=None):
        self._start = time.time()
        if it is not None:
            self._name = it
        else:
            pass

    def end(self):
        self._end = time.time()

    def diff(self):
        tt = self._end-self._start
        return tt

    def eclapse(self):
        self.end()
        tt = self.diff()
        if self._display:
            print('<<%s>> eclapse: %f sec...'%(self._name,tt))
        return tt


def datetime_verify(date):
    """判断是否是一个有效的日期字符串"""
    try:
        time.strptime(date, "%Y-%m-%d %H:%M:%S")
        # if ":" in date:
        #     time.strptime(date, "%Y-%m-%d %H:%M:%S")
        # else:
        #     time.strptime(date, "%Y-%m-%d")
        return True
    except Exception as e:
        # print(e)
        return False

def filter_time(datestr:str):
    res = filter(lambda ch: ch in '0123456789.- :', datestr)
    return ''.join(list(res))


def timestamp_to_timestr(timestamp:int, tmfmt = "%Y-%m-%d %H:%M:%S" ):
    # 转换成localtime
    time_local = time.localtime(timestamp)
    # 转换成新的时间格式(2016-05-05 20:28:54)
    dt = time.strftime( tmfmt, time_local)
    return dt


class unsupervised_cluster(object):
    def __init__(self, distance_threshold=0.5):
        super().__init__()
        self.labels_ = None
        self.n_clusters_ = None
        self.distance_threshold = distance_threshold

    def fit(self, X):
        """
        :param X: [N,D] N samples with dimensionality D 
        :return: 
        """
        N, D = X.shape

        Cmtx = X
        label = -1 * np.ones(N)

        n_cluster = 0
        # 计算各个类中心之间的距离
        dist = Cmtx @ Cmtx.transpose()
        inds = np.array(range(N))

        while len(inds) > 0:
            i = inds[0]
            ovr = dist[i, inds]
            similary_group = np.where(ovr >= self.distance_threshold)
            unsimilary_group = np.where(ovr < self.distance_threshold)

            label[inds[similary_group]] = n_cluster
            n_cluster += 1

            inds = inds[unsimilary_group]
            # print(inds)

        self.labels_ = label
        self.n_clusters_ = n_cluster

        # print('%d cluster find' % n_cluster)

        return self