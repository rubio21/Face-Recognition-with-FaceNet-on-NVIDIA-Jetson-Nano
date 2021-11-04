
import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
import re
import cv2
import _face_detection as ftk
tf.disable_eager_execution()


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def make_image_tensor(img, image_size, do_prewhiten=True):
    image = np.zeros((1, image_size, image_size, 3))
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    image[0, :, :, :] = img
    return image


def make_images_tensor(img1, img2, image_size, do_prewhiten=True):
    images = np.zeros((2, image_size, image_size, 3))
    for i, img in enumerate([img1, img2]):
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        images[i, :, :, :] = img
    return images


def load_model(model, session):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        meta_file, ckpt_file = get_model_filenames(model_exp)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(session, os.path.join(model_exp, ckpt_file))


class Verification:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.images_placeholder = ''
        self.embeddings = ''
        self.phase_train_placeholder = ''
        self.embedding_size = ''
        self.session_closed = False

    def __del__(self):
        if not self.session_closed:
            self.session.close()

    def kill_session(self):
        self.session_closed = True
        self.session.close()

    def load_model(self, model):
        load_model(model, self.session)

    def initial_input_output_tensors(self):
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def img_to_encoding(self, img, image_size):
        image = make_image_tensor(img, image_size)

        feed_dict = {self.images_placeholder: image, self.phase_train_placeholder: False}
        emb_array = np.zeros((1, self.embedding_size))
        emb_array[0, :] = self.session.run(self.embeddings, feed_dict=feed_dict)

        return np.squeeze(emb_array)


class FaceDetection:

    # Modify the verification_threshold incase you want to edit 
    verification_threshold = 0.8
    v, net = None, None

    def __init__(self):
        FaceDetection.net = FaceDetection.load_opencv()
        FaceDetection.v = FaceDetection.load_model()

    @staticmethod
    def load_opencv():
        model_path = "./Models/OpenCV/opencv_face_detector_uint8.pb"
        model_weights = "./Models/OpenCV/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(model_path, model_weights)
        return net

    @staticmethod
    def load_model():
        v = ftk.Verification()
        v.load_model("./Models/FaceDetection/")
        v.initial_input_output_tensors()
        return v

    @staticmethod
    def is_same(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff < FaceDetection.verification_threshold, diff

    @staticmethod
    def fetch_embeddings(image):

        image_size = 160
        height, width, channels = image.shape

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        FaceDetection.net.setInput(blob)
        detections = FaceDetection.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append([x1, y1, x2 - x1, y2 - y1])

                # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.imshow("img", image)
                # cv2.waitKey(0)

        if len(faces) == 1:
            face = faces[0]
            x, y, w, h = face
            im_face = image[y:y + h, x:x + w]
            img = cv2.resize(im_face, (200, 200))
            user_embed = FaceDetection.v.img_to_encoding(cv2.resize(img, (160, 160)), image_size)
        else:
            return None

        return user_embed

    @staticmethod
    def verify_face(image1, image2):

        if not FaceDetection.v:
            FaceDetection.v = FaceDetection.load_model()

        if not FaceDetection.net:
            FaceDetection.net = FaceDetection.load_opencv()

        img1_emb = FaceDetection.fetch_embeddings(image1)
        img2_emb = FaceDetection.fetch_embeddings(image2)

        if img1_emb is not None and img2_emb is not None:
            response = FaceDetection.is_same(img1_emb, img2_emb)
            return {"response": "API result", "verified": str(response[0]), "accuracy": response[1]}

        cv2.destroyAllWindows()
        return {"response": "Face unavailable in either image", "verified": str(False), "accuracy": 0}