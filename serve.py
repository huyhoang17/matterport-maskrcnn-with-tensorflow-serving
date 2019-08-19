import os
import copy
import uuid
import time
import json

import cv2
from django.conf import settings as st
import numpy as np
import matplotlib.pyplot as plt
import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from mrcnn import visualize
from mrcnn.model import utils as mrcnn_utils
from mrcnn import model as modellib

from api.helpers import utils as api_utils
import configs as cf
from model_configs import mconfig as mcf


def _grpc_client_request(img_arr,
                         image_meta,
                         anchors,
                         host,
                         port,
                         in_tensor_image,
                         in_tensor_image_meta,
                         in_tensor_anchors,
                         in_tensor_dtype,
                         img_size,
                         model_sig_name,
                         model_spec_name):

    channel = grpc.insecure_channel("{}:{}".format(host, port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Create PredictRequest ProtoBuf from image data
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    request.model_spec.signature_name = model_sig_name

    # input_image
    img_arr = np.expand_dims(img_arr, axis=0)
    request.inputs[in_tensor_image].CopyFrom(
        tf.compat.v1.make_tensor_proto(
            img_arr,
            dtype=in_tensor_dtype,
            shape=[1, img_arr.shape[1], img_arr.shape[2], img_arr.shape[3]]
        )
    )

    # input_image_meta
    image_meta = np.expand_dims(image_meta, axis=0)
    request.inputs[in_tensor_image_meta].CopyFrom(
        tf.compat.v1.make_tensor_proto(
            image_meta,
            dtype=in_tensor_dtype,
            shape=[1, image_meta.shape[1]]
        )
    )

    # input_anchors
    if len(anchors.shape) == 2:
        anchors = np.expand_dims(anchors, axis=0)
    request.inputs[in_tensor_anchors].CopyFrom(
        tf.compat.v1.make_tensor_proto(
            anchors,
            dtype=in_tensor_dtype,
            shape=[1, anchors.shape[1], anchors.shape[2]]
        )
    )

    predict_response = stub.Predict(request, timeout=cf.GRPC_TIMEOUT)

    return predict_response


def preprocess_input(img, img_size=640):

    if isinstance(img, str):
        img = api_utils.load_img(img)

    if img_size is not None:
        img = cv2.resize(img, (img_size, img_size))

    molded_image, window, scale, padding, crop = mrcnn_utils.resize_image(
        img,
        min_dim=mcf.IMAGE_MIN_DIM,
        min_scale=mcf.IMAGE_MIN_SCALE,
        max_dim=mcf.IMAGE_MAX_DIM,
        mode=mcf.IMAGE_RESIZE_MODE
    )
    molded_image = modellib.mold_image(molded_image, mcf)

    image_meta = modellib.compose_image_meta(
        0, img.shape, molded_image.shape, window, scale,
        np.zeros([mcf.NUM_CLASSES], dtype=np.int32)
    )

    anchors = api_utils.get_anchors(molded_image.shape)

    return molded_image, image_meta, anchors, window


def grpc_inference(img):

    # preprocess input
    molded_image, image_meta, anchors, window = \
        preprocess_input(img, cf.IMAGE_SIZE)

    predict_res = _grpc_client_request(
        molded_image.astype(np.float32),
        image_meta.astype(np.float32),
        anchors.astype(np.float32),
        cf.HOST,
        cf.gRPC_PORT,
        in_tensor_image=cf.IN_TENSOR_IMAGE,
        in_tensor_image_meta=cf.IN_TENSOR_IMAGE_META,
        in_tensor_anchors=cf.IN_TENSOR_ANCHORS,
        in_tensor_dtype=cf.IN_TENSOR_DTYPE,
        img_size=cf.IMAGE_SIZE,
        model_sig_name=cf.MODEL_SIG_NAME,
        model_spec_name=cf.MODEL_SPEC_NAME
    )

    mrcnn_detection = np.array(
        predict_res.outputs[cf.OUT_TENSOR_DETECTION].float_val
    ).reshape((-1, *cf.OUT_DETECTION_SHAPE))  # noqa
    mrcnn_mask = np.array(
        predict_res.outputs[cf.OUT_TENSOR_MASK].float_val
    ).reshape((-1, *cf.OUT_MASK_SHAPE))  # noqa

    return mrcnn_detection, mrcnn_mask, molded_image, window


def do_inference(img):

    # do grpc inference
    mrcnn_detection, mrcnn_mask, molded_image, window = \
        grpc_inference(img)

    final_rois, final_class_ids, final_scores, final_masks = \
        api_utils.unmold_detections(
            mrcnn_detection,
            mrcnn_mask,
            img.shape,
            molded_image.shape,
            window
        )

    random_id = str(uuid.uuid4())
    mask_fn = "mask-{}.png".format(random_id)
    save_path = os.path.join("media", mask_fn)

    visualize.display_instances(
        img,
        final_rois,
        final_masks,
        final_class_ids,
        ["BG", *cf.DAMAGE_CLASSES],
        final_scores,
        ax=plt.axes(),
        save_path=save_path,
    )
    print(">>> Save image: {}".format(save_path))
    print(">>> Complete!")

    return save_path
