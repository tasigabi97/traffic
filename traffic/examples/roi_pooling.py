#! /usr/bin/env python
import numpy as np
from tensorflow.compat.v1 import (
    disable_eager_execution,
    placeholder,
    float32,
    Session,
    ConfigProto,
)
from tensorflow.config import list_physical_devices
from tensorflow.test import is_built_with_cuda
from traffic.roi_pooling import ROIPoolingLayer


if __name__ == "__main__":
    disable_eager_execution()
    assert len(list_physical_devices("GPU")) and is_built_with_cuda()

    # Define parameters
    batch_size = 1
    img_height = 200
    img_width = 100
    n_channels = 1
    n_rois = 2
    pooled_height = 3
    pooled_width = 7
    # Create feature map input
    feature_maps_shape = (batch_size, img_height, img_width, n_channels)
    feature_maps_tf = placeholder(float32, shape=feature_maps_shape)
    feature_maps_np = np.ones(feature_maps_tf.shape, dtype="float32")
    feature_maps_np[0, img_height - 1, img_width - 3, 0] = 50
    print(f"feature_maps_np.shape = {feature_maps_np.shape}")
    # Create batch size
    roiss_tf = placeholder(float32, shape=(batch_size, n_rois, 4))
    roiss_np = np.asarray(
        [[[0.5, 0.2, 0.7, 0.4], [0.0, 0.0, 1.0, 1.0]]], dtype="float32"
    )
    print(f"roiss_np.shape = {roiss_np.shape}")
    # Create layer
    roi_layer = ROIPoolingLayer(pooled_height, pooled_width)
    pooled_features = roi_layer([feature_maps_tf, roiss_tf])
    print(f"output shape of layer call = {pooled_features.shape}")
    # Run tensorflow session
    with Session() as session:
        result = session.run(
            pooled_features,
            feed_dict={feature_maps_tf: feature_maps_np, roiss_tf: roiss_np},
        )

    print(f"result.shape = {result.shape}")
    print(f"first  roi embedding=\n{result[0, 0, :, :, 0]}")
    print(f"second roi embedding=\n{result[0, 1, :, :, 0]}")
