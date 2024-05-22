from pathlib import Path

import numpy as np
import pytest
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

atol = 5e-3

@pytest.mark.parametrize(
    'model_type',
    [
        'max',
        'avg',
    ]
)
@pytest.mark.parametrize(
    'padding',
    [
        'same',
        'valid',
    ]
)
@pytest.mark.parametrize(
    'in_shape',
    [
        124
    ]
)
@pytest.mark.parametrize(
    'in_filt',
    [
        5,
    ]
)
@pytest.mark.parametrize(
    'io_type',
    [
        'io_parallel',
    ]
)
@pytest.mark.parametrize(
    'backend',
    [
        'Quartus',
        'Vitis',
        'Vivado',
        'Catapult'
    ]
)
def test_pool1d(model_type, padding, in_shape, in_filt, io_type, backend):

    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling1D(pool_size=3, input_shape=(in_shape, in_filt), padding=padding))
    elif model_type == 'max':
        model.add(MaxPooling1D(pool_size=3, input_shape=(in_shape, in_filt), padding=padding))

    data_1d = np.random.rand(100, in_shape, in_filt)

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_pool1d_{model_type}_w{in_shape}_f{in_filt}_{padding}_{backend}_{io_type}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_1d)
    y_hls = hls_model.predict(data_1d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)

@pytest.mark.parametrize(
    'model_type',
    [
        'max',
        'avg',
    ]
)
@pytest.mark.parametrize(
    'padding',
    [
        'same',
        'valid',
    ]
)
@pytest.mark.parametrize(
    'in_shape',
    [
        [124, 124]
    ]
)
@pytest.mark.parametrize(
    'in_filt',
    [
        5,
    ]
)
@pytest.mark.parametrize(
    'io_type',
    [
        'io_parallel',
    ]
)
@pytest.mark.parametrize(
    'backend',
    [
        'Quartus',
        'Vitis',
        'Vivado',
        'Catapult'
    ]
)
def test_pool2d(model_type, padding, in_shape, in_filt, io_type, backend):

    model = Sequential()
    if model_type == 'avg':
        model.add(AveragePooling2D(input_shape=(in_shape[0], in_shape[1], in_filt), padding=padding))
    elif model_type == 'max':
        model.add(MaxPooling2D(input_shape=(in_shape[0], in_shape[1], in_filt), padding=padding))

    data_2d = np.random.rand(100, in_shape[0], in_shape[1], in_filt)

    config = hls4ml.utils.config_from_keras_model(model, default_precision='ap_fixed<32,9>', granularity='name')

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        io_type=io_type,
        output_dir=str(test_root_path / f'hls4mlprj_pool2d_{model_type}_h{in_shape[0]}_w{in_shape[1]}_f{in_filt}_{padding}_{backend}_{io_type}'),
        backend=backend,
    )
    hls_model.compile()

    y_keras = model.predict(data_2d)
    y_hls = hls_model.predict(data_2d).reshape(y_keras.shape)
    np.testing.assert_allclose(y_keras, y_hls, rtol=0, atol=atol, verbose=True)
