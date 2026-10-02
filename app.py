from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import io
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="MobileNetV2 Layer-wise Image Classifier",
    description=(
        "Educational MobileNetV2 implementation built layer-by-layer in "
        "TensorFlow/Keras and initialized with ImageNet pretrained weights."
    ),
    version="2.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Serve your existing CSS and JavaScript folders, if present
CSS_DIR = os.path.join(BASE_DIR, "css")
JS_DIR = os.path.join(BASE_DIR, "js")

if os.path.isdir(CSS_DIR):
    app.mount("/css", StaticFiles(directory=CSS_DIR), name="css")

if os.path.isdir(JS_DIR):
    app.mount("/js", StaticFiles(directory=JS_DIR), name="js")


# ============================================================
# MobileNetV2 building blocks
# ============================================================

def relu6(x, name=None):
    """ReLU6 activation used by MobileNetV2."""
    return layers.ReLU(max_value=6.0, name=name)(x)


def correct_pad(inputs, kernel_size):
    """
    Reproduces the asymmetric padding strategy used by Keras MobileNetV2
    before stride-2 depthwise convolutions.
    """
    img_dim = 1 if tf.keras.backend.image_data_format() == "channels_last" else 2
    input_size = tf.keras.backend.int_shape(inputs)[img_dim:img_dim + 2]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def inverted_residual_block(
    x,
    expansion,
    filters,
    stride,
    block_id,
):
    """
    MobileNetV2 inverted residual block.

    Structure:
        Input
          ↓
        1×1 Expansion Conv     (except first block)
          ↓
        BatchNorm
          ↓
        ReLU6
          ↓
        3×3 Depthwise Conv
          ↓
        BatchNorm
          ↓
        ReLU6
          ↓
        1×1 Projection Conv
          ↓
        BatchNorm
          ↓
        Residual Add           (only if stride=1 and channels match)
    """

    in_channels = tf.keras.backend.int_shape(x)[-1]
    prefix = f"block_{block_id}_"

    # --------------------------------------------------------
    # 1. Expansion phase
    # --------------------------------------------------------
    if block_id != 0:
        expanded_channels = int(in_channels * expansion)

        x_expanded = layers.Conv2D(
            expanded_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)

        x_expanded = layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_BN",
        )(x_expanded)

        x_expanded = relu6(
            x_expanded,
            name=prefix + "expand_relu",
        )
    else:
        # First bottleneck has expansion factor 1, so no expansion Conv2D
        x_expanded = x

    # --------------------------------------------------------
    # 2. Depthwise convolution
    # --------------------------------------------------------
    if stride == 2:
        x_depthwise = layers.ZeroPadding2D(
            padding=correct_pad(x_expanded, 3),
            name=prefix + "pad",
        )(x_expanded)

        depthwise_padding = "valid"
    else:
        x_depthwise = x_expanded
        depthwise_padding = "same"

    x_depthwise = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding=depthwise_padding,
        use_bias=False,
        name=prefix + "depthwise",
    )(x_depthwise)

    x_depthwise = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )(x_depthwise)

    x_depthwise = relu6(
        x_depthwise,
        name=prefix + "depthwise_relu",
    )

    # --------------------------------------------------------
    # 3. Projection phase
    # --------------------------------------------------------
    x_projected = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )(x_depthwise)

    x_projected = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )(x_projected)

    # --------------------------------------------------------
    # 4. Linear residual connection
    # --------------------------------------------------------
    if stride == 1 and in_channels == filters:
        return layers.Add(name=prefix + "add")([x, x_projected])

    return x_projected


# ============================================================
# MobileNetV2 architecture built layer-by-layer
# ============================================================

def build_mobilenet_v2_layerwise(
    input_shape=(224, 224, 3),
    num_classes=1000,
):
    """
    Builds MobileNetV2 manually using TensorFlow/Keras layers.

    Standard MobileNetV2 bottleneck configuration:
        t = expansion factor
        c = output channels
        n = number of repeats
        s = stride of first block in the stage

        t   c    n   s
        1   16   1   1
        6   24   2   2
        6   32   3   2
        6   64   4   2
        6   96   3   1
        6   160  3   2
        6   320  1   1
    """

    inputs = layers.Input(
        shape=input_shape,
        name="input_layer",
    )

    # --------------------------------------------------------
    # Stem
    # 224×224×3 → 112×112×32
    # --------------------------------------------------------
    x = layers.ZeroPadding2D(
        padding=((0, 1), (0, 1)),
        name="Conv1_pad",
    )(inputs)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="valid",
        use_bias=False,
        name="Conv1",
    )(x)

    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name="bn_Conv1",
    )(x)

    x = relu6(x, name="Conv1_relu")

    # --------------------------------------------------------
    # Bottleneck stages
    # --------------------------------------------------------

    # Stage 1: 112×112×32 → 112×112×16
    x = inverted_residual_block(
        x,
        expansion=1,
        filters=16,
        stride=1,
        block_id=0,
    )

    # Stage 2: → 56×56×24
    x = inverted_residual_block(x, 6, 24, 2, 1)
    x = inverted_residual_block(x, 6, 24, 1, 2)

    # Stage 3: → 28×28×32
    x = inverted_residual_block(x, 6, 32, 2, 3)
    x = inverted_residual_block(x, 6, 32, 1, 4)
    x = inverted_residual_block(x, 6, 32, 1, 5)

    # Stage 4: → 14×14×64
    x = inverted_residual_block(x, 6, 64, 2, 6)
    x = inverted_residual_block(x, 6, 64, 1, 7)
    x = inverted_residual_block(x, 6, 64, 1, 8)
    x = inverted_residual_block(x, 6, 64, 1, 9)

    # Stage 5: → 14×14×96
    x = inverted_residual_block(x, 6, 96, 1, 10)
    x = inverted_residual_block(x, 6, 96, 1, 11)
    x = inverted_residual_block(x, 6, 96, 1, 12)

    # Stage 6: → 7×7×160
    x = inverted_residual_block(x, 6, 160, 2, 13)
    x = inverted_residual_block(x, 6, 160, 1, 14)
    x = inverted_residual_block(x, 6, 160, 1, 15)

    # Stage 7: → 7×7×320
    x = inverted_residual_block(x, 6, 320, 1, 16)

    # --------------------------------------------------------
    # Final 1×1 convolution
    # 7×7×320 → 7×7×1280
    # --------------------------------------------------------
    x = layers.Conv2D(
        1280,
        kernel_size=1,
        use_bias=False,
        name="Conv_1",
    )(x)

    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name="Conv_1_bn",
    )(x)

    x = relu6(x, name="out_relu")

    # --------------------------------------------------------
    # Classification head
    # 7×7×1280 → 1280 → 1000 ImageNet classes
    # --------------------------------------------------------
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    return Model(
        inputs,
        outputs,
        name="MobileNetV2_LayerWise",
    )


# ============================================================
# Build our educational layer-wise model
# ============================================================

print("Building MobileNetV2 layer-by-layer...")
model = build_mobilenet_v2_layerwise()

# ============================================================
# Load official ImageNet pretrained weights
# ============================================================
#
# We create the official TensorFlow/Keras MobileNetV2 only once,
# copy its pretrained weights into our layer-wise implementation,
# and then use OUR manually constructed model for inference.
#
# This keeps the architecture educational while still using
# genuine ImageNet pretrained weights.
# ============================================================

print("Loading official ImageNet pretrained MobileNetV2 weights...")

official_pretrained_model = MobileNetV2(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3),
)

try:
    model.set_weights(official_pretrained_model.get_weights())
except ValueError as exc:
    raise RuntimeError(
        "The manually constructed MobileNetV2 architecture does not match "
        "the official TensorFlow MobileNetV2 weight structure."
    ) from exc

# Free the second model object after copying weights
del official_pretrained_model

print("MobileNetV2 is ready.")
print("Total parameters:", model.count_params())


# ============================================================
# Educational architecture information
# ============================================================

MOBILENET_V2_STAGES = [
    {
        "stage": "Stem",
        "operation": "3x3 Conv2D",
        "expansion": "-",
        "output_channels": 32,
        "repeats": 1,
        "stride": 2,
    },
    {
        "stage": "Bottleneck 1",
        "operation": "Inverted Residual",
        "expansion": 1,
        "output_channels": 16,
        "repeats": 1,
        "stride": 1,
    },
    {
        "stage": "Bottleneck 2",
        "operation": "Inverted Residual",
        "expansion": 6,
        "output_channels": 24,
        "repeats": 2,
        "stride": 2,
    },
    {
        "stage": "Bottleneck 3",
        "operation": "Inverted Residual",
        "expansion": 6,
        "output_channels": 32,
        "repeats": 3,
        "stride": 2,
    },
    {
        "stage": "Bottleneck 4",
        "operation": "Inverted Residual",
        "expansion": 6,
        "output_channels": 64,
        "repeats": 4,
        "stride": 2,
    },
    {
        "stage": "Bottleneck 5",
        "operation": "Inverted Residual",
        "expansion": 6,
        "output_channels": 96,
        "repeats": 3,
        "stride": 1,
    },
    {
        "stage": "Bottleneck 6",
        "operation": "Inverted Residual",
        "expansion": 6,
        "output_channels": 160,
        "repeats": 3,
        "stride": 2,
    },
    {
        "stage": "Bottleneck 7",
        "operation": "Inverted Residual",
        "expansion": 6,
        "output_channels": 320,
        "repeats": 1,
        "stride": 1,
    },
    {
        "stage": "Final Conv",
        "operation": "1x1 Conv2D",
        "expansion": "-",
        "output_channels": 1280,
        "repeats": 1,
        "stride": 1,
    },
    {
        "stage": "Classifier",
        "operation": "GlobalAveragePooling + Dense",
        "expansion": "-",
        "output_channels": 1000,
        "repeats": 1,
        "stride": "-",
    },
]


# ============================================================
# Frontend route
# ============================================================

@app.get("/")
def home():
    index_file = os.path.join(BASE_DIR, "index.html")

    if not os.path.exists(index_file):
        return {
            "message": "MobileNetV2 API is running.",
            "architecture": "/api/architecture",
            "health": "/health",
        }

    return FileResponse(index_file)


# ============================================================
# Architecture endpoint
# ============================================================

@app.get("/api/architecture")
def architecture():
    """
    Returns MobileNetV2 stage-wise architecture for teaching/demo purposes.
    """
    return {
        "model": "MobileNetV2",
        "input_shape": [224, 224, 3],
        "pretrained_on": "ImageNet",
        "number_of_classes": 1000,
        "total_parameters": int(model.count_params()),
        "core_idea": (
            "Inverted residual blocks with linear bottlenecks and "
            "depthwise separable convolutions."
        ),
        "stages": MOBILENET_V2_STAGES,
    }


# ============================================================
# Detailed Keras layer endpoint
# ============================================================

@app.get("/api/layers")
def layers_list():
    """
    Returns every Keras layer name, type, and output shape.
    Useful for understanding the model layer-by-layer.
    """

    details = []

    for index, layer in enumerate(model.layers):
        try:
            output_shape = list(layer.output.shape)
        except Exception:
            output_shape = None

        details.append(
            {
                "index": index,
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": output_shape,
                "trainable": layer.trainable,
            }
        )

    return {
        "model": "MobileNetV2 Layer-wise",
        "layer_count": len(model.layers),
        "layers": details,
    }


# ============================================================
# Image prediction endpoint
# ============================================================

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image and returns the top-3 ImageNet predictions.
    """

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # MobileNetV2 preprocessing maps RGB values to [-1, 1]
        image_array = preprocess_input(image_array)

        predictions = model.predict(
            image_array,
            verbose=0,
        )

        decoded = decode_predictions(
            predictions,
            top=3,
        )[0]

        results = []

        for class_id, label, probability in decoded:
            results.append(
                {
                    "class_id": class_id,
                    "label": label.replace("_", " "),
                    "confidence": round(float(probability) * 100, 2),
                }
            )

        return {
            "model": "MobileNetV2",
            "weights": "ImageNet pretrained",
            "prediction_count": len(results),
            "predictions": results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "MobileNetV2",
        "implementation": "Layer-wise TensorFlow/Keras",
        "weights": "ImageNet pretrained",
        "input_size": "224x224x3",
        "classes": 1000,
        "parameters": int(model.count_params()),
    }


# ============================================================
# Local development
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
