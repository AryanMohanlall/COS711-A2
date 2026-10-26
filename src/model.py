import tensorflow as tf

from .data import load_image, instance_mask_for, load_annotations
import numpy as np
import cv2

def encoder(inputs, num_filters):
    """The encoder is responsible for extracting features from the input image. 
    It applies two convolutional layers followed by a ReLU Activation to learn patterns and 
    then uses max pooling to reduce the image size help the model focus on important features."""
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="valid")(inputs)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='valid')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    return x

def decoder(inputs, skip_features, num_filters):
    """The decoder helps restore the original image size while combining the low-level and high-level features. 
    It starts by upsampling the feature map, resizes the corresponding encoder output (skip connection), 
    merges them and then applies two convolution layers with ReLU."""

    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='valid')(inputs)

    skip_features = tf.keras.layers.Resizing(x.shape[1], x.shape[2])(skip_features)

    x = tf.keras.layers.Concatenate()([x, skip_features])

    x = tf.keras.layers.Conv2D(num_filters, 3, padding='valid')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='valid')(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x

def unet_model(input_shape=(256, 256, 1), num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Contracting Path (Encoder)
    s1 = encoder(inputs, 64)
    s2 = encoder(s1, 128)
    s3 = encoder(s2, 256)
    s4 = encoder(s3, 512)
    
    # Bottleneck
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='valid')(s4)
    b1 = tf.keras.layers.Activation('relu')(b1)
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='valid')(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)
    
    # Expansive Path (Decoder)
    d1 = decoder(b1, s4, 512)
    d2 = decoder(d1, s3, 256)
    d3 = decoder(d2, s2, 128)
    d4 = decoder(d3, s1, 64)
    
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='valid', activation='sigmoid')(d4)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net')
    return model

def load_image_and_mask(path, image_id, size=256):
    img = load_image(path)
    anns = load_annotations(path)

    # Generate the ground-truth instance mask
    mask = instance_mask_for(image_id, anns, img.shape)

    # Resize the input image
    img = cv2.resize(
        img, (size, size),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float and normalize if pixel values are 0–255
    img = img.astype(np.float32) / 255.0

    # Convert all filament instances into one foreground class
    mask = (mask > 0).astype(np.float32)

    # Resize mask without introducing new class values
    mask = cv2.resize(
        mask, (size, size),
        interpolation=cv2.INTER_NEAREST
    )

    # Add the channel dimension
    if img.ndim == 2:
        img = img[..., np.newaxis]

    mask = mask[..., np.newaxis]

    return img, mask





