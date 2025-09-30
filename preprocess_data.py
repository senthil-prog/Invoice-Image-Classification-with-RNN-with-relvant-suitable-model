import tensorflow as tf

IMG_SIZE = 128

def load_data(data_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True
    )
    return generator, generator.class_indices
