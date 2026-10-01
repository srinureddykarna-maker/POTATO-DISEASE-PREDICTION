import tensorflow as tf

DATASET_PATH = "dataset/PlantVillage"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names BEFORE applying tf.data operations
class_names = train_ds.class_names


# Load remaining 20%
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Split remaining 20% into validation and testing
val_batches = int(len(temp_ds) * 0.5)

val_ds = temp_ds.take(val_batches)
test_ds = temp_ds.skip(val_batches)


# tf.data pipeline
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(
    buffer_size=1000,
    seed=SEED
).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

test_ds = test_ds.prefetch(AUTOTUNE)


# Check
print("Classes:", class_names)
print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Testing batches:", len(test_ds))
