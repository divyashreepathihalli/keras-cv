import keras
import numpy as np

from keras_cv.models import CLIP
from keras_cv.models.feature_extractor.clip import CLIPProcessor

model = CLIP()
image = np.random.rand(1, 224, 224, 3)  # Each sample is a single image
text = np.random.randint(
    0, 100, size=(4, 77)
)  # 3 corresponding texts per image
attention_mask = np.random.choice([True, False], size=(4, 77))
image_logits, text_logits = model(
    {
        "image": image,
        "text": text,
        "attention_mask": attention_mask,
    }
)
output = keras.layers.Softmax()(image_logits)
print(output)
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
)
batch_size = 20  # Number of image samples

# Input Data
input_image = np.random.rand(
    batch_size, 224, 224, 3
)  # Each sample is a single image
input_text = np.random.randint(
    0, 100, size=(batch_size, 3, 77)
)  # 3 corresponding texts per image
input_attention_mask = np.random.choice([True, False], size=(batch_size, 3, 77))

# Output Data
output_image_logits = np.random.rand(batch_size, 1, 3)  # length of image
output_text_logits = np.random.rand(batch_size, 3, 1)  # Length of text

model.fit(
    {
        "image": input_image,
        "text": input_text,
        "attention_mask": input_attention_mask,
    },
    (output_image_logits, output_text_logits),
    epochs=2,
    batch_size=1,
)
print("done with model fit")
