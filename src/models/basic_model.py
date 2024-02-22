from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),
            layers.Conv2D(2, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(4, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(8, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.55),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.55),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )