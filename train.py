import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
plt.rcParams['font.family'] = ['Microsoft JhengHei']

def check_dataset_structure(dataset_dir):
    required_dirs = [
        os.path.join(dataset_dir, 'train', 'cats'),
        os.path.join(dataset_dir, 'train', 'dogs'),
        os.path.join(dataset_dir, 'validation', 'cats'),
        os.path.join(dataset_dir, 'validation', 'dogs')
    ]
    for d in required_dirs:
        if not os.path.exists(d):
            return False
    return True

class CatDogClassifier:
    def __init__(self, img_size=(128, 128), model_path='cat_dog_classifier.h5'):
        self.img_size = img_size
        self.model_path = model_path
        self.model = None
        self.history = None

    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def prepare_data(self, dataset_dir, batch_size=32):
        train_gen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True)
        val_gen = ImageDataGenerator(rescale=1./255)

        train_data = train_gen.flow_from_directory(
            os.path.join(dataset_dir, 'train'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        val_data = val_gen.flow_from_directory(
            os.path.join(dataset_dir, 'validation'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        return train_data, val_data

    def train(self, dataset_dir, batch_size=32, epochs=20):
        train_data, val_data = self.prepare_data(dataset_dir, batch_size)
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
        ]
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

    def evaluate(self, test_dir, batch_size=32):
        test_gen = ImageDataGenerator(rescale=1./255)
        test_data = test_gen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        return self.model.evaluate(test_data)

    def predict_image(self, image_path):
        img = Image.open(image_path).convert('RGB').resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prob = self.model.predict(img_array)[0][0]
        label = "狗" if prob > 0.5 else "貓"
        print(f"{label} (信心度: {prob:.4f})")

    def plot_history(self):
        if not self.history: return
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='訓練')
        plt.plot(self.history.history['val_accuracy'], label='驗證')
        plt.legend(); plt.title('準確率')
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='訓練')
        plt.plot(self.history.history['val_loss'], label='驗證')
        plt.legend(); plt.title('損失')
        plt.show()

if __name__ == '__main__':
    dataset_path = 'dataset'
    if not check_dataset_structure(dataset_path):
        print("資料集目錄結構錯誤")
        exit()

    clf = CatDogClassifier()
    clf.train(dataset_path, batch_size=32, epochs=10)
    clf.plot_history()
    clf.evaluate('dataset/test')
