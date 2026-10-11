import os
import matplotlib
matplotlib.use('Agg') # Non-interactive background plot generation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_face_mask_model(base_dir):
    dataset_dir = os.path.join(base_dir, "dataset")
    with_mask_dir = os.path.join(dataset_dir, "with_mask")
    without_mask_dir = os.path.join(dataset_dir, "without_mask")
    
    if not os.path.exists(with_mask_dir) or not os.path.exists(without_mask_dir):
        raise FileNotFoundError(f"Dataset directories not found under {dataset_dir}. Run download_real_dataset.py first.")
        
    num_with_mask = len(os.listdir(with_mask_dir))
    num_without_mask = len(os.listdir(without_mask_dir))
    print(f"Dataset Loaded:")
    print(f" - with_mask: {num_with_mask} images")
    print(f" - without_mask: {num_without_mask} images")

    # Hyperparameters
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 32
    EPOCHS = 15

    # 1. ImageDataGenerator with Normalization [0, 255] -> [0, 1] & Validation Split
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # 80% Training, 20% Validation
    )

    print("\n--- Preparing Data Generators ---")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical', # 2 classes with Softmax
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_indices = train_generator.class_indices
    print("Class mapping:", class_indices) # {'with_mask': 0, 'without_mask': 1}

    # 2. Build Suggested CNN Architecture (with Bonus Dropout)
    # Input (128x128x3) -> Conv2D(32) -> MaxPooling -> Conv2D(64) -> MaxPooling -> Flatten -> Dense(128) -> Output(2 Softmax)
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Regularization (Bonus Task)
        Dropout(0.5),
        
        # Flattening 2D feature maps into 1D vector
        Flatten(),
        
        # Dense Fully Connected Layers
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.summary()

    # 3. Compile Model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train Model
    print(f"\n--- Training CNN Model for {EPOCHS} Epochs ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # 5. Evaluate Final Performance
    val_loss, val_acc = model.evaluate(val_generator)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")

    # 6. Save Trained Model (.keras format)
    model_save_path = os.path.join(base_dir, "face_mask_model.keras")
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")

    # 7. Plot Accuracy & Loss Graphs
    plot_path = os.path.join(base_dir, "accuracy_loss_plot.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2563eb', linewidth=2.5)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#16a34a', linewidth=2.5, linestyle='--')
    axes[0].set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Loss Plot
    axes[1].plot(history.history['loss'], label='Train Loss', color='#dc2626', linewidth=2.5)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='#ea580c', linewidth=2.5, linestyle='--')
    axes[1].set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Accuracy & Loss plot saved to {plot_path}")

    # 8. Test on Sample Unseen Images and Save Predictions Plot
    test_and_visualize(model, val_generator, base_dir, class_indices)

def test_and_visualize(model, val_generator, base_dir, class_indices):
    inv_class_map = {v: k for k, v in class_indices.items()}
    
    # Get a batch of images and labels from validation generator
    val_images, val_labels = next(val_generator)
    predictions = model.predict(val_images)
    
    plt.figure(figsize=(12, 10))
    num_samples = min(9, len(val_images))
    
    for i in range(num_samples):
        plt.subplot(3, 3, i + 1)
        img = val_images[i]
        true_idx = np.argmax(val_labels[i])
        pred_idx = np.argmax(predictions[i])
        confidence = predictions[i][pred_idx] * 100
        
        true_label = inv_class_map[true_idx]
        pred_label = inv_class_map[pred_idx]
        
        color = 'green' if true_idx == pred_idx else 'red'
        
        plt.imshow(img)
        plt.title(f"Pred: {pred_label} ({confidence:.1f}%)\nTrue: {true_label}", color=color, fontweight='bold', fontsize=11)
        plt.axis('off')
        
    plt.tight_layout()
    pred_plot_path = os.path.join(base_dir, "sample_predictions.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"Sample predictions visualization saved to {pred_plot_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_face_mask_model(base_dir)
