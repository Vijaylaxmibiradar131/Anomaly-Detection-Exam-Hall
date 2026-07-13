import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import cv2
from pathlib import Path
import json
from datetime import datetime

# Import config from video_analyzer
try:
    from src.video_analyzer import load_config
except ImportError:
    # Fallback if video_analyzer is not available
    def load_config(config_path='config.json'):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {
                "model": {
                    "image_size": [224, 224],
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 50
                }
            }

class ClassroomBehaviorTrainer:
    def __init__(self, dataset_path="CNN_Dataset", img_size=None, batch_size=None, config_path='config.json'):
        self.config = load_config(config_path)

        self.dataset_path = dataset_path
        self.img_size = img_size or tuple(self.config['model']['image_size'])
        self.batch_size = batch_size or self.config['model']['batch_size']
        self.learning_rate = self.config['model']['learning_rate']
        self.default_epochs = self.config['model']['epochs']

        self.classes = [
            'Normal', 'Discussing', 'Peeking', 'cheat passing', 'copying',
            'showing answer', 'suspicious', 'using copy cheat', 'using mobile'
        ]
        self.num_classes = len(self.classes)
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing dataset...")
        
        # Create data generators with augmentation
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training data
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data
        self.validation_generator = test_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Classes: {list(self.train_generator.class_indices.keys())}")
        
        return self.train_generator, self.validation_generator
    
    def build_model(self):
        """Build CNN model architecture"""
        print("Building CNN model...")

        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),

            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Global Average Pooling
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")

        return model
    
    def train_model(self, epochs=None):
        """Train the model"""
        if epochs is None:
            epochs = self.default_epochs
        print(f"Starting training for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_classroom_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(self.validation_generator)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = self.validation_generator.classes
        
        # Classification report
        class_names = list(self.validation_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        print("Classification Report:")
        print(report)
        
        # Save classification report
        Path('outputs').mkdir(exist_ok=True)
        with open('outputs/classification_report.txt', 'w') as f:
            f.write("Classification Report\n")
            f.write("===================\n\n")
            f.write(report)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate additional metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        return report, cm
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='models/classroom_behavior_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return
        
        self.model.save(filepath)
        
        # Save class indices
        Path('models').mkdir(exist_ok=True)
        class_indices = self.train_generator.class_indices
        with open('models/class_indices.json', 'w') as f:
            json.dump(class_indices, f, indent=2)
        
        print(f"Model saved to {filepath}")
        print("Class indices saved to models/class_indices.json")
    
    def load_model(self, filepath='models/classroom_behavior_model.h5'):
        """Load a trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            
            # Load class indices
            with open('class_indices.json', 'r') as f:
                class_indices = json.load(f)
            self.classes = list(class_indices.keys())
            print(f"Classes loaded: {self.classes}")
            
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

def main():
    """Main training function"""
    print("🎓 Classroom Behavior Detection - Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ClassroomBehaviorTrainer()
    
    # Load and preprocess data
    train_gen, val_gen = trainer.load_and_preprocess_data()
    
    # Build model
    model = trainer.build_model()
    
    # Display model summary
    model.summary()
    
    # Train model
    history = trainer.train_model(epochs=50)
    
    # Evaluate model
    report, cm = trainer.evaluate_model()
    
    # Save model
    trainer.save_model()
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print("📁 Model saved as 'classroom_behavior_model.h5'")
    print("📈 Training plots saved as PNG files")

if __name__ == "__main__":
    main()
