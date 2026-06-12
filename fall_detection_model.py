"""
Pre-Collapse Guardian - Fall Detection Model
Machine Learning model for real-time fall detection using accelerometer data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import joblib
import json

# Constants
WINDOW_SIZE = 60  # Time steps (3 seconds at 20Hz sampling)
N_FEATURES = 3    # X, Y, Z acceleration
N_CLASSES = 2     # Fall, No-Fall
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

class FallDetectionModel:
    """
    Convolutional Neural Network for fall detection
    Processes 3-axis accelerometer data to classify fall events
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.input_shape = (WINDOW_SIZE, N_FEATURES)
        
    def build_model(self) -> keras.Model:
        """
        Build CNN model architecture
        Input: (batch_size, 60, 3) - time steps x acceleration axes
        Output: (batch_size, 2) - [prob_no_fall, prob_fall]
        """
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Conv block 1
            layers.Conv1D(
                filters=32,
                kernel_size=3,
                activation='relu',
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # Conv block 2
            layers.Conv1D(
                filters=64,
                kernel_size=3,
                activation='relu',
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # Conv block 3
            layers.Conv1D(
                filters=128,
                kernel_size=3,
                activation='relu',
                padding='same'
            ),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(N_CLASSES, activation='softmax')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize and split data for training
        
        Args:
            X: Input features (samples, time_steps, features)
            y: Labels (samples,)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        # Reshape for scaling
        samples, timesteps, features = X.shape
        X_reshaped = X.reshape(-1, features)
        
        # Fit scaler on training data
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        X_scaled = X_scaled.reshape(samples, timesteps, features)
        
        # Convert labels to one-hot
        y_categorical = keras.utils.to_categorical(y, num_classes=N_CLASSES)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y_categorical,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = EPOCHS,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            verbose: Logging level
            
        Returns:
            Training history
        """
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        return history.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Metrics dictionary
        """
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
    
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X: Input features (samples, time_steps, features)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions and confidence scores
        """
        
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Scale input
        samples, timesteps, features = X.shape
        X_reshaped = X.reshape(-1, features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(samples, timesteps, features)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Extract fall predictions (class 1)
        fall_predictions = np.argmax(predictions, axis=1)
        fall_confidence = predictions[:, 1]  # Probability of fall
        
        if return_confidence:
            return fall_predictions, fall_confidence
        else:
            return fall_predictions
    
    def predict_single(
        self,
        acceleration_sequence: List[List[float]]
    ) -> Dict[str, float]:
        """
        Predict on a single acceleration sequence
        Real-time inference method
        
        Args:
            acceleration_sequence: List of [x, y, z] acceleration values
                                   Should contain 60 samples (3 seconds)
        
        Returns:
            Prediction dict with is_fall, confidence, and other metrics
        """
        
        if len(acceleration_sequence) < WINDOW_SIZE:
            raise ValueError(f"Sequence too short. Expected {WINDOW_SIZE}, got {len(acceleration_sequence)}")
        
        # Take only the last WINDOW_SIZE samples
        sequence = np.array(acceleration_sequence[-WINDOW_SIZE:])
        sequence = sequence.reshape(1, WINDOW_SIZE, N_FEATURES)
        
        # Normalize
        sequence_reshaped = sequence.reshape(-1, N_FEATURES)
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(1, WINDOW_SIZE, N_FEATURES)
        
        # Predict
        predictions = self.model.predict(sequence_scaled, verbose=0)
        fall_prob = predictions[0, 1]
        no_fall_prob = predictions[0, 0]
        
        # Calculate acceleration metrics
        magnitude = np.sqrt(np.sum(sequence**2, axis=1))
        max_magnitude = np.max(magnitude)
        mean_magnitude = np.mean(magnitude)
        
        return {
            'is_fall': bool(fall_prob > 0.8),
            'fall_probability': float(fall_prob),
            'no_fall_probability': float(no_fall_prob),
            'confidence': float(max(fall_prob, no_fall_prob)),
            'max_acceleration': float(max_magnitude),
            'mean_acceleration': float(mean_magnitude),
            'peak_jerk': float(np.max(np.diff(magnitude)))
        }
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """Save model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: str, scaler_path: str) -> None:
        """Load model and scaler"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
    
    def convert_to_tflite(self, output_path: str) -> None:
        """Convert model to TensorFlow Lite format for mobile deployment"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")


# Example usage and training
if __name__ == "__main__":
    # Generate sample data (replace with real dataset)
    print("Generating sample training data...")
    
    # 1000 samples per class
    n_samples = 2000
    
    # Fall data: high acceleration changes
    fall_data = np.random.randn(n_samples // 2, WINDOW_SIZE, N_FEATURES) * 2.5
    fall_data = np.abs(fall_data)
    fall_labels = np.ones(n_samples // 2)
    
    # No-fall data: lower acceleration
    no_fall_data = np.random.randn(n_samples // 2, WINDOW_SIZE, N_FEATURES) * 0.8
    no_fall_data = np.abs(no_fall_data)
    no_fall_labels = np.zeros(n_samples // 2)
    
    # Combine
    X = np.vstack([fall_data, no_fall_data])
    y = np.hstack([fall_labels, no_fall_labels])
    
    print(f"Data shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")
    
    # Initialize and train model
    print("\nInitializing model...")
    model = FallDetectionModel()
    model.build_model()
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test = model.prepare_data(X, y)
    
    print("Training model...")
    history = model.train(X_train, y_train, X_test, y_test, epochs=30)
    
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    
    # Test prediction on single sample
    print("\nTesting single prediction...")
    test_sample = X_test[0:1]
    prediction = model.predict_single(test_sample[0])
    print(json.dumps(prediction, indent=2))
    
    # Save models
    print("\nSaving models...")
    model.save('fall_detection_model.h5', 'scaler.pkl')
    model.convert_to_tflite('fall_detection_model.tflite')