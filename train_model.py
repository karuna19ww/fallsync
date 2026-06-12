"""
FallSync Model Training Script
Trains and saves the fall detection model
"""

import numpy as np
import json
import os
from fall_detection_model import FallDetectionModel, WINDOW_SIZE, N_FEATURES, N_CLASSES, EPOCHS

def generate_realistic_data(n_samples=2000):
    """
    Generate realistic fall detection training data
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        X: Input features (samples, time_steps, features)
        y: Labels (samples,)
    """
    
    print("Generating training data...")
    
    # Fall data: sudden acceleration changes
    fall_data = []
    for _ in range(n_samples // 2):
        # Simulate a fall: starts normal, then sharp spike
        sequence = np.zeros((WINDOW_SIZE, N_FEATURES))
        
        # Normal movement at start
        for i in range(20):
            sequence[i] = np.random.randn(N_FEATURES) * 0.5
        
        # Sudden fall: high acceleration spike
        for i in range(20, 45):
            sequence[i] = np.random.randn(N_FEATURES) * 2.5
        
        # After fall: stabilization
        for i in range(45, WINDOW_SIZE):
            sequence[i] = np.random.randn(N_FEATURES) * 0.8
        
        fall_data.append(sequence)
    
    fall_labels = np.ones(n_samples // 2)
    
    # No-fall data: normal movement
    no_fall_data = []
    for _ in range(n_samples // 2):
        # Random normal movement
        sequence = np.random.randn(WINDOW_SIZE, N_FEATURES) * 0.8
        no_fall_data.append(sequence)
    
    no_fall_labels = np.zeros(n_samples // 2)
    
    # Combine data
    X = np.vstack([fall_data, no_fall_data])
    y = np.hstack([fall_labels, no_fall_labels])
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    print(f"Data shape: {X.shape}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")
    
    return X, y


def main():
    """Main training function"""
    
    print("=" * 60)
    print("FallSync Model Training")
    print("=" * 60)
    
    # Generate data
    X, y = generate_realistic_data(n_samples=3000)
    
    # Initialize model
    print("\nInitializing model...")
    model = FallDetectionModel()
    model.build_model()
    print("Model architecture created")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = model.prepare_data(X, y)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\nTraining model...")
    print("This may take a few minutes...")
    history = model.train(X_train, y_train, X_test, y_test, epochs=EPOCHS)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    
    # Save model
    print("\nSaving model...")
    model.save('fall_detection_model.h5', 'scaler.pkl')
    
    # Convert to TFLite
    print("Converting to TensorFlow Lite...")
    model.convert_to_tflite('fall_detection_model.tflite')
    
    # Save training metrics
    print("Saving training metrics...")
    metrics_data = {
        'model_type': 'CNN',
        'window_size': WINDOW_SIZE,
        'n_features': N_FEATURES,
        'n_classes': N_CLASSES,
        'epochs': EPOCHS,
        'test_metrics': {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall'])
        }
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nModel files:")
    print("- fall_detection_model.h5 (Keras model)")
    print("- scaler.pkl (Feature scaler)")
    print("- fall_detection_model.tflite (Mobile model)")
    print("- model_metrics.json (Metrics)")
    print("\nTo run the server:")
    print("  python app.py")


if __name__ == '__main__':
    main()