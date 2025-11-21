"""
Model Conversion Script for NeuroStress Pro
Converts older Keras/TensorFlow models to compatible format
"""

import tensorflow as tf
from tensorflow import keras
import sys
from pathlib import Path

def convert_model(input_path='model_c.h5', output_path='model_c_converted.h5'):
    """
    Convert older Keras model to compatible format
    
    Args:
        input_path: Path to original model
        output_path: Path to save converted model
    """
    print("🔄 Starting model conversion...")
    print(f"📁 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"🔧 TensorFlow version: {tf.__version__}")
    
    try:
        # Method 1: Try loading with compile=False
        print("\n📥 Attempting to load model (Method 1: compile=False)...")
        model = keras.models.load_model(input_path, compile=False)
        print("✅ Model loaded successfully!")
        
    except Exception as e1:
        print(f"❌ Method 1 failed: {e1}")
        
        try:
            # Method 2: Load weights only
            print("\n📥 Attempting to load model (Method 2: weights only)...")
            
            # First, let's inspect the model structure
            import h5py
            with h5py.File(input_path, 'r') as f:
                print("\n📊 Model file structure:")
                print(f"  Keys: {list(f.keys())}")
                if 'model_config' in f.attrs:
                    print("  Has model_config")
            
            # Try to load with custom objects
            model = tf.keras.models.load_model(
                input_path,
                compile=False,
                custom_objects=None
            )
            print("✅ Model loaded successfully!")
            
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            try:
                # Method 3: Build model from config and load weights
                print("\n📥 Attempting to load model (Method 3: rebuild from config)...")
                
                import json
                import h5py
                
                with h5py.File(input_path, 'r') as f:
                    if 'model_config' in f.attrs:
                        model_config = f.attrs['model_config']
                        if isinstance(model_config, bytes):
                            model_config = model_config.decode('utf-8')
                        model_config = json.loads(model_config)
                        
                        # Create model from config
                        model = tf.keras.models.model_from_json(json.dumps(model_config))
                        
                        # Load weights
                        model.load_weights(input_path)
                        print("✅ Model rebuilt and weights loaded!")
                    else:
                        raise Exception("No model_config found in file")
                        
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
                print("\n❌ All conversion methods failed!")
                print("\n💡 Possible solutions:")
                print("1. Use the exact TensorFlow version that created the model")
                print("2. Retrain the model with current TensorFlow version")
                print("3. Check if the model file is corrupted")
                return False
    
    # Model loaded successfully, now save it
    print(f"\n📊 Model Summary:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Total params: {model.count_params():,}")
    
    # Recompile the model
    print("\n🔧 Recompiling model with current optimizer...")
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Save in H5 format
    print(f"\n💾 Saving converted model to: {output_path}")
    model.save(output_path, save_format='h5')
    print("✅ Model saved in H5 format!")
    
    # Also save in SavedModel format (recommended)
    savedmodel_path = output_path.replace('.h5', '_savedmodel')
    print(f"\n💾 Also saving in SavedModel format to: {savedmodel_path}")
    model.save(savedmodel_path, save_format='tf')
    print("✅ Model saved in SavedModel format!")
    
    # Verify the converted model works
    print("\n🧪 Verifying converted model...")
    import numpy as np
    test_input = np.random.rand(1, 48, 48, 1).astype('float32')
    prediction = model.predict(test_input, verbose=0)
    print(f"  Test prediction shape: {prediction.shape}")
    print(f"  Test prediction sum: {prediction.sum():.4f}")
    print("✅ Model verification successful!")
    
    print("\n" + "="*60)
    print("✨ MODEL CONVERSION COMPLETED SUCCESSFULLY! ✨")
    print("="*60)
    print(f"\n📁 Converted files:")
    print(f"  1. {output_path} (H5 format)")
    print(f"  2. {savedmodel_path}/ (SavedModel format - recommended)")
    print(f"\n💡 Next steps:")
    print(f"  1. Rename {output_path} to model_c.h5")
    print(f"  2. Or update dashboard to use the SavedModel format")
    print(f"  3. Restart the dashboard")
    
    return True


def test_converted_model(model_path='model_c_converted.h5'):
    """Test if the converted model works"""
    print("\n🧪 Testing converted model...")
    
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        
        # Test prediction
        import numpy as np
        test_input = np.random.rand(1, 48, 48, 1).astype('float32')
        prediction = model.predict(test_input, verbose=0)
        
        print(f"📊 Test Results:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Prediction shape: {prediction.shape}")
        print(f"  Prediction probabilities: {prediction[0]}")
        print(f"  Predicted class: {np.argmax(prediction[0])}")
        print("✅ Model is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("🧠 NEUROSTRESS PRO - MODEL CONVERTER")
    print("="*60)
    
    # Check if model file exists
    input_model = 'model_c.h5'
    if not Path(input_model).exists():
        print(f"❌ Error: {input_model} not found in current directory!")
        print(f"📁 Current directory: {Path.cwd()}")
        print("\n💡 Make sure you're running this script from the same directory as model_c.h5")
        sys.exit(1)
    
    # Convert the model
    success = convert_model(input_model, 'model_c_converted.h5')
    
    if success:
        # Test the converted model
        print("\n" + "="*60)
        test_converted_model('model_c_converted.h5')
        
        print("\n" + "="*60)
        print("🎉 ALL DONE! Your model is ready to use!")
        print("="*60)
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
        sys.exit(1)
