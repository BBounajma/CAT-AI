import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Disable Lightning's distributed training to avoid DDP errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PL_TRAINER_DEFAULTS_ACCELERATOR'] = 'cpu'

print("="*70)
print("Loading Trained SAINT Models")
print("="*70)
print()

# Try loading joblib files first, but fall back to .pth files if needed
print("Loading models...")
gpu_model_path = 'trained_saint_gpu.joblib'
cpu_model_path = 'trained_saint_cpu.joblib'
gpu_pth_path = 'trained_saint_gpu.pth'
cpu_pth_path = 'trained_saint_cpu.pth'

classifier_gpu = None
classifier_cpu = None

# Try joblib first
try:
    classifier_gpu = joblib.load(gpu_model_path)
    print(f"✓ GPU model loaded from joblib file")
except Exception as e:
    print(f"⚠ Could not load GPU model from joblib: {e}")

try:
    classifier_cpu = joblib.load(cpu_model_path)
    print(f"✓ CPU model loaded from joblib file")
except Exception as e:
    print(f"⚠ Could not load CPU model from joblib: {e}")

print()

# Load preprocessing info
print("Loading preprocessing info...")
try:
    preprocessing_info = joblib.load("preprocessor.joblib")
    print(f"✓ Preprocessing info loaded")
    print(f"  Numerical columns: {preprocessing_info.get('numerical_cols', [])}")
    print(f"  Categorical columns: {preprocessing_info.get('categorical_cols', [])}")
except FileNotFoundError:
    print(f"✗ Preprocessing info not found")
    preprocessing_info = None

print()

# Load and prepare test data
print("Loading and preparing test data...")
df = pd.read_csv("Data/new_data2.csv")

# Rename columns
df.rename(columns={
    'roof_type(Bamboo/Timber-Heavy roof=0; Bamboo/Timber-Light roof=1; RCC/RB/RBC=2)': 'roof_type',
    'district_distance_to_earthquakecenter(mi)': 'distance'
}, inplace=True)

# Feature and target setup
num_cols = [
    "PGA_g",
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "per-height_ft_pre_eq",
    # Binary indicator features
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other"
]

target = "damage_grade"
to_drop = [
    "technical_solution_proposed",
    "condition_post_eq",
    "vdcmun_id",
    "ward_id",
    "land_surface_condition",
    "has_superstructure_mud_mortar_brick",
    "position",
    "has_superstructure_adobe_mud"
]

multi_class_cat_cols = [
    'foundation_type',
    'roof_type',
    'ground_floor_type'
]

# Drop unnecessary columns
df.drop(columns=to_drop, inplace=True)

# Adjust target variable
y = df["damage_grade"] - 1
df.drop("damage_grade", axis=1, inplace=True)

# Preprocess features
df_processed = df.copy()

# Impute categorical features
for col in multi_class_cat_cols:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

# Convert categorical columns to category dtype
for col in multi_class_cat_cols:
    df_processed[col] = df_processed[col].astype('category')

# Impute numerical features with median
df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())

# Scale numerical features using the saved scaler
if preprocessing_info and 'scaler' in preprocessing_info:
    scaler = preprocessing_info['scaler']
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

X_test = df_processed
print(f"✓ Test data prepared with shape: {X_test.shape}")
print(f"  Features: {list(X_test.columns)}")
print()

# ============================================================================
# Test GPU Model
# ============================================================================

if classifier_gpu is not None:
    print("="*70)
    print("Testing GPU Model")
    print("="*70)
    
    try:
        # Check if the model is properly fitted
        if not hasattr(classifier_gpu, 'trainer') or classifier_gpu.trainer is None:
            print("⚠ Joblib-loaded model is not properly fitted, loading from .pth file instead...")
            # Load the .pth file and create a fresh classifier
            from deeptab.models import SAINTClassifier
            classifier_gpu = SAINTClassifier(d_model=32, n_heads=8, n_layers=3, attn_dropout=0.1)
            
            # Initialize with dummy data to create the internal model
            X_dummy = np.random.randn(10, X_test.shape[1]).astype(np.float32)
            y_dummy = np.random.randint(0, 4, 10)
            
            # Create DataFrame with proper columns
            X_dummy_df = pd.DataFrame(X_dummy, columns=X_test.columns)
            for col in multi_class_cat_cols:
                if col in X_dummy_df.columns:
                    X_dummy_df[col] = X_dummy_df[col].astype('category')
            
            print("  Initializing model with dummy training...")
            classifier_gpu.fit(X_dummy_df, y_dummy, max_epochs=0, batch_size=10)
            
            # Load the trained weights from .pth
            print(f"  Loading trained weights from {gpu_pth_path}...")
            state_dict = torch.load(gpu_pth_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            classifier_gpu.task_model.load_state_dict(state_dict)
        
        # Ensure model is on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if hasattr(classifier_gpu, 'task_model') and classifier_gpu.task_model is not None:
            classifier_gpu.task_model.to(device)
            classifier_gpu.task_model.eval()
        
        print()
        
        # Make predictions
        print("Making predictions on test data...")
        y_pred_gpu = classifier_gpu.predict(X_test)
        
        print(f"✓ GPU model predictions generated")
        print(f"  Sample predictions: {y_pred_gpu[:10]}")
        print(f"  Unique classes: {np.unique(y_pred_gpu)}")
        print(f"  Prediction shape: {y_pred_gpu.shape}")
        
        # Calculate accuracy
        try:
            accuracy_gpu = accuracy_score(y, y_pred_gpu)
            print(f"  Accuracy: {accuracy_gpu:.4f}")
            print()
            print("Classification Report:")
            print(classification_report(y, y_pred_gpu))
        except Exception as e:
            print(f"  Note: Could not calculate accuracy: {e}")
        
    except Exception as e:
        print(f"✗ Error testing GPU model: {e}")
        import traceback
        traceback.print_exc()

print()

# ============================================================================
# Test CPU Model
# ============================================================================

if classifier_cpu is not None:
    print("="*70)
    print("Testing CPU Model")
    print("="*70)
    
    try:
        # Check if the model is properly fitted
        if not hasattr(classifier_cpu, 'trainer') or classifier_cpu.trainer is None:
            print("⚠ Joblib-loaded model is not properly fitted, loading from .pth file instead...")
            # Load the .pth file and create a fresh classifier
            from deeptab.models import SAINTClassifier
            classifier_cpu = SAINTClassifier(d_model=32, n_heads=8, n_layers=3, attn_dropout=0.1)
            
            # Initialize with dummy data to create the internal model
            X_dummy = np.random.randn(10, X_test.shape[1]).astype(np.float32)
            y_dummy = np.random.randint(0, 4, 10)
            
            # Create DataFrame with proper columns
            X_dummy_df = pd.DataFrame(X_dummy, columns=X_test.columns)
            for col in multi_class_cat_cols:
                if col in X_dummy_df.columns:
                    X_dummy_df[col] = X_dummy_df[col].astype('category')
            
            print("  Initializing model with dummy training...")
            classifier_cpu.fit(X_dummy_df, y_dummy, max_epochs=0, batch_size=10)
            
            # Load the trained weights from .pth
            print(f"  Loading trained weights from {cpu_pth_path}...")
            state_dict = torch.load(cpu_pth_path, map_location='cpu')
            classifier_cpu.task_model.load_state_dict(state_dict)
        
        # Ensure model is on CPU
        if hasattr(classifier_cpu, 'task_model') and classifier_cpu.task_model is not None:
            classifier_cpu.task_model.to('cpu')
            classifier_cpu.task_model.eval()
        
        print()
        
        # Make predictions
        print("Making predictions on test data...")
        y_pred_cpu = classifier_cpu.predict(X_test)
        
        print(f"✓ CPU model predictions generated")
        print(f"  Sample predictions: {y_pred_cpu[:10]}")
        print(f"  Unique classes: {np.unique(y_pred_cpu)}")
        print(f"  Prediction shape: {y_pred_cpu.shape}")
        
        # Calculate accuracy
        try:
            accuracy_cpu = accuracy_score(y, y_pred_cpu)
            print(f"  Accuracy: {accuracy_cpu:.4f}")
            print()
            print("Classification Report:")
            print(classification_report(y, y_pred_cpu))
        except Exception as e:
            print(f"  Note: Could not calculate accuracy: {e}")
        
    except Exception as e:
        print(f"✗ Error testing CPU model: {e}")
        import traceback
        traceback.print_exc()

print()

# ============================================================================
# Model Consistency Check
# ============================================================================

if classifier_gpu is not None and classifier_cpu is not None:
    print("="*70)
    print("Model Consistency Check")
    print("="*70)
    
    try:
        if 'y_pred_gpu' in locals() and 'y_pred_cpu' in locals():
            match_rate = np.mean(y_pred_gpu == y_pred_cpu)
            print(f"✓ Prediction match rate: {match_rate:.4f}")
            if match_rate == 1.0:
                print("  Both models produce identical predictions!")
            else:
                num_diff = np.sum(y_pred_gpu != y_pred_cpu)
                print(f"  ⚠ {num_diff} predictions differ between GPU and CPU models")
    except Exception as e:
        print(f"✗ Error in consistency check: {e}")

print()
print("="*70)
print("Test Complete")
print("="*70)
print("="*70)
