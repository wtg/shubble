import os
import random
import numpy as np
import tensorflow as tf

# --- 1. SET RANDOM SEEDS FOR REPRODUCIBILITY ---
# This must be done before any other imports or code
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import gc
import joblib 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# --- MODULAR IMPORT ---
from . import stop_location
from .stop_location import MODELS_DIR 


# --- LSTM-SPECIFIC CONSTANTS ---
SEQUENCE_LENGTH = 5

SEQ_FEATURES = [
    'distance_to_next_stop', 
    'speed_mph', 
    'heading_degrees',
    'lat_delta_1',
    'lon_delta_1',
    'speed_delta_1'
]

CTX_FEATURES = [
    'time_sin', 
    'time_cos',
    'day_sin', 
    'day_cos'
]

TARGET_NAME = 'ETA_seconds'

def create_sliding_window_dataset(df, segment_index_col, sequence_length):
    """Creates a sliding window dataset for a time-series model."""
    
    # Scalers
    seq_scaler = MinMaxScaler()
    ctx_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Copy unscaled speed before scaling
    df['unscaled_speed'] = df['speed_mph']

    # Scale data
    df[SEQ_FEATURES] = seq_scaler.fit_transform(df[SEQ_FEATURES])
    df[CTX_FEATURES] = ctx_scaler.fit_transform(df[CTX_FEATURES])
    df[TARGET_NAME] = target_scaler.fit_transform(df[[TARGET_NAME]])

    # Arrays to hold the windowed data
    all_X_seq = []
    all_X_ctx = []
    all_X_seg = []
    all_y = []
    all_speeds = [] 

    grouped = df.groupby('vehicle_id')

    print(f"Creating sliding windows for {len(grouped)} vehicles...")

    for vehicle_id, group in grouped:
        seq_data = group[SEQ_FEATURES].values
        ctx_data = group[CTX_FEATURES].values
        seg_data = group[segment_index_col].values
        target_data = group[TARGET_NAME].values
        speed_data_unscaled = group['unscaled_speed'].values 

        for i in range(len(group) - sequence_length):
            window_end = i + sequence_length
            
            all_X_seq.append(seq_data[i:window_end])
            all_X_ctx.append(ctx_data[window_end - 1])
            all_X_seg.append(seg_data[window_end - 1])
            all_y.append(target_data[window_end - 1])
            all_speeds.append(speed_data_unscaled[window_end - 1]) 

    if not all_X_seq:
        print("Warning: No windows were created. Check data and sequence length.")
        return None, None, None, None, None, None

    X_seq = np.array(all_X_seq)
    X_ctx = np.array(all_X_ctx)
    X_seg = np.array(all_X_seg)
    y = np.array(all_y)
    speeds = np.array(all_speeds) 

    print(f"Created {len(y)} total windows.")
    
    X_inputs = {
        'sequential_input': X_seq,
        'context_input': X_ctx,
        'segment_input': X_seg
    }
    
    return X_inputs, y, seq_scaler, ctx_scaler, target_scaler, speeds


def build_and_train_lstm(train_data, test_data, scalers, encoder, n_segments, sequence_length):
    """
    Builds, trains, evaluates, and SAVES the model and scalers.
    """
    (X_train, y_train) = train_data
    (X_test, y_test, speeds_test) = test_data
    
    # Unpack artifacts
    seq_scaler, ctx_scaler, target_scaler = scalers
    segment_encoder = encoder

    # --- 1. Define Manual Model Architecture ---
    # This is the 32/16 configuration that gave the best results
    print("Building model with manual parameters (32/16 LSTM units)...")

    # Input 1: Sequential
    seq_input = Input(shape=(sequence_length, len(SEQ_FEATURES)), name='sequential_input')
    lstm_out = LSTM(32, activation='relu', return_sequences=True)(seq_input)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(16, activation='relu')(lstm_out)
    
    # Input 2: Contextual
    ctx_input = Input(shape=(len(CTX_FEATURES),), name='context_input')
    
    # Input 3: Segment Embedding
    seg_input = Input(shape=(1,), name='segment_input')
    seg_embedding = Embedding(input_dim=n_segments, output_dim=8, name='segment_embedding')(seg_input)
    seg_embedding = Flatten()(seg_embedding)
    
    # Concatenate
    concatenated = Concatenate()([lstm_out, ctx_input, seg_embedding])
    
    # Dense Layers
    dense_1 = Dense(32, activation='relu')(concatenated)
    dense_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(16, activation='relu')(dense_1)
    output = Dense(1, activation='linear')(dense_2)
    
    model = Model(inputs=[seq_input, ctx_input, seg_input], outputs=output)
    
    # Explicitly use Adam with default learning rate for consistency
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()

    # --- 2. Train Model ---
    print(f"[Global Model] Training LSTM model...")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=2
    )

    # --- 3. Evaluate Model ---
    print("Evaluating model...")
    y_pred_scaled = model.predict(X_test)
    
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # --- Robust RMSE Calculation ---
    print("\n--- Model Performance Breakdown ---")
    
    y_test_flat = y_test_orig.squeeze()
    y_pred_flat = y_pred.squeeze()

    moving_mask = (speeds_test > 1)
    stopped_mask = (speeds_test <= 1)

    moving_count = moving_mask.sum()
    stopped_count = stopped_mask.sum()

    if moving_count > 0:
        rmse_moving = np.sqrt(mean_squared_error(y_test_flat[moving_mask], y_pred_flat[moving_mask]))
        print(f"MOVING pings (speed > 1) RMSE: {rmse_moving:.2f} seconds ({moving_count} samples)")
    else:
        print(f"MOVING pings (speed > 1) RMSE: N/A (0 samples)")

    if stopped_count > 0:
        rmse_stopped = np.sqrt(mean_squared_error(y_test_flat[stopped_mask], y_pred_flat[stopped_mask]))
        print(f"STOPPED pings (speed <= 1) RMSE: {rmse_stopped:.2f} seconds ({stopped_count} samples)")
    else:
        print(f"STOPPED pings (speed <= 1) RMSE: N/A (0 samples)")

    rmse_overall = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    print(f"OVERALL Model RMSE: {rmse_overall:.2f} seconds\n")

    # --- 4. Plot Predictions ---
    print("Generating plots...")
    
    plot_title = f"Global LSTM Model\nActual vs Predicted (Last 200 samples)"
    plt.figure(figsize=(12, 7))
    plt.plot(y_test_flat[-200:], label='Actual ETA', marker='o', markersize=5, linestyle='None')
    plt.plot(y_pred_flat[-200:], label='Predicted ETA', marker='x', markersize=5, linestyle='None')
    plt.xlabel('Sample Index') 
    plt.ylabel('ETA (seconds)')
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = stop_location._sanitize_filename(f"1_lstm_global_predictions_timeseries")
    plt.savefig(stop_location.PLOTS_DIR / f'{filename}.png')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.3, s=10)
    min_val = min(y_test_flat.min(), y_pred_flat.min())
    max_val = max(y_test_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual ETA (seconds)')
    plt.ylabel('Predicted ETA (seconds)')
    plt.title('Prediction vs. Actual ETA')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = stop_location._sanitize_filename(f"2_lstm_global_predictions_scatter")
    plt.savefig(stop_location.PLOTS_DIR / f'{filename}.png')
    plt.close()

    errors = y_test_flat - y_pred_flat
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_flat, errors, alpha=0.3, s=10)
    plt.axhline(0, color='red', linestyle='--', label='No Error')
    plt.xlabel('Predicted ETA (seconds)')
    plt.ylabel('Error (Actual - Predicted)')
    plt.title('Residual Plot (Error vs. Prediction)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = stop_location._sanitize_filename(f"3_lstm_global_predictions_residuals")
    plt.savefig(stop_location.PLOTS_DIR / f'{filename}.png')
    plt.close()
    
    # --- 5. Save the Model AND Artifacts ---
    print("Saving model and artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Keras Model
    model_save_path = MODELS_DIR / "global_lstm_eta_model.keras"
    model.save(model_save_path)
    
    # Save Scalers and Encoder via Joblib
    joblib.dump(seq_scaler, MODELS_DIR / 'seq_scaler.pkl')
    joblib.dump(ctx_scaler, MODELS_DIR / 'ctx_scaler.pkl')
    joblib.dump(target_scaler, MODELS_DIR / 'target_scaler.pkl')
    joblib.dump(segment_encoder, MODELS_DIR / 'segment_encoder.pkl')
    
    print(f"Model saved to {model_save_path}")
    print(f"Scalers and Encoder saved to {MODELS_DIR}")
    
    print(f"Global Model Root Mean Squared Error (RMSE): {rmse_overall:.2f} seconds")


# --- MAIN FUNCTION ---
def main():
    
    print("Starting LSTM ETA Prediction Process (Global Model)...")
    try:
        with stop_location.DATASET_PATH.open('r') as f:
            df = pd.read_csv(f, nrows=200000)
            
    except FileNotFoundError:
        print(f"Error: '{stop_location.DATASET_PATH}' not found.")
        return
        
    print(f"Original data columns: {df.columns.to_list()}")
        
    print("Optimizing data types...")
    df['latitude'] = df['latitude'].astype('float32')
    df['longitude'] = df['longitude'].astype('float32')
    df['speed_mph'] = df['speed_mph'].astype('float32')
    df['heading_degrees'] = df['heading_degrees'].astype('float32')
    df['vehicle_id'] = df['vehicle_id'].astype('category')
    print("Data types optimized.")
    
    separated_dataframes, processed_df = stop_location.process_vehicle_data(df)
    
    print("\n--- Preprocessing Complete ---")
    
    del processed_df
    gc.collect() 
    print("Freed memory by deleting full processed DataFrame.")
        
    stop_location.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    global_cache = stop_location._ROUTES_DATA_CACHE
    if global_cache is None:
        print("ERROR: Route cache is not populated. Cannot validate segments.")
        route_stop_lists = {}
    else:
        route_stop_lists = {
            route: data["STOPS"] 
            for route, data in global_cache.items() 
            if "STOPS" in data
        }
    
    print("\n--- Combining Segments for Global LSTM Model ---")
    
    all_moving_segments_df = []
    
    for (route_name, segment_id), group_df in separated_dataframes.items():
        
        if segment_id.startswith('AT_'):
            continue
        
        try:
            parts = segment_id.split('_')
            if parts[0] == 'From' and parts[1] == parts[3]:
                continue
        except IndexError:
            pass 

        if segment_id.startswith('From_') and route_name in route_stop_lists:
            try:
                parts = segment_id.split('_')
                from_stop = parts[1]
                to_stop = parts[3]
                
                stop_list = route_stop_lists[route_name]
                
                if from_stop in stop_list and to_stop in stop_list:
                    from_index = stop_list.index(from_stop)
                    to_index = stop_list.index(to_stop)
                    
                    if (to_index - from_index) > 1:
                        continue
                
            except (IndexError, ValueError):
                pass
        
        all_moving_segments_df.append(group_df)
    
    if not all_moving_segments_df:
        print("No valid 'moving' segments found to train on. Exiting.")
        return
        
    master_training_df = pd.concat(all_moving_segments_df, ignore_index=True)
    
    print(f"Combined {len(all_moving_segments_df)} segments into one "
          f"master DataFrame with {len(master_training_df)} total rows.")
          
    all_model_features = SEQ_FEATURES + CTX_FEATURES + [TARGET_NAME, 'speed_mph']
    df_clean = master_training_df.dropna(subset=all_model_features).copy()
    
    print("Encoding 'segment_id' for Embedding layer...")
    segment_encoder = LabelEncoder()
    df_clean['segment_index'] = segment_encoder.fit_transform(df_clean['segment_id'])
    n_segments = len(segment_encoder.classes_)
    print(f"Found {n_segments} unique segments.")
    
    
    if len(df_clean) < 1000:
        print(f"Not enough clean data ({len(df_clean)} rows) to train global model.")
        return
        
    # Unpack ALL scalers
    X_inputs, y, seq_scaler, ctx_scaler, target_scaler, speeds = create_sliding_window_dataset(
        df_clean, 
        'segment_index', 
        SEQUENCE_LENGTH
    )
    
    if X_inputs is None:
        print("Failed to create dataset. Exiting.")
        return

    print("Splitting data into train and test sets...")
    
    # IMPORTANT: Use the exact same random_state as the seed for consistency
    X_seq_train, X_seq_test, \
    X_ctx_train, X_ctx_test, \
    X_seg_train, X_seg_test, \
    y_train, y_test, \
    speeds_train, speeds_test = train_test_split(
        X_inputs['sequential_input'],
        X_inputs['context_input'],
        X_inputs['segment_input'],
        y,
        speeds, 
        test_size=0.2,
        random_state=SEED # Use the global seed constant
    )

    train_data = ({
        'sequential_input': X_seq_train,
        'context_input': X_ctx_train,
        'segment_input': X_seg_train
    }, y_train)
    
    test_data = ({
        'sequential_input': X_seq_test,
        'context_input': X_ctx_test,
        'segment_input': X_seg_test
    }, y_test, speeds_test)
    
    
    try:
        # Pass all artifacts to build_and_train
        build_and_train_lstm(
            train_data, 
            test_data, 
            (seq_scaler, ctx_scaler, target_scaler), # Tuple of scalers
            segment_encoder, # Segment encoder
            n_segments,
            SEQUENCE_LENGTH
        )
    except Exception as e:
        print(f"!! FAILED to build or train global LSTM model. Error: {e}")
        import traceback
        traceback.print_exc()

    
    print(f"\nAll plots have been saved to '{stop_location.PLOTS_DIR}' directory.")
    print(f"Model has been saved to '{MODELS_DIR}' directory.")

if __name__ == '__main__':
    main()