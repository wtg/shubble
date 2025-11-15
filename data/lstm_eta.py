import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import gc

# --- NEW IMPORTS FOR TENSORFLOW & KERAS TUNER ---
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# --- MODULAR IMPORT ---
from . import stop_location
from .stop_location import MODELS_DIR 


# --- LSTM-SPECIFIC CONSTANTS (SLIDING WINDOW) ---
SEQUENCE_LENGTH = 5  # Use the last 5 pings to predict the next ETA

# Features that change at every ping (the sequence)
SEQ_FEATURES = [
    'distance_to_next_stop', 
    'speed_mph', 
    'heading_degrees',
    'lat_delta_1',
    'lon_delta_1',
    'speed_delta_1'
]

# Features that are constant for a given ping (the context)
CTX_FEATURES = [
    'time_sin', 
    'time_cos',
    'day_sin', 
    'day_cos'
]

TARGET_NAME = 'ETA_seconds'
# --- (Removed STATUS_NAME, it was from a previous bug) ---


# --- NEW LSTM MODELING FUNCTIONS ---

def create_sliding_window_dataset(df, segment_index_col, sequence_length):
    """
    Creates a sliding window dataset for a time-series model.
    
    --- MODIFIED ---
    - Correctly extracts the *unscaled speed* for error analysis
      by copying it before scaling.
    """
    
    # Scalers
    seq_scaler = MinMaxScaler()
    ctx_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # --- FIX: Copy the unscaled speed *before* scaling ---
    df['unscaled_speed'] = df['speed_mph']

    # Scale data *before* creating windows
    # Note: This will scale 'speed_mph' in SEQ_FEATURES, but 'unscaled_speed' remains
    df[SEQ_FEATURES] = seq_scaler.fit_transform(df[SEQ_FEATURES])
    df[CTX_FEATURES] = ctx_scaler.fit_transform(df[CTX_FEATURES])
    df[TARGET_NAME] = target_scaler.fit_transform(df[[TARGET_NAME]])

    # Arrays to hold the windowed data
    all_X_seq = []
    all_X_ctx = []
    all_X_seg = []
    all_y = []
    all_speeds = [] 

    # Group by vehicle so windows don't cross vehicles
    grouped = df.groupby('vehicle_id')

    print(f"Creating sliding windows for {len(grouped)} vehicles...")

    for vehicle_id, group in grouped:
        seq_data = group[SEQ_FEATURES].values
        ctx_data = group[CTX_FEATURES].values
        seg_data = group[segment_index_col].values
        target_data = group[TARGET_NAME].values
        # --- FIX: Get the unscaled speed from the copy ---
        speed_data_unscaled = group['unscaled_speed'].values 

        # Iterate to create overlapping windows
        for i in range(len(group) - sequence_length):
            window_end = i + sequence_length
            
            all_X_seq.append(seq_data[i:window_end])
            
            # The context and target are from the *last* ping in the window
            all_X_ctx.append(ctx_data[window_end - 1])
            all_X_seg.append(seg_data[window_end - 1])
            all_y.append(target_data[window_end - 1])
            # --- FIX: Append the *true* unscaled speed ---
            all_speeds.append(speed_data_unscaled[window_end - 1]) 

    if not all_X_seq:
        print("Warning: No windows were created. Check data and sequence length.")
        return None, None, None, None

    # Convert to numpy arrays
    X_seq = np.array(all_X_seq)
    X_ctx = np.array(all_X_ctx)
    X_seg = np.array(all_X_seg)
    y = np.array(all_y)
    speeds = np.array(all_speeds) 

    print(f"Created {len(y)} total windows.")
    
    # Package data
    X_inputs = {
        'sequential_input': X_seq,
        'context_input': X_ctx,
        'segment_input': X_seg
    }
    
    return X_inputs, y, target_scaler, speeds


# --- NEW: Model Building Function for KerasTuner ---
def build_model(hp, n_segments, seq_len, n_seq_features, n_ctx_features):
    """
    Builds a compiled Keras model with hyperparameters.
    This function is designed to be used by KerasTuner.
    """
    
    # --- 1. Define Hyperparameters ---
    # Layer Units
    lstm_units_1 = hp.Int('lstm_units_1', min_value=16, max_value=64, step=16)
    lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=32, step=16)
    embed_dim = hp.Int('embed_dim', min_value=4, max_value=12, step=4)
    dense_units_1 = hp.Int('dense_units_1', min_value=16, max_value=64, step=16)
    dense_units_2 = hp.Int('dense_units_2', min_value=8, max_value=32, step=8)
    
    # Dropout Rate
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1)
    
    # Learning Rate
    learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    
    
    # --- 2. Define Model Architecture ---
    seq_input = Input(shape=(seq_len, n_seq_features), name='sequential_input')
    lstm_out = LSTM(lstm_units_1, activation='relu', return_sequences=True)(seq_input)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = LSTM(lstm_units_2, activation='relu')(lstm_out)
    
    ctx_input = Input(shape=(n_ctx_features,), name='context_input')
    
    seg_input = Input(shape=(1,), name='segment_input')
    seg_embedding = Embedding(input_dim=n_segments, output_dim=embed_dim, name='segment_embedding')(seg_input)
    seg_embedding = Flatten()(seg_embedding)
    
    concatenated = Concatenate()([lstm_out, ctx_input, seg_embedding])
    
    dense_1 = Dense(dense_units_1, activation='relu')(concatenated)
    dense_1 = Dropout(dropout_rate)(dense_1)
    dense_2 = Dense(dense_units_2, activation='relu')(dense_1)
    output = Dense(1, activation='linear')(dense_2)
    
    model = Model(inputs=[seq_input, ctx_input, seg_input], outputs=output)
    
    # --- 3. Compile Model ---
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='mean_squared_error'
    )
    
    return model


# --- NEW: Evaluation Function ---
def evaluate_model(model, X_test, y_test, speeds_test, y_scaler):
    """
    Takes a trained model and evaluates it, creating plots and saving the model.
    """
    
    # --- 1. Evaluate Model ---
    print("Evaluating best model...")
    y_pred_scaled = model.predict(X_test)
    
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # --- Robust RMSE Calculation ---
    print("\n--- Best Model Performance Breakdown ---")
    
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

    # --- 2. Plot Predictions ---
    print("Generating plots...")
    
    plot_title = f"Global LSTM Model (Tuned)\nActual vs Predicted (Last 200 samples)"
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
    plt.title('Prediction vs. Actual ETA (Tuned)')
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
    plt.title('Residual Plot (Tuned)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = stop_location._sanitize_filename(f"3_lstm_global_predictions_residuals")
    plt.savefig(stop_location.PLOTS_DIR / f'{filename}.png')
    plt.close()
    
    # --- 3. Save the Model ---
    print("Saving best model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "global_lstm_eta_model.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


# --- MODIFIED MAIN FUNCTION ---
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
        
    X_inputs, y, target_scaler, speeds = create_sliding_window_dataset(
        df_clean, 
        'segment_index', 
        SEQUENCE_LENGTH
    )
    
    if X_inputs is None:
        print("Failed to create dataset. Exiting.")
        return

    print("Splitting data into train and test sets...")
    
    X_seq_train, X_seq_test, \
    X_ctx_train, X_ctx_test, \
    X_seg_train, X_seg_test, \
    y_train, y_test, \
    speeds_train, speeds_test = train_test_split(
        X_inputs['sequential_input'],
        X_inputs['context_input'],
        X_inputs['segment_input'],
        y,
        speeds, # Split this too
        test_size=0.2,
        random_state=42
    )

    # Package data into the formats KerasTuner expects
    train_data_inputs = {
        'sequential_input': X_seq_train,
        'context_input': X_ctx_train,
        'segment_input': X_seg_train
    }
    
    test_data_inputs = {
        'sequential_input': X_seq_test,
        'context_input': X_ctx_test,
        'segment_input': X_seg_test
    }
    
    # --- NEW: Get data shapes for the model builder ---
    n_seq_features = X_inputs['sequential_input'].shape[2]
    n_ctx_features = X_inputs['context_input'].shape[1]
    
    # --- NEW: KerasTuner Setup ---
    
    model_builder = lambda hp: build_model(
        hp,
        n_segments=n_segments,
        seq_len=SEQUENCE_LENGTH,
        n_seq_features=n_seq_features,
        n_ctx_features=n_ctx_features
    )

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='keras_tuner',
        project_name='shubble_eta_lstm'
    )
    
    search_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    print("\n--- Starting Hyperparameter Search ---")
    tuner.search(
        train_data_inputs,
        y_train,
        epochs=50,
        validation_data=(test_data_inputs, y_test),
        callbacks=[search_early_stopping]
    )

    print("\n--- Hyperparameter Search Complete ---")
    tuner.results_summary()

    try:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best HPs found: {best_hps.values}")

        best_model = tuner.get_best_models(num_models=1)[0]
        
        # --- Evaluate the best model ---
        evaluate_model(
            best_model, 
            test_data_inputs, 
            y_test, 
            speeds_test, 
            target_scaler  # <--- THIS IS THE FIX
        )
        
    except Exception as e:
        print(f"!! FAILED to evaluate the best model. Error: {e}")
        import traceback
        traceback.print_exc()

    
    print(f"\nAll plots have been saved to '{stop_location.PLOTS_DIR}' directory.")
    print(f"Model has been saved to '{MODELS_DIR}' directory.")

if __name__ == '__main__':
    main()