from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def sample_data(df, n_samples=1000):
    no_finding = df[df['Is_Finding'] == 0].sample(n_samples, random_state=42)
    is_finding = df[df['Is_Finding'] == 1].sample(n_samples, random_state=42)
    return pd.concat([no_finding, is_finding])

def create_model_for_gridsearch(learning_rate=1e-3, dense_units=256):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid', dtype='float32')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    data_path = "Data_Entry_2017_v2020.csv"
    images_folder = 'final_images/images'
    batch_size = 32 * strategy.num_replicas_in_sync
    epochs = 10
    img_size = (160, 160)

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    df = pd.read_csv(data_path)
    df['Finding Labels'] = df['Finding Labels'].str.split('|')
    df['Is_Finding'] = df['Finding Labels'].apply(lambda x: 1 if x != ['No Finding'] else 0)

    # Sample 1000 images for each class
    sampled_df = sample_data(df, n_samples=1000)
    X_sampled = sampled_df['Image Index']
    y_sampled = sampled_df['Is_Finding']

    # Create datasets for GridSearchCV
    sampled_ds = create_dataset(X_sampled.values, y_sampled.values, batch_size=batch_size, 
                                img_size=img_size, image_dir=images_folder, is_training=True)

    # Wrap the model for GridSearchCV
    model = KerasClassifier(build_fn=create_model_for_gridsearch, epochs=5, batch_size=batch_size, verbose=0)

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'dense_units': [128, 256]
    }

    # Perform GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)
    grid_result = grid.fit(sampled_ds)

    print(f"Best parameters: {grid_result.best_params_}")
    print(f"Best score: {grid_result.best_score_}")

    # Train the best model on the entire dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['Image Index'], df['Is_Finding'], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42)

    train_ds = create_dataset(X_train.values, y_train.values, batch_size=batch_size, 
                              img_size=img_size, image_dir=images_folder, is_training=True)
    val_ds = create_dataset(X_val.values, y_val.values, batch_size=batch_size, 
                            img_size=img_size, image_dir=images_folder, is_training=False)

    best_model = create_model_for_gridsearch(
        learning_rate=grid_result.best_params_['learning_rate'],
        dense_units=grid_result.best_params_['dense_units']
    )

    checkpoint = ModelCheckpoint(
        'models/chest_xray_model_best.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=3,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    history = best_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping]
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('results/training_history_full.csv', index=False)

    # Evaluate on test set
    test_ds = create_dataset(X_test.values, y_test.values, batch_size=batch_size, 
                             img_size=img_size, image_dir=images_folder, is_training=False)
    test_loss, test_acc = best_model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc}")

if __name__ == '__main__':
    main()
