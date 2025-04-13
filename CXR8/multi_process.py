import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet121, EfficientNetB3, ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

class ChestXRayClassifier:
    def __init__(self, data_path=None, images_folder=None, img_size=(224, 224), batch_size=32, epochs=10, model_type='densenet', multi_label=True):
        self.data_path = data_path
        self.images_folder = images_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type
        self.multi_label = multi_label
        
        self.strategy = tf.distribute.MirroredStrategy()
        self.batch_size *= self.strategy.num_replicas_in_sync

        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        self.df = None
        self.unique_findings = None
        self.model = None
    
    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)

        self.df['Finding Labels'] = self.df['Finding Labels'].str.split('|')
        self.df['Is_Finding'] = self.df['Finding Labels'].apply(lambda x: 0 if x == ['No Finding'] else 1)

        if self.multi_label:
            all_findings = []
            for finding in self.df['Finding Labels']:
                all_findings.extend(finding)
            self.unique_findings = list(set(all_findings))

            if 'No Finding' in self.unique_findings:
                self.unique_findings.remove('No Finding')
            
            for finding in self.unique_findings:
                self.df[finding] = self.df['Finding Labels'].apply(lambda x: 1 if finding in x else 0)
        
        return self.df
    
    def create_dataset(self, image_indices, labels, is_training=False):
        def process_path(image_index, label):
            image_index = tf.strings.strip(tf.strings.as_string(image_index))
            image_path = tf.strings.join([self.images_folder, "/", image_index])

            def load_image(path):
                try:
                    img = tf.io.read_file(path)
                    img = tf.image.decode_jpeg(img, channels=3)
                    img = tf.image.resize(img, self.img_size)
                    img = img / 255.0

                    if is_training:
                        img = tf.image.random_flip_left_right(img)
                        img = tf.image.random_flip_up_down(img)
                        img = tf.image.random_brightness(img, max_delta=0.2)
                        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

                        crop_size = tf.random.uniform(shape=[],
                                                      minval=tf.cast(tf.cast(self.img_size[0], tf.float32) * 0.8, tf.int32),
                                                      maxval=self.img_size[0],
                                                      dtype=tf.int32)
                        
                        crop_size = tf.minimum(crop_size, self.img_size[0])
                        img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
                        img = tf.image.resize(img, self.img_size)

                        noise = tf.random.noermal(shape=tf.shape(img), mean=0.0, stddev=0.01)
                        img = tf.clip_by_value(img + noise, 0.0, 1.0)

                    return img
                except tf.errors.InvalidArgumentError:
                    print(f"Error loading image: {path}")
                    return tf.zeros(self.img_size + (3,))
            
            img = load_image(image_path)
            return img, label
        
        image_ds = tf.data.Dataset.from_tensor_slices(image_indices)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((image_ds, label_ds))

        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            ds = ds.shuffle(bufffer_size=1000)
        
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds
    
    def create_model(self, num_classes=1):
        input_shape = (*self.img_size, 3)

        with self.strategy.scope():
            if self.model_type == 'efficientnet':
                base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
            elif self.model_type == 'resnet':
                base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
            else:
                base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            x = Dropout(0.3)(x)

            x = BatchNormalization()(x)

            x = Dense(512, activation='relu')(x)
            x = Dropout(0.4)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)

            predictions = Dense(num_classes, activation='sigmoid', dtype='float32')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            for layer in base_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = True
                else:
                    layer.trainable = False
            
            trainable_layers = int(len(base_model.layers) * 0.3)
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
            
            initial_learning_rate = 1e-3
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=initial_learning_rate,
                first_decay_steps=1000,
                t_mul=2.0,
                m_mul=0.9,
                alpha=1e-5
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

            metrics = [
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.metrics.AUC(name='pr_auc', curve='PR')
            ]

            if num_classes == 1:
                metrics.apppend(tf.keras.metrics.F1Score(name='f1_score', threshold=0.5))
            
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

        return model
    
    def get_callbacks(self, model_name):
        checkpoint = ModelCheckpoint(
            f'models/{model_name}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1,
            write_graph=True
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    def train_eval(self):
        if self.df is None:
            self.load_data()
        
        if self.multi_label:
            X = self.df['Image Index']
            y = self.df[self.unique_findings]
            num_classes = len(self.unique_findings)
            model_name = 'chest_xray_multilabel'
        else:
            X = self.df['Image Index']
            y = self.df['Is_Finding']
            num_classes = 1
            model_name = 'chest_xray_binary'
        
        print("Splitting data...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if not self.multi_label else None)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val if not self.multi_label else None)

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        print("Creating datasets...")
        train_ds = self.create_dataset(X_train.values, y_train.values.astype('float32'), is_training=True)
        val_ds = self.create_dataset(X_val.values, y_val.values.astype('float32'), is_training=False)
        test_ds = self.create_dataset(X_test.values, y_test.values.astype('float32'), is_training=False)

        print(f"Creating {model_name} model...")
        self.model = self.create_model(num_classes=num_classes)
        self.model.summary()

        class_weights = None
        if not self.multi_label:
            class_counts = y_train.value_counts()
            total = class_counts.sum()
            class_weights = {
                0: total / (2 * class_counts[0]),
                1: total / (2 * class_counts[1])
            }
            print(f"Class weights: {class_weights}")
        
        print("Training model...")
        history = self.model.fit(
            train_ds, 
            epochs=self.epochs,
            validation_data=val_ds,
            callbacks=self.get_callbacks(model_name),
            class_weight=class_weights,
        )

        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(f'results/training_history_{model_name}.csv', index=False)

        print("Evaluating model...")
        test_results = self.model.evaluate(test_ds, verbose=1)

        print("\nTest results:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"{metric_name}: {test_results[i]:.4f}")
        
        y_pred = self.model.predict(test_ds)

        self._save_prediction_results(X_test, y_test, y_pred, model_name)

        return self.model, history
    
    def train_with_kfold(self, k=5):
        if self.df is None:
            self.load_data()
        
        if self.multi_label:
            X = self.df['Image Index']
            y = self.df[self.unique_findings]
            num_classes = len(self.unique_findings)
            model_name_base = 'chest_xray_multilabel'
        else:
            X = self.df['Image Index']
            y = self.df['Is_Finding']
            num_classes = 1
            model_name_base = 'chest_xray_binary'
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n===== Training fold {fold+1}/{k} =====")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_ds = self.create_dataset(X_train.values, y_train.values.astype('float32'), is_training=True)
            val_ds = self.create_dataset(X_val.values, y_val.values.astype('float32'), is_training=False)

            model_name = f"{model_name_base}_fold_{fold+1}"
            self.model = self.create_model(num_classes=num_classes)

            class_weights = None
            if not self.multi_label:
                class_counts = y_train.value_counts()
                total = class_counts.sum()
                class_weights = {
                    0: total / (2 * class_counts[0]),
                    1: total / (2 * class_counts[1])
                }
                print(f"Class weights: {class_weights}")

            history = self.model.fit(
                train_ds,
                epochs=self.epochs,
                validation_data=val_ds,
                callbacks=self.get_callbacks(model_name),
                class_weight=class_weights
            )

            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(f'results/training_history_{model_name}.csv', index=False)

            y_pred = self.model.predict(val_ds)

            if self.multi_label:
                fold_metrics = self._calculate_multilabel_metrics(y_val, y_pred)
                fold_results.append({
                    'fold': fold+1,
                    'val_loss': history.history['val_loss'][-1],
                    'val_auc': history.history['val_auc'][-1],
                    'macro_f1': fold_metrics['macro_f1'],
                    'macro_auc': fold_metrics['macro_auc'],
                    'model_path': f'models/{model_name}.h5'
                })
            else:
                y_pred_classes = (y_pred > 0.5).astype(int).flatten()
                f1 = f1_score(y_val.values, y_pred_classes)
                auc = roc_auc_score(y_val.values, y_pred.flatten())
                
                print(f"Fold {fold+1} - F1: {f1:.4f}, AUC: {auc:.4f}")
                print(classification_report(y_val.values, y_pred_classes))
                
                fold_results.append({
                    'fold': fold+1,
                    'f1': f1,
                    'auc': auc,
                    'model_path': f'models/{model_name}.h5'
                })
            
            self._save_prediction_results(X_val, y_val, y_pred, model_name)
        
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f'results/kfold_results_{model_name_base}.csv', index=False)

        if self.multi_label:
            avg_macro_f1 = fold_df['macro_f1'].mean()
            avg_macro_auc = fold_df['macro_auc'].mean()
            
            print(f"\n===== Cross-validation results =====")
            print(f"Average Macro F1: {avg_macro_f1:.4f}")
            print(f"Average Macro AUC: {avg_macro_auc:.4f}")
        else:
            avg_f1 = fold_df['f1'].mean()
            avg_auc = fold_df['auc'].mean()
            
            print(f"\n===== Cross-validation results =====")
            print(f"Average F1: {avg_f1:.4f}")
            print(f"Average AUC: {avg_auc:.4f}")
        
        return fold_df
    
    def train_ensemble(self, architectures=None):
        if architectures is None:
            architectures = ['densenet', 'efficientnet', 'resnet']

        if self.df is None:
            self.load_and_preprocess_data()

        if self.multi_label:
            X = self.df['Image Index']
            y = self.df[self.unique_findings]
            num_classes = len(self.unique_findings)
            model_name_base = 'chest_xray_multilabel'
        else:
            X = self.df['Image Index']
            y = self.df['Is_Finding']
            num_classes = 1
            model_name_base = 'chest_xray_binary'
        
        print("Splitting data into train/validation/test sets...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if not self.multi_label else None)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val if not self.multi_label else None)
        
        train_ds = self.create_dataset(X_train.values, y_train.values.astype('float32'), is_training=True)
        val_ds = self.create_dataset(X_val.values, y_val.values.astype('float32'), is_training=False)
        test_ds = self.create_dataset(X_test.values, y_test.values.astype('float32'), is_training=False)

        class_weights = None
        if not self.multi_label:
            class_counts = y_train.value_counts()
            total = class_counts.sum()
            class_weights = {
                0: total / (2 * class_counts[0]),
                1: total / (2 * class_counts[1])
            }
            print(f"Class weights: {class_weights}")
        
        models = []
        for arch in architectures:
            print(f"\n===== Training {arch} model =====")
            model_name = f"{model_name_base}_{arch}"

            original_model_type = self.model_type
            self.model_type = arch
            model = self.create_model(num_classes=num_classes)
            self.model_type = original_model_type

            history = model.fit(
                train_ds,
                epochs=self.epochs,
                validation_data=val_ds,
                callbacks=self.get_callbacks(model_name),
                class_weight=class_weights
            )

            hist_df = pd.DataFrame(history.history)
            hist_df.to_csv(f'results/training_history_{model_name}.csv', index=False)
            
            models.append(model)
        
        print("Generating ensemble predictions...")
        ensemble_preds = self._ensemble_predict(models, test_ds)

        model_name = f"{model_name_base}_ensemble"
        self._save_prediction_results(X_test, y_test, ensemble_preds, model_name)
        
        return models
    
    def ensemble_predict(self, models, dataset):
        all_preds = []
        for model in models:
            preds = model.predict(dataset)
            all_preds.append(preds)
        
        ensemble_preds = np.mean(all_preds, axis=0)
        return ensemble_preds
    
    def _calculate_multilabel_metrics(self, y_true, y_pred):
        y_pred_classes = (y_pred > 0.5).astype(int)

        disease_metrics = []
        for i, finding in enumerate(self.unique_findings):
            try:
                disease_f1 = f1_score(y_true[finding].values, y_pred_classes[:, i])
                disease_auc = roc_auc_score(y_true[finding].values, y_pred[:, i])
                print(f"Finding: {finding} - F1: {disease_f1:.4f}, AUC: {disease_auc:.4f}")
                disease_metrics.append({
                    'finding': finding,
                    'f1': disease_f1,
                    'auc': disease_auc
                })
            except Exception as e:
                print(f"Error calculating metrics for {finding}: {e}")
                disease_metrics.append({
                    'finding': finding,
                    'f1': 0.0,
                    'auc': 0.5
                })
        
        macro_f1 = np.mean([m['f1'] for m in disease_metrics])
        macro_auc = np.mean([m['auc'] for m in disease_metrics])

        return {
            'disease_metrics': disease_metrics,
            'macro_f1': macro_f1,
            'macro_auc': macro_auc
        }
    
    def _save_prediction_results(self, X, y, y_pred, model_name):
        if self.multi_label:
            y_pred_classes = (y_pred > 0.5).astype(int)

            metrics = self._calculate_multilabel_metrics(y, y_pred)
            print(f"Macro F1: {metrics['macro_f1']:.4f}, Macro AUC: {metrics['macro_auc']:.4f}")

            results = pd.DataFrame({
                'Image_Index': X.values
            })

            for i, finding in enumerate(self.unique_findings):
                results[f'Actual_{finding}'] = y[finding].values
                results[f'Pred_Prob_{finding}'] = y_pred[:, i]
                results[f'Pred_Class_{finding}'] = y_pred_classes[:, i]

            results.to_csv(f'results/{model_name}_predictions.csv', index=False)

            metrics_df = pd.DataFrame(metrics['disease_metrics'])
            metrics_df.to_csv(f'results/{model_name}_metrics.csv', index=False)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            
            f1 = f1_score(y.values, y_pred_classes)
            auc = roc_auc_score(y.values, y_pred.flatten())

            print(f"F1: {f1:.4f}, AUC: {auc:.4f}")
            print(classification_report(y.values, y_pred_classes))

            results_df = pd.DataFrame({
                'Image_Index': X.values,
                'Actual': y.values,
                'Pred_Prob': y_pred.flatten(),
                'Pred_Class': y_pred_classes
            })
            results_df.to_csv(f'results/{model_name}_predictions.csv', index=False)

def main():
    data_path = "Data_Entry_2017_v2020.csv"
    images_folder = 'final_images/images'
    img_size = (224, 224)
    batch_size = 32
    epochs = 15
    model_type = 'densenet'
    multi_label = True
    use_kfold = False
    use_ensemble = False

    classifier = ChestXRayClassifier(
        data_path=data_path,
        images_folder=images_folder,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        model_type=model_type,
        multi_label=multi_label
    )

    classifier.load_data()

    if use_kfold:
        print("Starting K-Fold Cross-Validation...")
        fold_results = classifier.train_with_kfold(k=5)
        print("KFold completed.")
        return fold_results
    elif use_ensemble:
        print("Starting Ensemble Training...")
        models = classifier.train_ensemble(architectures=['densenet', 'efficientnet', 'resnet'])
        print("Ensemble training completed.")
        return models
    else:
        print("Starting Training and Evaluation...")
        model, history = classifier.train_eval()
        print("Training and evaluation completed.")
        return model, history
    


if __name__ == "__main__":
    main()
