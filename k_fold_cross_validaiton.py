#594
import os
import gc
import shutil
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import time
# from tensorflow.keras import backend as K
# from tensorflow.keras.losses import Loss

#----------------------------------------------------------------------------------------#
# Set environment variable to use the first GPU (NVIDIA GeForce RTX 3050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Get the list of physical GPUs
physical_devices = tf.config.list_physical_devices('GPU')

# Check if the list of GPUs is non-empty
if physical_devices:
    try:
        # Limit GPU memory usage to 4000 MB for the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]
        )
        
        print("Physical GPUs available:", len(physical_devices))
    except RuntimeError as e:
        print("Error setting GPU configuration:", e)
else:
    print("No physical GPUs available, using CPU instead.")
#----------------------------------------------------------------------------------------#


def run_experiment_with_k_fold_advanced(x_data, y_data, k=5):
    skf = KFold(n_splits=k, shuffle=True)
    fold_results = []
    # histories = []
    test_accuracies = []
    
    # Define paths
    delete_path = os.path.join(first_path, database, image_type, 'tmp')
    if os.path.exists(delete_path):
        shutil.rmtree(delete_path)
        print("tmp folder has been deleted.")
    else:
        print("tmp folder does not exist.")
        os.makedirs(delete_path, exist_ok=True)
    
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.02,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.6, 0.9],
        fill_mode='reflect'  # 'reflect', 'nearest'
    )    

    # Get the folds
    folds = list(skf.split(x_data, y_data))
    np.random.shuffle(folds)

    for iteration in range(k):
        print(f"Running iteration {iteration + 1}")

        # Create a new model instance for each fold
        model = create_advanced_vit_classifier()

        # Compile the model with SWA
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # Wrap it with StochasticWeightAveraging
        optimizer = StochasticWeightAveraging(optimizer=base_optimizer, start_averaging=3, average_period=6)
        model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: sparse_categorical_crossentropy_with_label_smoothing(y_true, y_pred, label_smoothing=0.1),
            metrics=['accuracy']
        )

        test_fold = folds[iteration]
        val_fold = folds[(iteration + 1) % k]
        train_folds = [folds[i] for i in range(k) if i != iteration and i != (iteration + 1) % k]

        x_test_fold, y_test_fold = x_data[test_fold[1]], y_data[test_fold[1]]
        x_val_fold, y_val_fold = x_data[val_fold[1]], y_data[val_fold[1]]
        train_indices = np.concatenate([train_fold[1] for train_fold in train_folds])
        x_train_fold, y_train_fold = x_data[train_indices], y_data[train_indices]
    
        # Convert one-hot encoded labels to integer class labels
        y_test_fold = np.argmax(y_test_fold, axis=1)
        y_train_fold = np.argmax(y_train_fold, axis=1)
        y_val_fold = np.argmax(y_val_fold, axis=1)
    
        # Display model summary only for the first iteration
        if iteration == 0:
            model.summary()

        # Print class distribution for each iteration
        print(f"Class distribution in test data in iteration {iteration + 1}:")
        print(np.bincount(y_test_fold))        
        print(f"Class distribution in training data in iteration {iteration + 1}:")
        print(np.bincount(y_train_fold))        
        print(f"Class distribution in validation data in iteration {iteration + 1}:")
        print(np.bincount(y_val_fold))

        path = os.path.join(first_path, database, image_type + preprocessed_path, 'Outputs', transformer, version, output_type, f"iteration_{iteration + 1}")
        os.makedirs(path, exist_ok=True)

        # Visualize random train image and patches
        plt.figure(figsize=(12, 12))
        image = x_train_fold[np.random.choice(range(x_train_fold.shape[0]))]
        plt.set_cmap('Blues')
        plt.imshow(image)
        plt.axis("off")
        plt.savefig((os.path.join(path, 'Random_Train_Image.png')), dpi=1600)
        plt.close()

        # Convert PIL Image to NumPy array
        image_array = np.array(image)
        # Resize the NumPy array using TensorFlow
        resized_image = tf.image.resize(tf.convert_to_tensor([image_array]), size=(image_size, image_size))
        # Create patch image
        patches = Patches(patch_size)(resized_image)
        
        print(f"Image size: {image_size} X {image_size}")
        print(f"Patch size: {patch_size} X {patch_size}")
        print(f"Patches per image: {patches.shape[1]}")
        print(f"Elements per patch: {patches.shape[-1]}")

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(12, 12))
        for i, patch in enumerate(patches[0]):
            plt.subplot(n, n, i + 1)
            # Assuming 'patch' contains the pixel values of the image patch
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(patch_img.numpy() / 255.0)
            plt.axis("off")
        # Save the entire figure after all patches are plotted    
        plt.savefig((os.path.join(path, 'Patch_Image.png')), dpi=1600)
        # plt.show()
        plt.close()

        # Checkpoint and callback settings
        path2 = os.path.join(first_path, database, image_type + preprocessed_path, 'tmp', transformer, version, output_type)
        os.makedirs(path2, exist_ok=True)
        
        checkpoint_dir = os.path.join(path2, f'checkpoint_iteration_{iteration + 1}')
        delete_path = checkpoint_dir
        if os.path.exists(delete_path):
            shutil.rmtree(delete_path)
            print("tmp/checkpoint_dir/ folder has been deleted.")
        else:
            print("tmp/checkpoint_dir/ folder does not exist.")
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filepath = os.path.join(checkpoint_dir, 'model_checkpoint')
        checkpoint_model_filepath = os.path.join(checkpoint_dir, 'model.hdf5')
        # Introduce a delay between saving and loading model weights
        time.sleep(2)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
        
        # One Cycle LR Callback
        lr_scheduler = one_cycle_lr(
            total_epochs=num_epochs,
            lr_max=0.001,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # ReduceLROnPlateau Callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            mode='min',
            min_lr=1e-6
        )
        
        activation_updater = ActivationUpdater(model, threshold=0.85)
        
        callbacks_list = [checkpoint_callback, early_stop, lr_scheduler, reduce_lr, activation_updater]
        
        start = datetime.now()

#----------------------augmentation operation of testing images---------------------------#

        num_augmented_images_per_class = 400
        batch_size = 6
        
        # Check if y_test_fold is one-hot encoded or not
        if y_test_fold.ndim == 2:
            class_labels = np.argmax(y_test_fold, axis=1)  # Convert from one-hot to class labels
        else:
            class_labels = y_test_fold  # Already class labels
        
        unique_classes = np.unique(class_labels)
        
        x_test_augmented = []
        y_test_augmented = []
        
        for class_label in unique_classes:
            # Filter testing data for the current class
            class_indices = np.where(class_labels == class_label)[0]
            x_class_fold = x_test_fold[class_indices]
            y_class_fold = y_test_fold[class_indices]
        
            num_current_images = len(x_class_fold)
            num_images_to_augment = max(0, num_augmented_images_per_class - num_current_images)
            
            if num_images_to_augment > 0:
                # Setup augmentation generator
                augmented_data = datagen.flow(x_class_fold, y_class_fold, batch_size=batch_size)
                
                augmented_images = []
                augmented_labels = []
        
                # Augment images until we reach the target count
                while len(augmented_images) * batch_size < num_images_to_augment:
                    imgs, labels = augmented_data.next()
                    augmented_images.append(imgs)
                    augmented_labels.append(labels)
                    
                    # Stop if we have generated enough images
                    if len(augmented_images) * batch_size >= num_images_to_augment:
                        break
        
                augmented_images = np.concatenate(augmented_images, axis=0)[:num_images_to_augment]
                augmented_labels = np.concatenate(augmented_labels, axis=0)[:num_images_to_augment]
                
                x_test_augmented.append(augmented_images)
                y_test_augmented.append(augmented_labels)
        
        # Concatenate all augmented images and labels
        x_test_augmented = np.concatenate(x_test_augmented, axis=0)
        y_test_augmented = np.concatenate(y_test_augmented, axis=0)
        
        # Combine original and augmented data
        x_test_combined = np.concatenate([x_test_fold, x_test_augmented], axis=0)
        y_test_combined = np.concatenate([y_test_fold, y_test_augmented], axis=0)
        
        # Shuffle the combined dataset
        shuffled_indices = np.random.permutation(len(y_test_combined))
        x_test_combined = x_test_combined[shuffled_indices]
        y_test_combined = y_test_combined[shuffled_indices]
        
        # Print class distribution
        if y_test_combined.ndim == 2:
            y_test_combined_labels = np.argmax(y_test_combined, axis=1)
        else:
            y_test_combined_labels = y_test_combined
        
        print(f"Class distribution in augmented testing data in iteration {iteration + 1}:")
        print(np.bincount(y_test_combined_labels))


#----------------------augmentation operation of training images---------------------------#

        num_augmented_images_per_class = 2000
        batch_size = 6
        
        # Check if y_train_fold is one-hot encoded or not
        if y_train_fold.ndim == 2:
            class_labels = np.argmax(y_train_fold, axis=1)  # Convert from one-hot to class labels
        else:
            class_labels = y_train_fold  # Already class labels
        
        unique_classes = np.unique(class_labels)
        
        x_train_augmented = []
        y_train_augmented = []
        
        for class_label in unique_classes:
            # Filter training data for the current class
            class_indices = np.where(class_labels == class_label)[0]
            x_class_fold = x_train_fold[class_indices]
            y_class_fold = y_train_fold[class_indices]
        
            num_current_images = len(x_class_fold)
            num_images_to_augment = max(0, num_augmented_images_per_class - num_current_images)
            
            if num_images_to_augment > 0:
                # Setup augmentation generator
                augmented_data = datagen.flow(x_class_fold, y_class_fold, batch_size=batch_size)
                
                augmented_images = []
                augmented_labels = []
        
                # Augment images until we reach the target count
                while len(augmented_images) * batch_size < num_images_to_augment:
                    imgs, labels = augmented_data.next()
                    augmented_images.append(imgs)
                    augmented_labels.append(labels)
                    
                    # Stop if we have generated enough images
                    if len(augmented_images) * batch_size >= num_images_to_augment:
                        break
        
                augmented_images = np.concatenate(augmented_images, axis=0)[:num_images_to_augment]
                augmented_labels = np.concatenate(augmented_labels, axis=0)[:num_images_to_augment]
                
                x_train_augmented.append(augmented_images)
                y_train_augmented.append(augmented_labels)
        
        # Concatenate all augmented images and labels
        x_train_augmented = np.concatenate(x_train_augmented, axis=0)
        y_train_augmented = np.concatenate(y_train_augmented, axis=0)
        
        # Combine original and augmented data
        x_train_combined = np.concatenate([x_train_fold, x_train_augmented], axis=0)
        y_train_combined = np.concatenate([y_train_fold, y_train_augmented], axis=0)
        
        # Shuffle the combined dataset
        shuffled_indices = np.random.permutation(len(y_train_combined))
        x_train_combined = x_train_combined[shuffled_indices]
        y_train_combined = y_train_combined[shuffled_indices]
        
        # Print class distribution
        if y_train_combined.ndim == 2:
            y_train_combined_labels = np.argmax(y_train_combined, axis=1)
        else:
            y_train_combined_labels = y_train_combined
        
        print(f"Class distribution in augmented training data in iteration {iteration + 1}:")
        print(np.bincount(y_train_combined_labels))

#-----------------------------------------------------------------------------------#

        # Train the model
        history = model.fit(
            x_train_combined, y_train_combined,
            epochs=num_epochs,
            validation_data=(x_val_fold, y_val_fold),
            callbacks=callbacks_list,
            batch_size=batch_size
        )
        
        stop = datetime.now()
        training_time = stop - start
        print("Training execution time is: ", training_time)

        # histories.append(history)

        # Evaluate on validation set
        _, accuracy = model.evaluate(x_val_fold, y_val_fold)
        print(f"Validation accuracy for iteration {iteration + 1}: {round(accuracy * 100, 2)}%")

        # Evaluate on test set
        _, test_accuracy = model.evaluate(x_test_fold, y_test_fold, verbose=1)
        print(f"Test accuracy for iteration {iteration + 1}: {test_accuracy}")
        test_accuracies.append(test_accuracy)

        fold_results.append({
            'history': history.history,
            'checkpoint_filepath': checkpoint_filepath,
            'model_filepath': checkpoint_model_filepath,
            'epochs': len(history.history['loss'])
        })

        # Confusion matrix and metrics
        y_pred = model.predict(x_test_fold)
        y_pred_labels = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_test_fold, y_pred_labels)
        tp, fp, fn, tn = cm.ravel()
        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, fmt="", cmap='YlGnBu', annot_kws={"size": 36}, cbar=False)
        # Annotate TP, TN, FP, FN values with labels
        plt.text(0.5, 0.5, f'TN: {tn}', horizontalalignment='center', verticalalignment='center', fontsize=36, color='black')
        plt.text(0.5, 1.5, f'FP: {fp}', horizontalalignment='center', verticalalignment='center', fontsize=36, color='black')
        plt.text(1.5, 0.5, f'FN: {fn}', horizontalalignment='center', verticalalignment='center', fontsize=36, color='black')
        plt.text(1.5, 1.5, f'TP: {tp}', horizontalalignment='center', verticalalignment='center', fontsize=36, color='white')
        plt.xlabel('Predicted label', fontsize=36)
        plt.ylabel('True label', fontsize=36)
        plt.title('Confusion Matrix', fontsize=36)
        plt.xticks(fontsize=36)
        plt.yticks(fontsize=36)
        plt.set_cmap('YlGnBu')
        plt.savefig(os.path.join(path, f'iteration_{iteration + 1}_Confusion Matrix.png'), dpi=1600)
        plt.show()
    
        prec = precision_score(y_test_fold, y_pred_labels) * 100
        f1 = f1_score(y_test_fold, y_pred_labels)
        sen = tp / (tp + fn) * 100
        spec = tn / (tn + fp) * 100
        acc = ((tp + tn) / (tp + tn + fp + fn)) * 100

        print(f"TP: {tp}")
        print(f"TN: {tn}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")       
        print('Precision:', prec)
        print('Sensitivity:', sen)
        print('Specificity:', spec)
        print('Test Accuracy:', acc)
        print('F1 score:', f1)

        num_classes=2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_fold == i, y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        plt.figure()
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(path, f'iteration_{iteration + 1}_ROC Curve.png'), dpi=1600)
        plt.show()
       
        # Plot training and validation loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(f'Training and validation loss - iteration {iteration + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.grid(True)
        plt.savefig(os.path.join(path, f'iteration_{iteration + 1}_Training_and_validation_loss.png'), dpi=1600)
        plt.tight_layout()
        # plt.show()
        plt.close()

        # Plot training and validation accuracy
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title(f'Training and validation accuracy - iteration {iteration + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.grid(True)
        plt.savefig(os.path.join(path, f'iteration_{iteration + 1}_Training_and_validation_accuracy.png'), dpi=1600)
        plt.tight_layout()
        # plt.show()
        plt.close()
        
        # Reset the session and clear memory
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"Iteration {iteration + 1} completed.\n")

    # Final results
    avg_test_accuracy = np.mean(test_accuracies)
    print(f"Average test accuracy across all folds: {avg_test_accuracy * 100:.2f}%")
    
    # Save the results and models
    results_path = os.path.join(first_path, database, image_type + preprocessed_path, 'Outputs', transformer, version, output_type, "results.npz")
    np.savez(results_path, fold_results=fold_results, test_accuracies=test_accuracies)
    # print(f"Results saved to {results_path}")

    return fold_results, avg_test_accuracy





# """




