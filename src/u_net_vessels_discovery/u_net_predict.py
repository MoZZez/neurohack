# coding=utf-8
###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

import ConfigParser

import argh
import time
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from keras.models import model_from_json
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from help_functions import *
from extract_patches import recompone
from pre_processing import my_PreProc
import cv2
import os
import numpy as np
from extract_patches import extract_ordered
import logging

LOG = logging.getLogger('u_net_predict')
LOG.setLevel('INFO')

# Load the original data and return the extracted patches for training/testing
def split_to_patches(samples,
                     labels,
                     patch_height=48,
                     patch_width=48):
    samples = my_PreProc(samples)
    labels /= 255.0
    patches_imgs_test = extract_ordered(samples, patch_height, patch_width)
    patches_masks_test = extract_ordered(labels, patch_height, patch_width)
    return patches_imgs_test, patches_masks_test


def load_image_set(dataset_dir, to_grayscale=False):
    files = os.listdir(dataset_dir)
    files = map(lambda x: os.path.join(dataset_dir, x), files)
    images = np.asarray(map(cv2.imread, files))
    images = np.transpose(images, (0, 3, 1, 2))
    if to_grayscale:
        images = images.mean(axis=1).reshape((images.shape[0], 1, images.shape[2], images.shape[3]))
    return images


def load_model(model_structure, model_weights):
    with open(model_structure) as f:
        model = model_from_json(f.read())

    model.load_weights(model_weights)
    return model


def predict(config):
    images  = load_image_set(config.get('data paths', 'images'))
    targets = load_image_set(config.get('data paths', 'targets'), to_grayscale=True)

    patch_height = int(config.get('data attributes', 'patch_height'))
    patch_width = int(config.get('data attributes', 'patch_width'))

    images_patches, targets_patches = split_to_patches(images, targets, patch_height, patch_width)
    model = load_model(config.get("testing settings", "model_architecture"),
                       config.get("testing settings", "model_weights"))

    predictions_patches = model.predict(images_patches, batch_size=32, verbose=2)
    print "predicted images size :"
    print predictions_patches.shape

    predictions_patches = pred_to_imgs(predictions_patches, "original")

    full_images_height = images.shape[2]
    full_images_width = images.shape[3]
    patches_height_count = int(full_images_height / patch_height)
    patches_width_count = int(full_images_width / patch_width)

    predictions_restored = recompone(predictions_patches, patches_height_count, patches_width_count)
    images_restored = recompone(images_patches, patches_height_count, patches_width_count)
    targets_restored = recompone(targets_patches, patches_height_count, patches_width_count)

    images_restored = images_restored[:, :, 0:full_images_height, 0:full_images_width]
    predictions_restored = predictions_restored[:, :, 0:full_images_height, 0:full_images_width]
    targets_restored = targets_restored[:, :, 0:full_images_height, 0:full_images_width]
    return images_restored, predictions_restored, targets_restored


def evaluate(images, predictions, targets, config):
    print "\n\n========  Evaluate the results ======================="
    N_visual = int(config.get('testing settings', 'N_group_visual'))
    name_experiment = config.get('experiment name', 'name')
    path_experiment = './' + name_experiment + '/'
    if not os.path.isdir(path_experiment):
        os.mkdir(path_experiment)

    print "Orig imgs shape: " + str(images.shape)
    print "pred imgs shape: " + str(predictions.shape)
    print "Gtruth imgs shape: " + str(targets.shape)
    visualize(group_images(images, N_visual), path_experiment + "all_originals")
    visualize(group_images(predictions, N_visual), path_experiment + "all_predictions")
    visualize(group_images(targets, N_visual), path_experiment + "all_groundTruths")

    # visualize results comparing mask and prediction:
    assert (images.shape[0] == predictions.shape[0] and images.shape[0] == targets.shape[0])
    N_predicted = images.shape[0]
    group = N_visual
    assert (N_predicted % group == 0)
    for i in range(int(N_predicted / group)):
        images_stripe = group_images(images[i * group:(i * group) + group, :, :, :], group)
        targets_stripe = group_images(targets[i * group:(i * group) + group, :, :, :], group)
        predictions_stripe = group_images(predictions[i * group:(i * group) + group, :, :, :], group)
        total = np.concatenate((images_stripe, targets_stripe, predictions_stripe), axis=0)
        visualize(total, path_experiment + name_experiment + "_Original_GroundTruth_Prediction" + str(i))

    y_scores = predictions.reshape(np.prod(predictions.shape))
    y_true = targets.reshape(np.prod(targets.shape))

    print "Calculating results only inside the FOV:"
    print "y scores pixels: " + str(
        y_scores.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        predictions.shape[0] * predictions.shape[2] * predictions.shape[3]) + " (584*565==329960)"
    print "y true pixels: " + str(
        y_true.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        targets.shape[2] * targets.shape[3] * targets.shape[0]) + " (584*565==329960)"
    print "y_scores:", y_true[:100]
    y_true[y_true < 0.5] = 0
    y_true[y_true >= 0.5] = 1
    print "y_scores:", y_true[:100]

    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print "\nArea under the ROC curve: " + str(AUC_ROC)
    plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "ROC.png")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
    plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "Precision_recall.png")

    # Confusion matrix
    threshold_confusion = 0.5
    print "\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion)
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print confusion
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print "Global Accuracy: " + str(accuracy)
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print "Specificity: " + str(specificity)
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print "Sensitivity: " + str(sensitivity)
    precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print "Precision: " + str(precision)

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print "\nJaccard similarity score: " + str(jaccard_index)

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print "\nF1 score (F-measure): " + str(F1_score)

    # Save the results
    file_perf = open(path_experiment + 'performances.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\nF1 score (F-measure): " + str(F1_score)
                    + "\n\nConfusion matrix:"
                    + str(confusion)
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    )
    file_perf.close()


@argh.arg('config', type=load_config, help='config file')
def predict_eval(config):
    images_restored, predictions_restored, targets_restored = predict(config)
    evaluate(images_restored, predictions_restored, targets_restored, config)


@argh.arg('config', type=load_config, help='config file')
def retrain_nn(config):
    LOG.info("retrain neural net with config: {}".format(config))
    images  = load_image_set(config.get('data paths', 'images'))
    targets = load_image_set(config.get('data paths', 'targets'), to_grayscale=True)

    LOG.info("images shape: {}".format(images.shape))
    LOG.info("targets shape: {}".format(targets.shape))

    images_train = images[:-1,:,:,:]
    targets_train = targets[:-1,:,:,:]
    images_test = images[-1:,:,:,:]
    targets_test = targets[-1:,:,:,:]
    images_train_patches, targets_targets_patches = split_to_patches(images_train,
                                                                     targets_train,
                                                                     int(config.get('data attributes', 'patch_height')),
                                                                     int(config.get('data attributes', 'patch_width')))
    images_test_patches, targets_test_patches = split_to_patches(images_test,
                                                                 targets_test,
                                                                 int(config.get('data attributes', 'patch_height')),
                                                                 int(config.get('data attributes', 'patch_width')))

    LOG.info("train images patches shape: {}".format(images_train_patches.shape))
    LOG.info("train targets patches shape: {}".format(targets_targets_patches.shape))
    LOG.info("test images patches shape: {}".format(images_test_patches.shape))
    LOG.info("test targets patches shape: {}".format(targets_test_patches.shape))

    LOG.info("Compiling model... ")
    start = time.time()
    model = load_model(config.get("retrain", "model_architecture"),
                       config.get("retrain", "input_weights"))
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    LOG.info("... done in {}s".format(time.time() - start))

    LOG.info("Training models")
    start = time.time()
    model_checkpoint = ModelCheckpoint(filepath=config.get("retrain", "best_weights"),
                                       verbose=1,
                                       monitor='val_loss',
                                       mode='auto',
                                       save_best_only=True)
    model.fit(images_train_patches,
              masks_Unet(targets_targets_patches),
              nb_epoch=int(config.get("retrain", "N_epochs")),
              batch_size=int(config.get("retrain", "batch_size")),
              verbose=2,
              shuffle=True,
              validation_data=(images_test_patches, masks_Unet(targets_test_patches)),
              callbacks=[model_checkpoint])
    LOG.info("... done in {}s".format(time.time() - start))

    LOG.info("Saving model...")
    start = time.time()
    model.save_weights(config.get("retrain", "last_weights"), overwrite=True)
    LOG.info("... done in {}s".format(time.time() - start))


def main():
    commands = [
        retrain_nn,
        predict_eval
    ]
    argh.dispatch_commands(commands)

if __name__ == "__main__":
    main()