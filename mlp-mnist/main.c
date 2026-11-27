#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "img.h"
#include "activation.h"
#include "network.h"
#include "matrix.h"
#include "operations.h"
#include "export.h"

void shuffle_imgs(Img** imgs, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Img* temp = imgs[i];
        imgs[i] = imgs[j];
        imgs[j] = temp;
    }
}

int main() {
    srand(time(NULL));

    //train
    int train_imgs_count = 10000;
    Img** train_imgs = csv_to_imgs("mnist_train.csv", train_imgs_count);
    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    for (int epoch = 0; epoch < 15; epoch++) {
        printf("--- Epoch %d (lr: %.5f) ---\n", epoch + 1, net->learning_rate);
        shuffle_imgs(train_imgs, train_imgs_count);
        network_train_batch_imgs(net, train_imgs, train_imgs_count);
        net->learning_rate *= 0.92;
    }

    //export
    network_save(net, "mnist_model.nn");
    network_export_header(net, "trained_weights.h");
    network_export_features_bmp(net, "features.bmp");

    imgs_free(train_imgs, train_imgs_count);
    network_free(net);

    //prediction
    int test_imgs_count = 3000;
    Img** test_imgs = csv_to_imgs("mnist_test.csv", test_imgs_count);
    NeuralNetwork* loaded_net = network_load("mnist_model.nn");

    double score = network_predict_imgs(loaded_net, test_imgs, test_imgs_count);
    printf("\nTest Accuracy Score: %1.5f\n", score);

    network_preview_predictions(loaded_net, test_imgs, 5);
    network_export_errors_csv(loaded_net, test_imgs, test_imgs_count, "errors.csv");

    imgs_free(test_imgs, test_imgs_count);
    network_free(loaded_net);
    return 0;
}
