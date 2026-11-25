#include "network.h"
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "operations.h"
#include "activation.h"

#ifdef _WIN32
    #include <direct.h>
    #define MAKE_DIR(dir) _mkdir(dir)
    #define CHANGE_DIR(dir) _chdir(dir)
#else
    #include <unistd.h>
    #define MAKE_DIR(dir) mkdir(dir, 0777)
    #define CHANGE_DIR(dir) chdir(dir)
#endif

#define MAXCHAR 1000

NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    net->input = input;
    net->hidden = hidden;
    net->output = output;
    net->learning_rate = lr;
    Matrix* hidden_layer = mtx_create(hidden, input);
    Matrix* output_layer = mtx_create(output, hidden);
    mtx_rnd(hidden_layer, hidden);
    mtx_rnd(output_layer, output);
    net->hidden_weights = hidden_layer;
    net->output_weights = output_layer;
    return net;
}

void network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
    // Feed forward
    Matrix* hidden_inputs  = dot(net->hidden_weights, input);
    Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
    Matrix* final_inputs   = dot(net->output_weights, hidden_outputs);
    Matrix* final_outputs  = apply(sigmoid, final_inputs);

    // Find errors
    Matrix* output_errors  = subtract(output, final_outputs);
    Matrix* transposed_mat = transpose(net->output_weights);
    Matrix* hidden_errors  = dot(transposed_mat, output_errors);
    mtx_free(transposed_mat);

    // Backpropagate output weights
    Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
    Matrix* multiplied_mat     = multiply(output_errors, sigmoid_primed_mat);
    transposed_mat             = transpose(hidden_outputs);
    Matrix* dot_mat            = dot(multiplied_mat, transposed_mat);
    Matrix* scaled_mat         = scale(net->learning_rate, dot_mat);
    Matrix* added_mat          = add(net->output_weights, scaled_mat);

    mtx_free(net->output_weights);
    net->output_weights = added_mat;

    mtx_free(sigmoid_primed_mat);
    mtx_free(multiplied_mat);
    mtx_free(transposed_mat);
    mtx_free(dot_mat);
    mtx_free(scaled_mat);

    // Backpropagate hidden weights
    sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
    multiplied_mat     = multiply(hidden_errors, sigmoid_primed_mat);
    transposed_mat     = transpose(input);
    dot_mat            = dot(multiplied_mat, transposed_mat);
    scaled_mat         = scale(net->learning_rate, dot_mat);
    added_mat          = add(net->hidden_weights, scaled_mat);

    mtx_free(net->hidden_weights);
    net->hidden_weights = added_mat;

    mtx_free(sigmoid_primed_mat);
    mtx_free(multiplied_mat);
    mtx_free(transposed_mat);
    mtx_free(dot_mat);
    mtx_free(scaled_mat);

    // Free forward/error matrices
    mtx_free(hidden_inputs);
    mtx_free(hidden_outputs);
    mtx_free(final_inputs);
    mtx_free(final_outputs);
    mtx_free(output_errors);
    mtx_free(hidden_errors);
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        if (i % 100 == 0) printf("Img No. %d\n", i);
        Img* cur_img = imgs[i];
        Matrix* img_data = mtx_flatten(cur_img->img_data, 0);
        Matrix* output = mtx_create(10, 1);
        mtx_fill(output, 0);
        output->data[cur_img->label][0] = 1;
        network_train(net, img_data, output);
        mtx_free(output);
        mtx_free(img_data);
    }
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
    Matrix* img_data = mtx_flatten(img->img_data, 0);
    Matrix* res = network_predict(net, img_data);
    mtx_free(img_data);
    return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
    int n_correct = 0;
    for (int i = 0; i < n; i++) {
        Matrix* prediction = network_predict_img(net, imgs[i]);
        if (mtx_argmax(prediction) == imgs[i]->label) {
            n_correct++;
        }
        mtx_free(prediction);
    }
    return 1.0 * n_correct / n;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
    Matrix* hidden_inputs  = dot(net->hidden_weights, input_data);
    Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
    Matrix* final_inputs   = dot(net->output_weights, hidden_outputs);
    Matrix* final_outputs  = apply(sigmoid, final_inputs);
    Matrix* result         = softmax(final_outputs);

    mtx_free(hidden_inputs);
    mtx_free(hidden_outputs);
    mtx_free(final_inputs);
    mtx_free(final_outputs);

    return result;
}

void network_save(NeuralNetwork* net, char* file_string) {
    FILE* fp = fopen(file_string, "wb");
    if (fp == NULL) {
        printf("Error: Could not open %s for writing\n", file_string);
        exit(EXIT_FAILURE);
    }

    // 1. Magic bytes + version identifier
    const char magic[4] = {'N', 'N', '0', '1'};
    fwrite(magic, sizeof(char), 4, fp);

    // 2. Architecture metadata
    int dims[3] = {net->input, net->hidden, net->output};
    fwrite(dims, sizeof(int), 3, fp);

    // 3. Raw weight matrices (row by row since data is double**)
    for (int i = 0; i < net->hidden_weights->rows; i++) {
        fwrite(net->hidden_weights->data[i], sizeof(double), net->hidden_weights->cols, fp);
    }
    for (int i = 0; i < net->output_weights->rows; i++) {
        fwrite(net->output_weights->data[i], sizeof(double), net->output_weights->cols, fp);
    }

    fclose(fp);
    printf("Saved binary model to '%s'\n", file_string);
}

NeuralNetwork* network_load(char* file_string) {
    FILE* fp = fopen(file_string, "rb");
    if (fp == NULL) {
        printf("Error: Could not open %s for reading\n", file_string);
        exit(EXIT_FAILURE);
    }

    char magic[4];
    if (fread(magic, sizeof(char), 4, fp) != 4 || memcmp(magic, "NN01", 4) != 0) {
        printf("Error: Invalid model file format in %s\n", file_string);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    int dims[3];
    fread(dims, sizeof(int), 3, fp);

    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    net->input = dims[0];
    net->hidden = dims[1];
    net->output = dims[2];
    net->learning_rate = 0.0;
    net->hidden_weights = mtx_create(net->hidden, net->input);
    net->output_weights = mtx_create(net->output, net->hidden);

    for (int i = 0; i < net->hidden_weights->rows; i++) {
        fread(net->hidden_weights->data[i], sizeof(double), net->hidden_weights->cols, fp);
    }
    for (int i = 0; i < net->output_weights->rows; i++) {
        fread(net->output_weights->data[i], sizeof(double), net->output_weights->cols, fp);
    }

    fclose(fp);
    printf("Loaded binary model from '%s'\n", file_string);
    return net;
}

void network_print(NeuralNetwork* net) {
    printf("# of Inputs: %d\n", net->input);
    printf("# of Hidden: %d\n", net->hidden);
    printf("# of Output: %d\n", net->output);
    printf("Hidden Weights: \n");
    mtx_print(net->hidden_weights);
    printf("Output Weights: \n");
    mtx_print(net->output_weights);
}

void network_free(NeuralNetwork *net) {
    if (net == NULL) return;
    mtx_free(net->hidden_weights);
    mtx_free(net->output_weights);
    free(net);
}

// Exports all hidden neurons (assuming 784 inputs = 28x28) into one montage PGM image
void network_export_features_pgm(NeuralNetwork* net, const char* filename) {
    if (net->input != 784) return;
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    int grid_cols = 20;
    int grid_rows = (net->hidden + grid_cols - 1) / grid_cols;
    int tile = 29; // 28x28 pixels + 1px black border
    int width = grid_cols * tile;
    int height = grid_rows * tile;

    // Precompute per-neuron min/max so every filter has full contrast
    double* min_w = malloc(net->hidden * sizeof(double));
    double* max_w = malloc(net->hidden * sizeof(double));
    for (int n = 0; n < net->hidden; n++) {
        min_w[n] = max_w[n] = net->hidden_weights->data[n][0];
        for (int k = 1; k < 784; k++) {
            double v = net->hidden_weights->data[n][k];
            if (v < min_w[n]) min_w[n] = v;
            if (v > max_w[n]) max_w[n] = v;
        }
    }

    fprintf(fp, "P2\n%d %d\n255\n", width, height);
    for (int py = 0; py < height; py++) {
        int tr = py / tile, lr = py % tile;
        for (int px = 0; px < width; px++) {
            int tc = px / tile, lc = px % tile;
            int idx = tr * grid_cols + tc;
            if (lr == 28 || lc == 28 || idx >= net->hidden) {
                fprintf(fp, "0 ");
            } else {
                double range = max_w[idx] - min_w[idx];
                double val = net->hidden_weights->data[idx][lr * 28 + lc];
                int pixel = (range > 1e-9) ? (int)(((val - min_w[idx]) / range) * 255.0) : 0;
                fprintf(fp, "%d ", pixel);
            }
        }
        fprintf(fp, "\n");
    }

    free(min_w);
    free(max_w);
    fclose(fp);
    printf("Exported neuron montage to '%s'\n", filename);
}

void network_export_header(NeuralNetwork* net, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "#pragma once\n\n");
    fprintf(fp, "#define NN_INPUT %d\n#define NN_HIDDEN %d\n#define NN_OUTPUT %d\n\n",
            net->input, net->hidden, net->output);

    fprintf(fp, "static const double HIDDEN_WEIGHTS[%d][%d] = {\n", net->hidden, net->input);
    for (int i = 0; i < net->hidden; i++) {
        fprintf(fp, "    {");
        for (int j = 0; j < net->input; j++) {
            fprintf(fp, "%.12g%s", net->hidden_weights->data[i][j], j < net->input - 1 ? ", " : "");
        }
        fprintf(fp, "}%s\n", i < net->hidden - 1 ? "," : "");
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const double OUTPUT_WEIGHTS[%d][%d] = {\n", net->output, net->hidden);
    for (int i = 0; i < net->output; i++) {
        fprintf(fp, "    {");
        for (int j = 0; j < net->hidden; j++) {
            fprintf(fp, "%.12g%s", net->output_weights->data[i][j], j < net->hidden - 1 ? ", " : "");
        }
        fprintf(fp, "}%s\n", i < net->output - 1 ? "," : "");
    }
    fprintf(fp, "};\n");
    fclose(fp);
}

// Logs every wrong test prediction to CSV for inspection
void network_export_errors_csv(NeuralNetwork* net, Img** imgs, int n, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "image_index,true_label,predicted_label,confidence\n");
    for (int i = 0; i < n; i++) {
        Matrix* pred = network_predict_img(net, imgs[i]);
        int guess = mtx_argmax(pred);
        if (guess != imgs[i]->label) {
            fprintf(fp, "%d,%d,%d,%.4f\n", i, imgs[i]->label, guess, pred->data[guess][0]);
        }
        mtx_free(pred);
    }
    fclose(fp);
    printf("Exported misclassifications to '%s'\n", filename);
}