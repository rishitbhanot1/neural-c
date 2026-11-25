#include "export.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Helper: computes min and max weight per hidden neuron for contrast scaling
static void compute_neuron_ranges(NeuralNetwork* net, double* min_w, double* max_w) {
    for (int n = 0; n < net->hidden; n++) {
        min_w[n] = max_w[n] = net->hidden_weights->data[n][0];
        for (int k = 1; k < net->input; k++) {
            double v = net->hidden_weights->data[n][k];
            if (v < min_w[n]) min_w[n] = v;
            if (v > max_w[n]) max_w[n] = v;
        }
    }
}

// Exports a 24-bit Windows BMP montage of all 784-input hidden neurons
void network_export_features_bmp(NeuralNetwork* net, const char* filename) {
    if (net->input != 784) return;
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    int grid_cols = 20;
    int grid_rows = (net->hidden + grid_cols - 1) / grid_cols;
    int tile = 29; // 28x28 neuron tile + 1px border
    int width = grid_cols * tile;
    int height = grid_rows * tile;
    int row_stride = (width * 3 + 3) & ~3; // 4-byte row alignment
    uint32_t image_size = (uint32_t)(row_stride * height);
    uint32_t file_size = 54 + image_size;

    // 14-byte BMP file header (byte-packed to avoid strict-aliasing warnings)
    uint8_t file_header[14] = {
        'B', 'M',
        (uint8_t)(file_size), (uint8_t)(file_size >> 8),
        (uint8_t)(file_size >> 16), (uint8_t)(file_size >> 24),
        0, 0, 0, 0,
        54, 0, 0, 0
    };
    fwrite(file_header, 1, 14, fp);

    // 40-byte BITMAPINFOHEADER
    uint32_t dib_size = 40;
    int32_t w = width, h = height;
    uint16_t planes = 1, bpp = 24;
    uint32_t zero = 0;
    int32_t ppm = 2835;
    fwrite(&dib_size, 4, 1, fp);
    fwrite(&w, 4, 1, fp);
    fwrite(&h, 4, 1, fp);
    fwrite(&planes, 2, 1, fp);
    fwrite(&bpp, 2, 1, fp);
    fwrite(&zero, 4, 1, fp);
    fwrite(&image_size, 4, 1, fp);
    fwrite(&ppm, 4, 1, fp);
    fwrite(&ppm, 4, 1, fp);
    fwrite(&zero, 4, 1, fp);
    fwrite(&zero, 4, 1, fp);

    double* min_w = malloc(net->hidden * sizeof(double));
    double* max_w = malloc(net->hidden * sizeof(double));
    compute_neuron_ranges(net, min_w, max_w);

    // BMP rows are stored bottom-to-top in BGR order
    uint8_t* row_buf = calloc(row_stride, 1);
    for (int py = height - 1; py >= 0; py--) {
        int tr = py / tile, lr = py % tile;
        for (int px = 0; px < width; px++) {
            int tc = px / tile, lc = px % tile;
            int idx = tr * grid_cols + tc;
            if (lr == 28 || lc == 28 || idx >= net->hidden) {
                row_buf[px * 3 + 0] = 60; // B
                row_buf[px * 3 + 1] = 40; // G
                row_buf[px * 3 + 2] = 30; // R
            } else {
                double range = max_w[idx] - min_w[idx];
                double val = net->hidden_weights->data[idx][lr * 28 + lc];
                uint8_t pix = (range > 1e-9) ? (uint8_t)(((val - min_w[idx]) / range) * 255.0) : 0;
                row_buf[px * 3 + 0] = pix;
                row_buf[px * 3 + 1] = pix;
                row_buf[px * 3 + 2] = pix;
            }
        }
        fwrite(row_buf, 1, row_stride, fp);
    }

    free(row_buf);
    free(min_w);
    free(max_w);
    fclose(fp);
    printf("Exported BMP neuron montage to '%s'\n", filename);
}

// Exports an ASCII Portable GrayMap (P2) montage of all 784-input hidden neurons
void network_export_features_pgm(NeuralNetwork* net, const char* filename) {
    if (net->input != 784) return;
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    int grid_cols = 20;
    int grid_rows = (net->hidden + grid_cols - 1) / grid_cols;
    int tile = 29;
    int width = grid_cols * tile;
    int height = grid_rows * tile;

    double* min_w = malloc(net->hidden * sizeof(double));
    double* max_w = malloc(net->hidden * sizeof(double));
    compute_neuron_ranges(net, min_w, max_w);

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
                int pix = (range > 1e-9) ? (int)(((val - min_w[idx]) / range) * 255.0) : 0;
                fprintf(fp, "%d ", pix);
            }
        }
        fprintf(fp, "\n");
    }

    free(min_w);
    free(max_w);
    fclose(fp);
    printf("Exported PGM neuron montage to '%s'\n", filename);
}

// Logs every wrong test prediction to CSV
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

// Renders sample test digits in ASCII alongside predicted class probabilities
void network_preview_predictions(NeuralNetwork* net, Img** imgs, int count) {
    const char* shades = " .:-=+*#%@";
    for (int idx = 0; idx < count; idx++) {
        Matrix* pred = network_predict_img(net, imgs[idx]);
        int guess = mtx_argmax(pred);
        printf("\nSample #%d | True: %d | Predicted: %d (%.1f%%)\n",
               idx, imgs[idx]->label, guess, pred->data[guess][0] * 100.0);

        for (int r = 0; r < 28; r += 2) {
            for (int c = 0; c < 28; c++) {
                int level = (int)(imgs[idx]->img_data->data[r][c] * 9.0);
                if (level < 0) level = 0;
                if (level > 9) level = 9;
                putchar(shades[level]);
            }
            putchar('\n');
        }
        mtx_free(pred);
    }
}

// Exports trained weights as static const C arrays for standalone inference
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
    printf("Exported C header to '%s'\n", filename);
}