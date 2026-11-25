#include "activation.h"
#include <math.h>
#include "operations.h"

double sigmoid(double input) {
    return 1.0 / (1.0 + exp(-1.0 * input));
}

// Note: Expects m to already be sigmoid-activated output: m * (1 - m)
Matrix* sigmoidPrime(Matrix* m) {
    Matrix* ones = mtx_create(m->rows, m->cols);
    mtx_fill(ones, 1);
    Matrix* subtracted = subtract(ones, m);
    Matrix* multiplied = multiply(m, subtracted);
    mtx_free(ones);
    mtx_free(subtracted);
    return multiplied;
}

Matrix* softmax(Matrix* m) {
    double max_val = m->data[0][0];
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (m->data[i][j] > max_val) {
                max_val = m->data[i][j];
            }
        }
    }

    double total = 0.0;
    Matrix* mat = mtx_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat->data[i][j] = exp(m->data[i][j] - max_val);
            total += mat->data[i][j];
        }
    }

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] /= total;
        }
    }
    return mat;
}