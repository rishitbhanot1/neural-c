#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 100

Matrix* mtx_create(int row, int col) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = row;
    matrix->cols = col;
    matrix->data = malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        matrix->data[i] = malloc(col * sizeof(double));
    }
    return matrix;
}

void mtx_fill(Matrix *m, int n) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = n;
        }
    }
}

void mtx_free(Matrix *m) {
    if (m == NULL) return;
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

void mtx_print(Matrix *m) {
    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%1.3f ", m->data[i][j]);
        }
        printf("\n");
    }
}

Matrix* mtx_copy(Matrix *m) {
    Matrix* cmat = mtx_create(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            cmat->data[i][j] = m->data[i][j];
        }
    }
    return cmat;
}

void mtx_save(Matrix *m, char* file_string) {
    FILE* file = fopen(file_string, "w");
    if (file == NULL) {
        printf("Error: Could not open %s for writing\n", file_string);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d\n", m->rows);
    fprintf(file, "%d\n", m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fprintf(file, "%.6f\n", m->data[i][j]);
        }
    }
    printf("Successfully saved matrix to %s\n", file_string);
    fclose(file);
}

Matrix* mtx_load(char* file_string) {
    FILE* file = fopen(file_string, "r");
    if (file == NULL) {
        printf("Error: Could not open %s for reading\n", file_string);
        exit(EXIT_FAILURE);
    }
    char entry[MAXCHAR];
    fgets(entry, MAXCHAR, file);
    int rows = atoi(entry);
    fgets(entry, MAXCHAR, file);
    int cols = atoi(entry);
    Matrix* m = mtx_create(rows, cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fgets(entry, MAXCHAR, file);
            m->data[i][j] = strtod(entry, NULL);
        }
    }
    printf("Successfully loaded matrix from %s\n", file_string);
    fclose(file);
    return m;
}

double udiff(double low, double high) {
    double difference = high - low;
    int scale = 10000;
    int scaled_difference = (int)(difference * scale);
    if (scaled_difference <= 0) return low;
    return low + (1.0 * (rand() % scaled_difference) / scale);
}

void mtx_rnd(Matrix* m, int n) {
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i][j] = udiff(min, max);
        }
    }
}

int mtx_argmax(Matrix* m) {
    double max_score = m->data[0][0];
    int max_idx = 0;
    for (int i = 1; i < m->rows; i++) {
        if (m->data[i][0] > max_score) {
            max_score = m->data[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}

Matrix* mtx_flatten(Matrix* m, int axis) {
    Matrix* mat;
    if (axis == 0) {
        mat = mtx_create(m->rows * m->cols, 1);
    } else if (axis == 1) {
        mat = mtx_create(1, m->rows * m->cols);
    } else {
        printf("Argument to matrix_flatten must be 0 or 1\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (axis == 0) mat->data[i * m->cols + j][0] = m->data[i][j];
            else if (axis == 1) mat->data[0][i * m->cols + j] = m->data[i][j];
        }
    }
    return mat;
}