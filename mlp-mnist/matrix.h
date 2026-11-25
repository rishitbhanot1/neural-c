#pragma once

typedef struct {
    double **data;
    int rows;
    int cols;
} Matrix;

Matrix* mtx_create(int row, int col);
void mtx_fill(Matrix *m, int n);
void mtx_free(Matrix *m);
void mtx_print(Matrix *m);
Matrix* mtx_copy(Matrix *m);
void mtx_save(Matrix *m, char* file_string);
Matrix* mtx_load(char* file_string);
void mtx_rnd(Matrix *m, int n);
int mtx_argmax(Matrix* m);
Matrix* mtx_flatten(Matrix* m, int axis);