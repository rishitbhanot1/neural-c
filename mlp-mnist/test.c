#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"
#include "operations.h"

// Helper for floating-point comparison
int approx_equal(double a, double b) {
    return fabs(a - b) < 1e-6;
}

double square(double x) {
    return x * x;
}

void test_creation_and_fill() {
    Matrix *m = mtx_create(2, 3);
    mtx_fill(m, 5);
    assert(m->rows == 2 && m->cols == 3);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            assert(approx_equal(m->data[i][j], 5.0));
        }
    }
    mtx_free(m);
    printf("[PASS] Creation and Fill\n");
}

void test_dot_product() {
    // A = [[1, 2, 3], [4, 5, 6]] (2x3)
    Matrix *A = mtx_create(2, 3);
    double val = 1.0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            A->data[i][j] = val++;

    // B = [[7, 8], [9, 10], [11, 12]] (3x2)
    Matrix *B = mtx_create(3, 2);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            B->data[i][j] = val++;

    // Expected A . B = [[58, 64], [139, 154]]
    Matrix *C = dot(A, B);
    assert(C->rows == 2 && C->cols == 2);
    assert(approx_equal(C->data[0][0], 58.0));
    assert(approx_equal(C->data[0][1], 64.0));
    assert(approx_equal(C->data[1][0], 139.0));
    assert(approx_equal(C->data[1][1], 154.0));

    mtx_free(A);
    mtx_free(B);
    mtx_free(C);
    printf("[PASS] Dot Product\n");
}

void test_elementwise_and_apply() {
    Matrix *A = mtx_create(2, 2);
    mtx_fill(A, 3);

    Matrix *B = apply(square, A); // Should be all 9.0
    Matrix *C = add(A, B);        // Should be all 12.0
    Matrix *D = subtract(C, A);   // Should be all 9.0
    Matrix *E = scale(0.5, D);    // Should be all 4.5

    assert(approx_equal(E->data[1][1], 4.5));

    mtx_free(A);
    mtx_free(B);
    mtx_free(C);
    mtx_free(D);
    mtx_free(E);
    printf("[PASS] Element-wise & Apply\n");
}

void test_transpose_and_flatten() {
    Matrix *A = mtx_create(2, 3);
    A->data[0][0] = 1; A->data[0][1] = 2; A->data[0][2] = 3;
    A->data[1][0] = 4; A->data[1][1] = 5; A->data[1][2] = 6;

    Matrix *T = transpose(A);
    assert(T->rows == 3 && T->cols == 2);
    assert(approx_equal(T->data[2][1], 6.0));

    Matrix *F0 = mtx_flatten(A, 0); // 6x1 column vector
    assert(F0->rows == 6 && F0->cols == 1);
    assert(approx_equal(F0->data[4][0], 5.0));

    mtx_free(A);
    mtx_free(T);
    mtx_free(F0);
    printf("[PASS] Transpose & Flatten\n");
}

void test_save_and_load() {
    Matrix *A = mtx_create(2, 2);
    A->data[0][0] = 1.25; A->data[0][1] = -3.5;
    A->data[1][0] = 0.0;  A->data[1][1] = 42.125;

    mtx_save(A, "test_matrix.txt");
    Matrix *B = mtx_load("test_matrix.txt");

    assert(B->rows == 2 && B->cols == 2);
    assert(approx_equal(B->data[0][1], -3.5));
    assert(approx_equal(B->data[1][1], 42.125));

    remove("test_matrix.txt"); // Clean up temp file
    mtx_free(A);
    mtx_free(B);
    printf("[PASS] Save & Load\n");
}

int main() {
    test_creation_and_fill();
    test_dot_product();
    test_elementwise_and_apply();
    test_transpose_and_flatten();
    test_save_and_load();
    printf("\nAll tests passed!\n");
    return 0;
}