#include "img.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 10000

Img** csv_to_imgs(char* file_string, int number_of_imgs) {
    FILE *fp = fopen(file_string, "r");
    if (fp == NULL) {
        printf("Error: Could not open dataset file %s\n", file_string);
        exit(EXIT_FAILURE);
    }

    Img** imgs = malloc(number_of_imgs * sizeof(Img*));
    char row[MAXCHAR];

    // Skip CSV header line
    fgets(row, MAXCHAR, fp);

    int i = 0;
    while (i < number_of_imgs && fgets(row, MAXCHAR, fp) != NULL) {
        imgs[i] = malloc(sizeof(Img));
        imgs[i]->img_data = mtx_create(28, 28);

        int j = 0;
        char* token = strtok(row, ",");
        while (token != NULL) {
            if (j == 0) {
                imgs[i]->label = atoi(token);
            } else if (j <= 784) {
                imgs[i]->img_data->data[(j - 1) / 28][(j - 1) % 28] = atoi(token) / 255.0;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);
    return imgs;
}

void img_print(Img* img) {
    mtx_print(img->img_data);
    printf("Img Label: %d\n", img->label);
}

void img_free(Img* img) {
    if (img == NULL) return;
    mtx_free(img->img_data);
    free(img);
}

void imgs_free(Img** imgs, int n) {
    if (imgs == NULL) return;
    for (int i = 0; i < n; i++) {
        img_free(imgs[i]);
    }
    free(imgs);
}