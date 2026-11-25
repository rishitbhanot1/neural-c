#pragma once

#include "network.h"
#include "img.h"

void network_export_features_bmp(NeuralNetwork* net, const char* filename);
void network_export_features_pgm(NeuralNetwork* net, const char* filename);
void network_export_errors_csv(NeuralNetwork* net, Img** imgs, int n, const char* filename);
void network_preview_predictions(NeuralNetwork* net, Img** imgs, int count);
void network_export_header(NeuralNetwork* net, const char* filename);