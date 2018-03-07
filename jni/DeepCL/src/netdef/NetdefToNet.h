// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../../../sonyOpenCLexample1.h"
#include <string>
#include <sstream>

#include "../DeepCLDllExport.h"

class NeuralNet;
class WeightsInitializer;

#define VIRTUAL virtual
#define STATIC static

/// \brief Add layers to a NeuralNet object, based on a netdef-string
///
/// eg "8c5-mp2" will add a convolutional layer with 8 filter, each 
/// 5 by 5; and one max-pooling layer, over 2x2
/// based on the notation proposed in 
/// [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf)
PUBLICAPI
class DeepCL_EXPORT NetdefToNet {
public:
	STATIC bool createNetFromNetdefPrediction(int batchsize,NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer);
    PUBLICAPI STATIC bool createNetFromNetdef(int batchsize,NeuralNet *net, std::string netdef);
    PUBLICAPI STATIC bool createNetFromNetdefCharStar(int batchsize,NeuralNet *net, const char *netdef);
    STATIC bool createNetFromNetdef(int batchsize,NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer);
    static void createConv(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef, std::string pooling_layerDef, WeightsInitializer *weightsInitializer) ;
    static void createMaxPooling(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
    static void createDropout(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
    static void createActivation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
    static void createRandomTranslation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
    static void createFullyConnectedLayer(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef,  WeightsInitializer *weightsInitializer) ;


    // [[[end]]]
};

