// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../../../sonyOpenCLexample1.h"
//#include "BackpropWeights.h"
#include "../../EasyCL/templates/TemplatedKernel.h"

#include "../../EasyCL/EasyCL.h"
#include "../activate/ActivationFunction.h"
#include "LayerDimensions.h"
#include "../DeepCLDllExport.h"

#define MEASURE_BACKWARD_PROP 0

#define STATIC static
#define VIRTUAL virtual

class BackpropWeightsNaive {
public:
    CLKernel *kernel;
    CLKernel *kernel2;

    int globalSize = 0;

    EasyCL *cl;
    LayerDimensions dim;

    int workgroupsize = 0;//cl->getMaxWorkgroupSize();
    bool setup;
    const float learningMultiplier;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackpropWeightsNaive();
    float learningRateToMultiplier(int batchSize);
    VIRTUAL void calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper, CLWrapper *weightWrapper, CLWrapper *previousStepVectorWrapper, CLWrapper *biasWrapper, CLWrapper *previousStepBiasVectorWrapper);
    BackpropWeightsNaive(EasyCL *cl, LayerDimensions dim);
    void buildKernelBackward( string kernelSource);
    void setupBuilderBackward(TemplatedKernel *builder);
    void setHintCompiler(int batchSize, TemplatedKernel *builder);
    void setupUpdateWeight(string &updateWeight,string &updateBiasWeights, string &defineVariableUpdates, string & updateBiasWeight);

    // [[[end]]]
};

