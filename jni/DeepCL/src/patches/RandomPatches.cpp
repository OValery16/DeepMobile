// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../net/NeuralNet.h"
#include "../layer/Layer.h"
#include "RandomPatches.h"
#include "RandomPatchesMaker.h"
#include "../util/RandomSingleton.h"
#include "PatchExtractor.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

RandomPatches::RandomPatches(Layer *previousLayer, RandomPatchesMaker *maker) :
        Layer(previousLayer, maker),
        patchSize(maker->_patchSize),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        outputSize(maker->_patchSize),
        output(0),
        batchSize(0),
        allocatedSize(0) {
    if(inputSize == 0) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0) {
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": output image size is 0");
    }
    if(previousLayer->needsBackProp()) {
        throw runtime_error("Error: RandomPatches layer does not provide backprop currently, so you cannot put it after a layer that needs backprop");
    }
}
VIRTUAL RandomPatches::~RandomPatches() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: ~RandomPatches");
#endif


    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string RandomPatches::getClassName() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: string RandomPatches::getClassName");
#endif


    return "RandomPatches";
}
VIRTUAL void RandomPatches::setBatchSize(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: setBatchSize");
#endif


    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    output = new float[ getOutputNumElements() ];
}
VIRTUAL int RandomPatches::getOutputNumElements() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *RandomPatches::getOutput() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: getOutput");
#endif


    return output;
}
VIRTUAL bool RandomPatches::needsBackProp() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: needsBackProp");
#endif


    return false;
}
VIRTUAL int RandomPatches::getOutputNumElements() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int RandomPatches::getOutputSize() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL int RandomPatches::getOutputPlanes() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: getOutputPlanes");
#endif


    return numPlanes;
}
VIRTUAL int RandomPatches::getPersistSize(int version) const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL bool RandomPatches::providesGradInputWrapper() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: providesGradInputWrapper");
#endif


    return false;
}
VIRTUAL bool RandomPatches::hasOutputWrapper() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: hasOutputWrapper");
#endif


    return false;
}
VIRTUAL void RandomPatches::forward() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: forward");
#endif


    float *upstreamOutput = previousLayer->getOutput();
    for(int n = 0; n < batchSize; n++) {
        int patchMargin = inputSize - outputSize;
        int patchRow = patchMargin / 2;
        int patchCol = patchMargin / 2;
        if(training) {
            patchRow = RandomSingleton::instance()->uniformInt(0, patchMargin);
            patchCol = RandomSingleton::instance()->uniformInt(0, patchMargin);
        }
        PatchExtractor::extractPatch(n, numPlanes, inputSize, patchSize, patchRow, patchCol, upstreamOutput, output);
    }
}
VIRTUAL std::string RandomPatches::asString() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/patches/RandomPatches.cpp: string RandomPatches::asString");
#endif


    return "RandomPatches{ inputPlanes=" + toString(numPlanes) + " inputSize=" + toString(inputSize) + " patchSize=" + toString(patchSize) + " }";
}


