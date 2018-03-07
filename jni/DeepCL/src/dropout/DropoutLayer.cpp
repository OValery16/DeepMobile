// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../net/NeuralNet.h"
#include "../layer/Layer.h"
#include "DropoutLayer.h"
#include "DropoutMaker.h"
#include "DropoutForward.h"
#include "DropoutBackward.h"
#include "../util/RandomSingleton.h"
#include "../clmath/MultiplyBuffer.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutLayer::DropoutLayer(EasyCL *cl, Layer *previousLayer, DropoutMaker *maker) :
        Layer(previousLayer, maker),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        dropRatio(maker->_dropRatio),
        outputSize(previousLayer->getOutputSize()),
        random(RandomSingleton::instance()),
        cl(cl),
        masks(0),
        output(0),
        gradInput(0),
        maskWrapper(0),
        outputWrapper(0),
        gradInputWrapper(0),
//        outputCopiedToHost(false),
//        gradInputCopiedToHost(false),
        batchSize(0),
        allocatedSize(0) {
    if(inputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Dropout layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Dropout layer " + toString(layerIndex) + ": output image size is 0");
    }
    dropoutForwardImpl = DropoutForward::instance(cl, numPlanes, inputSize, dropRatio);
    dropoutBackwardImpl = DropoutBackward::instance(cl, numPlanes, inputSize, dropRatio);
    multiplyBuffer = new MultiplyBuffer(cl);
}
VIRTUAL DropoutLayer::~DropoutLayer() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: ~DropoutLayer");
#endif


    delete multiplyBuffer;
    delete dropoutForwardImpl;
    delete dropoutBackwardImpl;
    if(maskWrapper != 0) {
        delete maskWrapper;
    }
    if(outputWrapper != 0) {
        delete outputWrapper;
    }
    if(masks != 0) {
        delete[] masks;
    }
    if(output != 0) {
        delete[] output;
    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string DropoutLayer::getClassName() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: string DropoutLayer::getClassName");
#endif


    return "DropoutLayer";
}
VIRTUAL void DropoutLayer::fortesting_setRandomSingleton(RandomSingleton *random) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: fortesting_setRandomSingleton");
#endif


    this->random = random;
}
VIRTUAL void DropoutLayer::setBatchSize(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: setBatchSize");
#endif


//    cout << "DropoutLayer::setBatchSize" << endl;
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(maskWrapper != 0) {
        delete maskWrapper;
    }
    if(outputWrapper != 0) {
        delete outputWrapper;
    }
    if(masks != 0) {
        delete[] masks;
    }
    if(output != 0) {
        delete[] output;
    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    masks = new unsigned char[ getOutputNumElements() ];
    maskWrapper = cl->wrap(getOutputNumElements(), masks);
    output = new float[ getOutputNumElements() ];
    outputWrapper = cl->wrap(getOutputNumElements(), output);
    gradInput = new float[ previousLayer->getOutputNumElements() ];
    gradInputWrapper = cl->wrap(previousLayer->getOutputNumElements(), gradInput);
    gradInputWrapper->createOnDevice();
}
VIRTUAL int DropoutLayer::getOutputNumElements() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *DropoutLayer::getOutput() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getOutput");
#endif


    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
//        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool DropoutLayer::needsBackProp() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: needsBackProp");
#endif


    return previousLayer->needsBackProp(); // seems highly unlikely that we wouldnt have to backprop
                                           // but anyway, we dont have any weights ourselves
                                           // so just depends on upstream
}
VIRTUAL int DropoutLayer::getOutputNumElements() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getOutputNumElements");
#endif


//    int outputSize = inputSize / dropoutSize;
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int DropoutLayer::getOutputSize() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL int DropoutLayer::getOutputPlanes() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getOutputPlanes");
#endif


    return numPlanes;
}
VIRTUAL int DropoutLayer::getPersistSize(int version) const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL bool DropoutLayer::providesGradInputWrapper() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: providesGradInputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *DropoutLayer::getGradInputWrapper() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapper;
}
VIRTUAL bool DropoutLayer::hasOutputWrapper() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: hasOutputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *DropoutLayer::getOutputWrapper() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getOutputWrapper");
#endif


    return outputWrapper;
}
VIRTUAL float *DropoutLayer::getGradInput() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getGradInput");
#endif


    return gradInput;
}
VIRTUAL ActivationFunction const *DropoutLayer::getActivationFunction() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: getActivationFunction");
#endif


    return new LinearActivation();
}
//VIRTUAL void DropoutLayer::generateMasks() {
//    int totalInputLinearSize = getOutputNumElements();
////    int numBytes = (totalInputLinearSize+8-1)/8;
////    unsigned char *bitsField = new unsigned char[numBytes];
//    int idx = 0;
//    unsigned char thisByte = 0;
//    int bitsPacked = 0;
//    for(int i = 0; i < totalInputLinearSize; i++) {
//        //double value = ((int)random() % 10000) / 20000.0f + 0.5f;
//        // 1 means we pass value through, 0 means we drop
//        // dropRatio is probability that mask value is 0 therefore
//        // so higher dropRatio => more likely to be 0
//        unsigned char bit = random->_uniform() <= dropRatio ? 0 : 1;
////        unsigned char bit = 0;
//        thisByte <<= 1;
//        thisByte |= bit;
//        bitsPacked++;
//        if(bitsPacked >= 8) {
//            masks[idx] = thisByte;
//            idx++;
//            bitsPacked = 0;
//        }
//    }
//}
VIRTUAL void DropoutLayer::generateMasks() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: generateMasks");
#endif


    int totalInputLinearSize = getOutputNumElements();
    for(int i = 0; i < totalInputLinearSize; i++) {
        masks[i] = random->_uniform() <= dropRatio ? 0 : 1;
    }
}
VIRTUAL void DropoutLayer::forward() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: forward");
#endif


    CLWrapper *upstreamOutputWrapper = 0;
    if(previousLayer->hasOutputWrapper()) {
        upstreamOutputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *upstreamOutput = previousLayer->getOutput();
        upstreamOutputWrapper = cl->wrap(previousLayer->getOutputNumElements(), upstreamOutput);
        upstreamOutputWrapper->copyToDevice();
    }

//    cout << "training: " << training << endl;
    if(training) {
        // create new masks...
        generateMasks();
        maskWrapper->copyToDevice();
        dropoutForwardImpl->forward(batchSize, maskWrapper, upstreamOutputWrapper, outputWrapper);
    } else {
        // if not training, then simply skip the dropout bit, copy the buffers directly
        multiplyBuffer->multiply(getOutputNumElements(), dropRatio, upstreamOutputWrapper, outputWrapper);
    }
    if(!previousLayer->hasOutputWrapper()) {
        delete upstreamOutputWrapper;
    }
}
VIRTUAL void DropoutLayer::backward() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: backward");
#endif


    // have no weights to backprop to, just need to backprop the errors

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if(nextLayer->providesGradInputWrapper()) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap(getOutputNumElements(), nextLayer->getGradInput());
        gradOutputWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }
    dropoutBackwardImpl->backward(batchSize, maskWrapper, gradOutputWrapper, gradInputWrapper);
    if(weOwnErrorsWrapper) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string DropoutLayer::asString() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutLayer.cpp: string DropoutLayer::asString");
#endif


    return "DropoutLayer{ dropRatio=" + toString(dropRatio) + " }";
}


