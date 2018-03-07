// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ForceBackpropLayerMaker.h"

#include "ForceBackpropLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

ForceBackpropLayer::ForceBackpropLayer(Layer *previousLayer, ForceBackpropLayerMaker *maker) :
       Layer(previousLayer, maker),
    outputPlanes(previousLayer->getOutputPlanes()),
    outputSize(previousLayer->getOutputSize()),
    batchSize(0),
    allocatedSize(0),
    output(0) {
}
VIRTUAL ForceBackpropLayer::~ForceBackpropLayer() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: ~ForceBackpropLayer");
#endif


    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string ForceBackpropLayer::getClassName() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: string ForceBackpropLayer::getClassName");
#endif


    return "ForceBackpropLayer";
}
VIRTUAL void ForceBackpropLayer::backward(float learningRate) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: backward");
#endif


    // do nothing...
}
VIRTUAL float *ForceBackpropLayer::getOutput() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: getOutput");
#endif


    return output;
}
VIRTUAL int ForceBackpropLayer::getPersistSize(int version) const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL bool ForceBackpropLayer::needsBackProp() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: needsBackProp");
#endif


    return true;
}
VIRTUAL void ForceBackpropLayer::printOutput() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: printOutput");
#endif


    if(output == 0) {
         return;
    }
    for(int n = 0; n < std::min(5,batchSize); n++) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: min(5,batchSize); n++) {");
#endif


        std::cout << "ForceBackpropLayer n " << n << ":" << std::endl;
        for(int plane = 0; plane < std::min(5, outputPlanes); plane++) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: min(5, outputPlanes); plane++) {");
#endif


            if(outputPlanes > 1) std::cout << "    plane " << plane << ":" << std::endl;
            for(int i = 0; i < std::min(5, outputSize); i++) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: min(5, outputSize); i++) {");
#endif


                std::cout << "      ";
                for(int j = 0; j < std::min(5, outputSize); j++) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: min(5, outputSize); j++) {");
#endif


                    std::cout << getOutput(n, plane, i, j) << " ";
//output[
//                            n * numPlanes * imageSize*imageSize +
//                            plane*imageSize*imageSize +
//                            i * imageSize +
//                            j ] << " ";
                }
                if(outputSize > 5) std::cout << " ... ";
                std::cout << std::endl;
            }
            if(outputSize > 5) std::cout << " ... " << std::endl;
        }
        if(outputPlanes > 5) std::cout << "   ... other planes ... " << std::endl;
    }
    if(batchSize > 5) std::cout << "   ... other n ... " << std::endl;
}
VIRTUAL void ForceBackpropLayer::print() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: print");
#endif


    printOutput();
}
//VIRTUAL bool ForceBackpropLayer::needErrorsBackprop() {
//    return true; // the main reason for this layer :-)
//}
VIRTUAL void ForceBackpropLayer::setBatchSize(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: setBatchSize");
#endif


    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    this->batchSize = batchSize;
    this->allocatedSize = allocatedSize;
    output = new float[ getOutputNumElements() ];
}
VIRTUAL void ForceBackpropLayer::forward() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: forward");
#endif


    int totalLinearLength = getOutputNumElements();
    float *input = previousLayer->getOutput();
    for(int i = 0; i < totalLinearLength; i++) {
        output[i] = input[i];
    }
}
VIRTUAL void ForceBackpropLayer::backward() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: backward");
#endif


  // do nothing... ?
}
VIRTUAL int ForceBackpropLayer::getOutputSize() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL int ForceBackpropLayer::getOutputPlanes() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: getOutputPlanes");
#endif


    return outputPlanes;
}
VIRTUAL int ForceBackpropLayer::getOutputCubeSize() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: getOutputCubeSize");
#endif


    return outputPlanes * outputSize * outputSize;
}
VIRTUAL int ForceBackpropLayer::getOutputNumElements() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: getOutputNumElements");
#endif


    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string ForceBackpropLayer::toString() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: string ForceBackpropLayer::toString");
#endif


    return toString();
}
VIRTUAL std::string ForceBackpropLayer::asString() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: string ForceBackpropLayer::asString");
#endif


    return std::string("") + "ForceBackpropLayer{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " }";
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayer.cpp: string");
#endif


}


