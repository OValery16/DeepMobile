// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "../../EasyCL/EasyCL.h"
#include "../util/stringhelper.h"
#include "../../EasyCL/util/StatefulTimer.h"

#include "DropoutBackwardCpu.h"
#include "DropoutBackwardGpuNaive.h"

#include "DropoutBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC DropoutBackward *DropoutBackward::instance(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: instance");
#endif


    return new DropoutBackwardGpuNaive(cl, numPlanes, inputSize, dropRatio);
}
STATIC DropoutBackward *DropoutBackward::instanceForTest(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: instanceForTest");
#endif


    return new DropoutBackwardGpuNaive(cl, numPlanes, inputSize, dropRatio);
}
STATIC DropoutBackward *DropoutBackward::instanceSpecific(int idx, EasyCL *cl, int numPlanes, int inputSize, float dropRatio) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: instanceSpecific");
#endif


    if(idx == 0) {
        return new DropoutBackwardCpu(cl, numPlanes, inputSize, dropRatio);
    }
    if(idx == 1) {
        return new DropoutBackwardGpuNaive(cl, numPlanes, inputSize, dropRatio);
    }
    throw runtime_error("DropoutBackward::instanceSpecific, idx not known: " + toString(idx) );
}
DropoutBackward::DropoutBackward(EasyCL *cl, int numPlanes, int inputSize, float dropRatio) :
        cl(cl),
        numPlanes(numPlanes),
        inputSize(inputSize),
        dropRatio(dropRatio),
//        dropoutSizeSquared(dropoutSize * dropoutSize),
        outputSize(inputSize) {
//    if(inputSize % dropoutSize != 0) {
//        throw runtime_error("inputSize should be an exact multiple of dropoutsize: " + toString(inputSize) + " " + toString(dropoutSize) );
//    }
}
VIRTUAL int DropoutBackward::getInputNumElements(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: getInputNumElements");
#endif


    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int DropoutBackward::getOutputNumElements(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL void DropoutBackward::backward(int batchSize, uchar *mask, float *gradOutput, float *gradInput) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: backward");
#endif


//    cout << "DropoutBackward::backward(float *)" << endl;
    StatefulTimer::instance()->timeCheck("DropoutBackward::backward float->wrapper start");
    CLWrapper *maskWrapper = cl->wrap(getOutputNumElements(batchSize), mask);
    CLWrapper *gradOutputWrapper = cl->wrap(getOutputNumElements(batchSize), gradOutput);
    CLWrapper *gradInputWrapper = cl->wrap(getInputNumElements(batchSize), gradInput);

    maskWrapper->copyToDevice();
    gradOutputWrapper->copyToDevice();
    gradInputWrapper->createOnDevice();

    backward(batchSize, maskWrapper, gradOutputWrapper, gradInputWrapper);

    gradInputWrapper->copyToHost();

    delete maskWrapper;
    delete gradOutputWrapper;
    delete gradInputWrapper;
    StatefulTimer::instance()->timeCheck("DropoutBackward::backward float->wrapper end");
}
VIRTUAL void DropoutBackward::backward(int batchSize, CLWrapper *maskWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/dropout/DropoutBackward.cpp: backward");
#endif


    throw runtime_error("DropoutBackward::backward wrappers not implemented");
}

