// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "../../EasyCL/EasyCL.h"
#include "../util/stringhelper.h"
#include "../../EasyCL/util/StatefulTimer.h"

#include "ActivationBackwardCpu.h"
#include "ActivationBackwardGpuNaive.h"

#include "ActivationBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC ActivationBackward *ActivationBackward::instance(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: instance");
#endif


    return new ActivationBackwardGpuNaive(cl, numPlanes, inputSize, fn);
}
STATIC ActivationBackward *ActivationBackward::instanceForTest(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: instanceForTest");
#endif


    return new ActivationBackwardCpu(cl, numPlanes, inputSize, fn);
}
STATIC ActivationBackward *ActivationBackward::instanceSpecific(int idx, EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: instanceSpecific");
#endif


    if(idx == 0) {
        return new ActivationBackwardCpu(cl, numPlanes, inputSize, fn);
    }
    if(idx == 1) {
        return new ActivationBackwardGpuNaive(cl, numPlanes, inputSize, fn);
    }
    throw runtime_error("ActivationBackward::instanceSpecific, idx not known: " + toString(idx) );
}
ActivationBackward::ActivationBackward(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const *fn) :
        cl(cl),
        numPlanes(numPlanes),
        inputSize(inputSize),
        fn(fn),
        outputSize(inputSize) {
}
VIRTUAL int ActivationBackward::getInputNumElements(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: getInputNumElements");
#endif


    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int ActivationBackward::getOutputNumElements(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL void ActivationBackward::backward(int batchSize, float *inputs, float *gradOutput, float *gradInput) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: backward");
#endif


//    cout << "ActivationBackward::backward(float *)" << endl;
    StatefulTimer::instance()->timeCheck("ActivationBackward::backward float->wrapper start");

    CLWrapper *inputsWrapper = cl->wrap(getInputNumElements(batchSize), inputs);
    CLWrapper *gradOutputWrapper = cl->wrap(getOutputNumElements(batchSize), gradOutput);
    CLWrapper *gradInputWrapper = cl->wrap(getInputNumElements(batchSize), gradInput);

    inputsWrapper->copyToDevice();
    gradOutputWrapper->copyToDevice();

    backward(batchSize, inputsWrapper, gradOutputWrapper, gradInputWrapper);

    gradInputWrapper->copyToHost();

    delete inputsWrapper;
    delete gradOutputWrapper;
    delete gradInputWrapper;
    StatefulTimer::instance()->timeCheck("ActivationBackward::backward float->wrapper end");
}
VIRTUAL void ActivationBackward::backward(int batchSize, CLWrapper *inputsWrapper, CLWrapper *gradOutputWrapper, CLWrapper *gradInputWrapper) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationBackward.cpp: backward");
#endif


    throw runtime_error("ActivationBackward::backward wrappers not implemented");
}

