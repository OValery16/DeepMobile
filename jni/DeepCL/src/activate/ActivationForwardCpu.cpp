// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "ActivationFunction.h"

#include "ActivationForwardCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

ActivationForwardCpu::ActivationForwardCpu(EasyCL *cl, int numPlanes, int inputSize, ActivationFunction const*fn) :
        ActivationForward(cl, numPlanes, inputSize, fn) {
}
VIRTUAL void ActivationForwardCpu::forward(int batchSize, CLWrapper *inputWrapper, CLWrapper *outputWrapper) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationForwardCpu.cpp: forward");
#endif


//    cout << "ActivationForwardCpu::forward(CLWrapper *)" << endl;

    inputWrapper->copyToHost();

    float *input = reinterpret_cast<float *>(inputWrapper->getHostArray());
    float *output = new float[ getOutputNumElements(batchSize) ];

    forward(batchSize, input, output);

    float *outputHostArray = reinterpret_cast<float *>(outputWrapper->getHostArray());
    memcpy(outputHostArray, output, sizeof(float) * getOutputNumElements(batchSize) );

    outputWrapper->copyToDevice();

    delete[] output;
}
VIRTUAL void ActivationForwardCpu::forward(int batchSize, float *input, float *output) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationForwardCpu.cpp: forward");
#endif


//    float *output = new float[ getOutputNumElements(batchSize) ];
//    cout << "ActivationForwardCpu::forward(float *)" << endl;
    StatefulTimer::instance()->timeCheck("ActivationForwardCpu::forward start");
    int totalLinearSize = batchSize * numPlanes * inputSize * inputSize;
    for(int i = 0; i < totalLinearSize; i++) {
        output[i] = fn->calc(input[i]);
    }
    StatefulTimer::instance()->timeCheck("ActivationForwardCpu::forward end");
//    return output;
}

