// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "AdadeltaState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL AdadeltaState::~AdadeltaState() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/AdadeltaState.cpp: ~AdadeltaState");
#endif


    delete sumGradSquaredWrapper;
    delete sumUpdateSquaredWrapper;
    delete[] sumGradSquared;
    delete[] sumUpdateSquared;
}

AdadeltaState::AdadeltaState(EasyCL *cl, int numWeights) :
        numWeights(numWeights) {
    sumGradSquared = new float[numWeights];
    sumUpdateSquared = new float[numWeights];
    for(int i = 0; i < numWeights; i++) {
        sumGradSquared[i] = 0.0000001f; // should move this into fudgefactor I guess?
        sumUpdateSquared[i] = 0.0000001f; // should move this into fudgefactor I guess?
    }
    sumGradSquaredWrapper = cl->wrap(numWeights, sumGradSquared);
    sumUpdateSquaredWrapper = cl->wrap(numWeights, sumUpdateSquared);
    sumGradSquaredWrapper->copyToDevice();
    sumUpdateSquaredWrapper->copyToDevice();
}


