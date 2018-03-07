// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "AdagradState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL AdagradState::~AdagradState() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/AdagradState.cpp: ~AdagradState");
#endif


    delete sumSquaresWrapper;
    delete[] sumSquares;
}

AdagradState::AdagradState(EasyCL *cl, int numWeights, float fudgeFactor) :
        numWeights(numWeights) {
    sumSquares = new float[numWeights];
    for(int i = 0; i < numWeights; i++) {
        sumSquares[i] = fudgeFactor;
    }
    sumSquaresWrapper = cl->wrap(numWeights, sumSquares);
    sumSquaresWrapper->copyToDevice();
}


