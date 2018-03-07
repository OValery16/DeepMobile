// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "AdagradStateMaker.h"
#include "AdagradState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

AdagradStateMaker::AdagradStateMaker(float fudgeFactor) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/AdagradStateMaker.cpp: AdagradStateMaker");
#endif


    this->fudgeFactor = fudgeFactor;
}
TrainerState *AdagradStateMaker::instance(EasyCL *cl, int numWeights) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/AdagradStateMaker.cpp: instance");
#endif


    AdagradState *state = new AdagradState(cl, numWeights, fudgeFactor);
    return state;
}
VIRTUAL bool AdagradStateMaker::created(TrainerState *state) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/AdagradStateMaker.cpp: created");
#endif


    return dynamic_cast< AdagradState * >(state) != 0;
}

