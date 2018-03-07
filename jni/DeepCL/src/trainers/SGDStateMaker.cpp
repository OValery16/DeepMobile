// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "SGDStateMaker.h"
#include "SGDState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

TrainerState *SGDStateMaker::instance(EasyCL *cl, int numWeights) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGDStateMaker.cpp: instance");
#endif


    SGDState *sgd = new SGDState(cl, numWeights);
    return sgd;
}
VIRTUAL bool SGDStateMaker::created(TrainerState *state) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGDStateMaker.cpp: created");
#endif


    return dynamic_cast< SGDState * >(state) != 0;
}

