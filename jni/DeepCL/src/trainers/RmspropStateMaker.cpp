// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "RmspropStateMaker.h"
#include "RmspropState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

TrainerState *RmspropStateMaker::instance(EasyCL *cl, int numWeights) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/RmspropStateMaker.cpp: instance");
#endif


    RmspropState *state = new RmspropState(cl, numWeights);
    return state;
}
VIRTUAL bool RmspropStateMaker::created(TrainerState *state) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/RmspropStateMaker.cpp: created");
#endif


    return dynamic_cast< RmspropState * >(state) != 0;
}

