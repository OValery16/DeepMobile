// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "SGDMaker.h"
#include "SGD.h"
#include "../../EasyCL/EasyCL.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL Trainer *SGDMaker::instance(EasyCL *cl) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGDMaker.cpp: instance");
#endif


    return new SGD(cl);
}

