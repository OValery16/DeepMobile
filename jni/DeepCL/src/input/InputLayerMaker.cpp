// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "InputLayerMaker.h"
#include "InputLayer.h"

using namespace std;

Layer *InputLayerMaker::createLayer(Layer *previousLayer) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/input/InputLayerMaker.cpp: createLayer");
#endif


    return new InputLayer(this);
}

