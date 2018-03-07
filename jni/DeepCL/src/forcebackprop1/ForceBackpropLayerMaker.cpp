// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ForceBackpropLayer.h"

#include "ForceBackpropLayerMaker.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL

Layer *ForceBackpropLayerMaker::createLayer(Layer *previousLayer) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/forcebackprop/ForceBackpropLayerMaker.cpp: createLayer");
#endif


    return new ForceBackpropLayer(previousLayer, this);
}

