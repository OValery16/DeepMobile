// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationLayer.h"

#include "NormalizationLayerMaker.h"

using namespace std;

Layer *NormalizationLayerMaker::createLayer(Layer *previousLayer) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayerMaker.cpp: createLayer");
#endif


    return new NormalizationLayer(previousLayer, this);
}


