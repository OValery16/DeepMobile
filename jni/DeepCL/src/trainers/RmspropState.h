// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../../../sonyOpenCLexample1.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "TrainerState.h"

class EasyCL;
class CLKernel;

#include "../DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class DeepCL_EXPORT RmspropState : public TrainerState {
public:
    const int numWeights;

    float *meanSquare;
    CLWrapper *meanSquareWrapper;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~RmspropState();
    RmspropState(EasyCL *cl, int numWeights);

    // [[[end]]]
};

