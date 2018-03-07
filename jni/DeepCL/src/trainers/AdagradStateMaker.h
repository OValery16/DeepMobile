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

#include "TrainerStateMaker.h"

#define VIRTUAL virtual
#define STATIC static

class AdagradStateMaker : public TrainerStateMaker {
public:
    float fudgeFactor;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    AdagradStateMaker(float fudgeFactor);
    TrainerState *instance(EasyCL *cl, int numWeights);
    VIRTUAL bool created(TrainerState *state);

    // [[[end]]]
};

