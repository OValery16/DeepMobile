// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../../EasyCL/EasyCL.h"

class OutputData;

class IAcceptsLabels {
public:
    virtual float calcLossFromLabels(int const*labels) = 0;
    virtual void calcGradInputFromLabels(int const*labels) = 0;
    virtual int calcNumRightFromLabels(int const*labels) = 0;
    virtual int getNumLabelsPerExample() = 0;
    virtual CLWrapper * getLossWrapper() = 0;
    virtual CLWrapper * getNbRightWrapper() = 0;
};

#include "../../../sonyOpenCLexample1.h"

