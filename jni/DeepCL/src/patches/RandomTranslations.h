// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#define VIRTUAL virtual
#define STATIC static

#include "../../../sonyOpenCLexample1.h"
#include "../layer/Layer.h"

class CLKernel;
class CLWrapper;
class PoolingForward;
class PoolingBackward;
class RandomTranslationsMaker;

class RandomTranslations : public Layer {
public:
    const int translateSize;
    const int numPlanes;
    const int inputSize;

    const int outputSize;

    float *output;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    RandomTranslations(Layer *previousLayer, RandomTranslationsMaker *maker);
    VIRTUAL ~RandomTranslations();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL int getOutputNumElements();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL void forward();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

