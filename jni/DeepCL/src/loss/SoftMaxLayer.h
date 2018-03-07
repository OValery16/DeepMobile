// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../../../sonyOpenCLexample1.h"
#include "../../EasyCL/templates/TemplatedKernel.h"
#include "../layer/Layer.h"
#include "LossLayer.h"
#include "IAcceptsLabels.h"
#include "../../EasyCL/EasyCL.h"

class SoftMaxMaker;

#define VIRTUAL virtual
#define STATIC static

// this doesnt have any weights as such, just handles propagation, and backpropagation
// it will have the same shape as the previous layer, ie same imagesize, same number of planes
// the softmax will be per-plane, or maybe that is configurable?
// this will ALWAYS use multinomial logistic loss (ie cross-entropy loss), at least for now
class SoftMaxLayer : public LossLayer, public IAcceptsLabels {
public:
    const bool perPlane;
    const int imageSize;
    const int numPlanes;
    const int imageSizeSquared;

    float *output;
    float *gradInput;
    int allocatedSize;
    int batchSize;

    EasyCL *const ptrcl; // NOT owned by us
    CLKernel *kernel;
    bool setup;

    CLWrapper * inputWrapper;
	CLWrapper* gradInputWrapper;
    float* lossFloat;
    CLWrapper *labelWrapper;
    int *labelData;
    CLWrapper *lossWrapper;
    //CLWrapper *outputWrapper;
	int* nbRight;
	CLWrapper * 	nbRightWrapper;
    int globalSize;
    int workgroupsize;

    int batchsize;

    bool setup1;


    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    SoftMaxLayer(Layer *previousLayer, SoftMaxMaker *maker);
    VIRTUAL ~SoftMaxLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getOutput();
    VIRTUAL float *getGradInput();
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL int getBatchSize();
    VIRTUAL float calcLossFromLabels(int const *labels);
    VIRTUAL float calcLoss(float const *expectedValues);
    VIRTUAL void calcGradInputFromLabels(int const *labels);
    VIRTUAL void calcGradInput(float const *expectedValues);
    VIRTUAL int getNumLabelsPerExample();
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL int calcNumRightFromLabels(int const*labels);
    VIRTUAL void forward();
    VIRTUAL void getLabels(int *labels);  // need to allocate labels array first, and have called 'forward' first
    VIRTUAL std::string asString() const;
    VIRTUAL CLWrapper * getLossWrapper();
    VIRTUAL CLWrapper * getNbRightWrapper();

    void createKernel();
    void buildKernelSoftmax_forward();
    void setupBuilderSoftmax_forward(TemplatedKernel *builder);
    STATIC std::string getKernelTemplateSoftmax_forward();
    void runSoftmax_forward(CLWrapper *inputWrapper/*,CLWrapper *outputWrapper*/,int const *labels) ;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();

    // [[[end]]]
};

