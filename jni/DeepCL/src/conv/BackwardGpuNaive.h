#pragma once

#include "../../../sonyOpenCLexample1.h"

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/templates/TemplatedKernel.h"

#include "../../../sonyOpenCLexample1.h"
#include <iostream>
#include <string>

#include "../../EasyCL/EasyCL.h"
#include "../activate/ActivationFunction.h"
#include "LayerDimensions.h"

#include "../DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuNaive {
public:
    CLKernel *kernel;
    CLKernel *kernel2;

    int globalSize;
    int workgroupsize;

    bool setup;

    EasyCL *cl;
    LayerDimensions dim;
//    CLKernel *broadcastMultiply;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuNaive();
    VIRTUAL void backward(int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);
    BackwardGpuNaive(EasyCL *cl, LayerDimensions dim);
    void setupBuilderBackward(TemplatedKernel *builder);
    void buildKernelBackward( string kernelSource);
    void inferenceBackward(string& kernelSource);
    void  setActivationFunction(TemplatedKernel *builder);



    // [[[end]]]
};

