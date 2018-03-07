#pragma once

#include "../../../sonyOpenCLexample1.h"

class EasyCL;
class CLWrapper;
class CLKernel;
class TemplatedKernel;

#include "../DeepCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class clNormalization {
    EasyCL *cl;
    int size;

    CLKernel *kernelNormalizationZeroMean;
    CLKernel *kernelNormalizationZeroMeanHalf;
    bool setup;
    int looping=16;
    int vectorSize=4;
    string vectorSizeString="4";

    int globalSize;
    int workgroupSize;


    public:
    int test=1;
    clNormalization(EasyCL *cl,float translate,float scale,int size,bool useHalf);
    VIRTUAL ~clNormalization();

    void NormalizationZeroMean(CLWrapper *imagesWrapper,int size);
    void NormalizationZeroMeanHalf(CLWrapper *imagesWrapper,int size);


    private:
    void buildKernelNormalization(float translate,float scale);
    void buildKernelNormalizationHalf(float translate,float scale);
    void setupBuilderNormalization(TemplatedKernel *builder,float translate,float scale);
    std::string getKernelTemplateNormalizationZeroMeanHalf();
    std::string getKernelTemplateNormalizationZeroMean();

    // [[[end]]]
};

