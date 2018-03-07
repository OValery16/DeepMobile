//#include "clblas/ClBlasInstance.h"
//#include "../clblas/ClBlasHelper.h"
#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/templates/TemplatedKernel.h"
#include "../../kernelManager/ConfigManager.h"

#include "clNormalization.h"

#include <sstream>
#include <iomanip>

#include <iostream>
#include <stdexcept>
using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL
#define PUBLIC

std::string to_string_with_precision6(const float a_value, const int n = 2)
{
	std::stringstream ss;
	if (a_value==0)
		ss << std::fixed << 0;
	else
		ss << std::fixed << std::setprecision(n) << a_value;
    return ss.str();
}



PUBLIC clNormalization::clNormalization(EasyCL *cl,float translate,float scale,int size,bool useHalf) :
        cl(cl),size(size)
        {
	setup=false;
	string filepath="default";
	ConfigManager* binariesManager=new ConfigManager();
	if (useHalf){
		string identifier="NormalizationZeroMeanHalf"+std::to_string(translate)+"-"+std::to_string(scale);
		if (binariesManager->alreadyCompiledKernel("NormalizationZeroMeanHalf","",identifier,filepath)){
			kernelNormalizationZeroMeanHalf=cl->buildKernelFromString(identifier,"", "NormalizationZeroMeanHalf", "", filepath);
		}else {
			buildKernelNormalizationHalf(translate,scale);
		}
	}else{
		LOGI("not half");
		string identifier="NormalizationZeroMean"+std::to_string(translate)+"-"+std::to_string(scale);
		if (binariesManager->alreadyCompiledKernel("NormalizationZeroMean","",identifier,filepath)){
			kernelNormalizationZeroMean=cl->buildKernelFromString(identifier,"", "NormalizationZeroMean", "", filepath);
		}else {
			buildKernelNormalization(translate,scale);
		}
	}
	delete binariesManager;




}
PUBLIC VIRTUAL clNormalization::~clNormalization() {

delete kernelNormalizationZeroMean;
delete kernelNormalizationZeroMeanHalf;
}


void clNormalization::buildKernelNormalization(float translate,float scale) {
    TemplatedKernel builder(cl);


        setupBuilderNormalization(&builder,translate,scale);

        LOGE("toto2");
        string identifier2="NormalizationZeroMean"+std::to_string(translate)+"-"+std::to_string(scale);

        this->kernelNormalizationZeroMean = builder.buildKernel(
           		identifier2,
               "NormalizationZeroMean.cl",
               getKernelTemplateNormalizationZeroMean(),
               "NormalizationZeroMean",
               false
        );
    }
void clNormalization::buildKernelNormalizationHalf(float translate,float scale) {
	TemplatedKernel builder(cl);


	setupBuilderNormalization(&builder,translate,scale);

	LOGE("toto2");
	string identifier2="NormalizationZeroMeanHalf"+std::to_string(translate)+"-"+std::to_string(scale);

	this->kernelNormalizationZeroMeanHalf = builder.buildKernel(
			identifier2,
            "NormalizationZeroMeanHalf.cl",
	        getKernelTemplateNormalizationZeroMeanHalf(),
	        "NormalizationZeroMeanHalf",
	        false
	        );

}


void clNormalization::setupBuilderNormalization(TemplatedKernel *builder,float translate,float scale) {

    builder->set("translate", translate);
    builder->set("scale", scale);
    if (size%(64)==0){
    	looping=16;
    	vectorSizeString="4";
    	vectorSize=4;
    }else if((size%32)==0){
		looping=8;
		vectorSizeString="4";
		vectorSize=4;
	}else if((size%32)==0){
		looping=8;
		vectorSizeString=4;
		vectorSize=4;
	}else if((size%16)==0){
			looping=8;
			vectorSizeString="4";
			vectorSize=4;
	}else {
		looping=1;
		vectorSizeString="";
		vectorSize=1;
	}




    builder->set("looping", looping);
    builder->set("vectorSize", vectorSizeString);
}

void clNormalization::NormalizationZeroMean(CLWrapper *imagesWrapper,int size){

	clock_t startTimer1, stopTimer1;

	startTimer1=clock();
	if (setup!=true){

		workgroupSize = kernelNormalizationZeroMean->get_kernel_work_group_size();
		globalSize = size/(looping*vectorSize);
	    globalSize = (( globalSize + workgroupSize - 1) / workgroupSize) * workgroupSize;
	}


	kernelNormalizationZeroMean->inout(imagesWrapper);

	kernelNormalizationZeroMean->run_1d(globalSize, workgroupSize);
	cl->finish();
	stopTimer1 = clock();
	LOGI("kernelNormalizationZeroMean took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;


}

void clNormalization::NormalizationZeroMeanHalf(CLWrapper *imagesWrapper,int size){

	LOGE("here3");
	kernelNormalizationZeroMeanHalf->inout(imagesWrapper);
	LOGE("here4");

	int workgroupSize = kernelNormalizationZeroMeanHalf->get_kernel_work_group_size();
	LOGI("workgroupsize %d, workgroupsize2 %d",workgroupSize, cl->getMaxWorkgroupSize());
	clock_t startTimer1, stopTimer1;

	int globalSize = size/(looping*vectorSize);
    globalSize = (( globalSize + workgroupSize - 1) / workgroupSize) * workgroupSize;
    startTimer1=clock();
    kernelNormalizationZeroMeanHalf->run_1d(globalSize, workgroupSize);
	cl->finish();
	stopTimer1 = clock();
	LOGI("kernelNormalizationZeroMeanHalf took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;


}





STATIC std::string clNormalization::getKernelTemplateNormalizationZeroMeanHalf() {
	const char * kernelSource =
			"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
			"kernel void NormalizationZeroMeanHalf(global half{{vectorSize}} * im_Matrix) {\n"
			"    \n"
            "    int gid = get_global_id(0)*{{looping}};	\n"
			"    #pragma unroll\n"
			"    for (int w=0; w<{{looping}}; ++w) "
			"        im_Matrix[gid+w]=(im_Matrix[gid+w]+ {{translate}}) * {{scale}};                      \n"
			"}\n"
			"							    \n"
			";\n";

	    return kernelSource;
	}


STATIC std::string clNormalization::getKernelTemplateNormalizationZeroMean() {

	const char * kernelSource =
			"kernel void NormalizationZeroMean(global float{{vectorSize}} * im_Matrix) {\n"
			"    \n"
            "    int gid = get_global_id(0)*{{looping}};	\n"
			"    #pragma unroll\n"
			"    for (int w=0; w<{{looping}}; ++w) "
			"        im_Matrix[gid+w]=(im_Matrix[gid+w]+ {{translate}}) * {{scale}};                      \n"
			"}\n"
			"							    \n"
			";\n";

	    return kernelSource;
	}




//STATIC std::string clNormalization::getKernelTemplatereNormalizationZeroMean() {
//
//	const char * kernelSource =
//			"#define gTranslate {{translate}}\n"
//			"#define gScale {{scale}}\n"
//			"    \n"
//			"kernel void NormalizationZeroMean(global float * im_Matrix) {\n"
//			"    \n"
//			"    int gid = get_global_id(0);	\n"
//			"        im_Matrix[gid]=(im_Matrix[gid]+ {{translate}}) * {{scale}};                      \n"
//			"}\n"
//			"							    \n"
//			";\n";
//
//
//	    return kernelSource;
//	}
