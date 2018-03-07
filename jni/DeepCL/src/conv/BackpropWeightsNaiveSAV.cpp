// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsNaive.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "../util/stringhelper.h"



using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}
VIRTUAL BackpropWeightsNaive::~BackpropWeightsNaive() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackpropWeightsNaive.cpp: ~BackpropWeightsNaive");
#endif


	if (dim.test)
		delete kernel;
    delete kernel2;
}
PUBLIC VIRTUAL void BackpropWeightsNaive::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper, CLWrapper *weightWrapper, CLWrapper *previousStepVectorWrapper, CLWrapper *biasWrapper, CLWrapper *previousStepBiasVectorWrapper){
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackpropWeightsNaive.cpp: calcGradWeights");
#endif

#if MEASURE_BACKWARD_PROP == 1
	clock_t startTimer1, stopTimer1;
	startTimer1=clock();

	float* temp= 0;
	float* temp2= 0;
	float *gradBias=0;
	float *grad=0;



	if (dim.test){

		StatefulTimer::instance()->timeCheck("BackpropWeightsNaive start");



		temp= new float[dim.filtersSize];
		temp2= new float[(dim.filtersSize/(dim.filterSizeSquared*dim.inputPlanes))];

		kernel
		   ->in(learningMultiplier)
		   ->in(batchSize)
		   ->in(gradOutputWrapper)
			->in(imagesWrapper)
		   ->inout(gradWeightsWrapper);
		if(dim.biased) {
			kernel->inout(gradBiasWrapper);
		}

		int globalSize0 = dim.filtersSize;

		int workgroupsize0 = kernel->get_kernel_work_group_size();//cl->getMaxWorkgroupSize();
		LOGI("------------------------globalSize %d",globalSize0);
		globalSize0 = ((globalSize0 + workgroupsize0 - 1) / workgroupsize0) * workgroupsize0;
		LOGI("------------------------globalSize %d workgroupsize %d",globalSize0,workgroupsize0);
		startTimer1=clock();
		kernel->run_1d(globalSize0, workgroupsize0);
		cl->finish();
		stopTimer1 = clock();
			LOGI("----------------------calculate weight took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;



		gradWeightsWrapper->copyToHost();
		gradBiasWrapper->copyToHost();
		gradBias=(float *)gradBiasWrapper->getHostArray();
		grad=(float *)gradWeightsWrapper->getHostArray();
		for (int i= 0; i < dim.filtersSize;i++)
			temp[i]=grad[i];
		for (int i= 0; i < (dim.filtersSize/(dim.filterSizeSquared*dim.inputPlanes));i++)
			temp2[i]=gradBias[i];
	}
#endif
	if (not setup){

		if (dim.isConv){
			globalSize=batchSize*dim.filtersSize;
			workgroupsize = batchSize;
		}else{
			globalSize = dim.filtersSize;
			workgroupsize = kernel2->get_kernel_work_group_size();
		}
		#if MEASURE_BACKWARD_PROP == 1
			LOGI("------------------------globalSize2 %d",globalSize);
		#endif
		globalSize = ((globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		#if MEASURE_BACKWARD_PROP == 1
			LOGI("------------------------globalSiz2e %d workgroupsize2 %d",globalSize,workgroupsize);
		#endif
		setup=true;
//		LOGI("wg size %d",workgroupsize);
//		LOGI("allowed %zu",kernel2->get_kernel_work_group_size());
		if (1){
			kernel2->input(dim.momentum);
			kernel2->input(dim.learning_rate);
			kernel2->inout(weightWrapper);
			kernel2->input(previousStepVectorWrapper);
			if (dim.biased){
				kernel2->input(biasWrapper);
				kernel2->input(previousStepBiasVectorWrapper);
			}
		}
		kernel2
       ->in(learningMultiplier)
       ->in(gradOutputWrapper)
        ->in(imagesWrapper);
		#if MEASURE_BACKWARD_PROP==1
			kernel2->inout(gradWeightsWrapper);
			if(dim.biased) {
				kernel2->inout(gradBiasWrapper);
			}
		#endif

    if (dim.needToNormalize){
		kernel2->input(dim.translate);
		kernel2->input(dim.scale);
    }
	}
//	float *temp4= new float[dim.filtersSize];
//		CLWrapper *tempWrapper=cl->wrap(dim.filtersSize, temp4);
//		tempWrapper->createOnDevice();
//		weightWrapper->copyTo(tempWrapper);





	#if MEASURE_BACKWARD_PROP == 1
		stopTimer1 = clock();
		LOGI("----------------------extra 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		LOGI("----------------------max size %d ",kernel2->get_kernel_work_group_size()) ;

		startTimer1=clock();
	#endif
    kernel2->run_1d(globalSize, workgroupsize);

    //cl->finish();
//    tempWrapper->copyToHost();
//    for(int i=0;i<20;i++)
//    	LOGI("test[%d]=%f",i,temp4[i]);
//    delete[] temp4;
	#if MEASURE_BACKWARD_PROP == 1
		stopTimer1 = clock();
		LOGI("----------------------calculate weight 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

		startTimer1=clock();
		if (dim.test){
			gradWeightsWrapper->copyToHost();
			gradBiasWrapper->copyToHost();

			float error=0.0f;
			for (int i= 0; i < dim.filtersSize;i++)
				error+=abs(temp[i]-grad[i]);
			LOGI("calculate weight) error backprop_floats %f",error);
			error=0.0f;
			for (int i= 0; i <(dim.filtersSize/(dim.filterSizeSquared*dim.inputPlanes));i++)
				error+=abs(temp2[i]-gradBias[i]);

			LOGI("error bias %f",error);
			StatefulTimer::instance()->timeCheck("BackpropWeightsNaive end");
			delete[] temp;
			delete[] temp2;
		}
		stopTimer1 = clock();
        LOGI("----------------------extra bis 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
	#endif
}

float BackpropWeightsNaive::learningRateToMultiplier(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackpropWeightsNaive.cpp: learningRateToMultiplier");
#endif


    return 1.0f;
}

BackpropWeightsNaive::BackpropWeightsNaive(EasyCL *cl, LayerDimensions dim) :
        learningMultiplier(learningRateToMultiplier(dim.batchsize))
            {
	this->cl=cl;
	this->dim=dim;
    std::string options = dim.buildOptionsString();
    string imageV_with_possible_normalization="";
    string temp="";
    if (dim.needToNormalize){
    		 imageV_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 }else
    		 imageV_with_possible_normalization="upstreamResult";

    if (dim.test){
	   temp=
		"// Copyright Hugh Perkins 2014,2015 hughperkins at gmail\n"
		"//\n"
		"// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
		"// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
		"// obtain one at http://mozilla.org/MPL/2.0/.\n"
		"\n"
		"// expected defines:\n"
		"// BIASED (or not)\n"
		"\n"
		"// globalId: [outPlane][inputPlane][filterRow][filterCol]\n"
		"// per-thread iteration: [n][outputRow][outputCol]\n"
		"void kernel backprop_floats(const float learningRateMultiplier,\n"
		"        const int batchSize,\n"
		"         global const float *gradOutput, global const float *images,\n"
		"        global float *gradWeights\n"
		"        #ifdef BIASED\n"
		"            , global float *gradBiasWeights\n"
		"        #endif\n"
		" ) {\n"
		"    int globalId = get_global_id(0);\n"
		"    if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize) {\n"
		"        return;\n"
		"    }\n"
		"\n"
		"    int IntraFilterOffset = globalId % gFilterSizeSquared;\n"
		"    int filterRow = IntraFilterOffset / gFilterSize;\n"
		"    int filterCol = IntraFilterOffset % gFilterSize;\n"
		"\n"
		"    int filter2Id = globalId / gFilterSizeSquared;\n"
		"    int outPlane = filter2Id / gInputPlanes;\n"
		"    int upstreamPlane = filter2Id % gInputPlanes;\n"
		"\n"
		"    float thiswchange = 0;\n"
		"    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
		"    //       aggregate over:  [outRow][outCol][n]\n"
		"#ifdef BIASED\n"
		"    float thisbiaschange = 0;\n"
		"#endif\n"
		"    for (int n = 0; n < batchSize; n++) {\n"
		"        for (int outRow = 0; outRow < gOutputSize; outRow++) {\n"
		"            int upstreamRow = outRow - gMargin + filterRow;\n"
		"            for (int outCol = 0; outCol < gOutputSize; outCol++) {\n"
		"                int upstreamCol = outCol - gMargin + filterCol;\n"
		"                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize\n"
		"                    && upstreamCol < gInputSize;\n"
		"                if (proceed) {\n"
		"                    int resultIndex = (( n * gNumFilters\n"
		"                              + outPlane) * gOutputSize\n"
		"                              + outRow) * gOutputSize\n"
		"                              + outCol;\n"
		"                    float error = gradOutput[resultIndex];\n"
		"                    int upstreamDataIndex = (( n * gInputPlanes\n"
		"                                     + upstreamPlane) * gInputSize\n"
		"                                     + upstreamRow) * gInputSize\n"
		"                                     + upstreamCol;\n"
		"                    float upstreamResult = images[upstreamDataIndex];\n"
		"                    float thisimagethiswchange = "+imageV_with_possible_normalization+" * error;\n"
		"                    thiswchange += thisimagethiswchange;\n"
		"    #ifdef BIASED\n"
		"                    thisbiaschange += error;\n"
		"    #endif\n"
		"                }\n"
		"            }\n"
		"        }\n"
		"    }\n"
		"    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
		"    //       aggregate over:  [outRow][outCol][n]\n"
		"    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n"
		"#ifdef BIASED\n"
		"    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;\n"
		"    if (writeBias) {\n"
		"        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n"
		"    }\n"
		"#endif\n"
		"}\n"
		"\n"
		"\n"
		"\n"
		"";

    }

    string kernelSource2 =
        		            "void kernel backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
        		            "         global const float *gradOutput, global const float *images,\n"
        		            "        global float *gradWeights\n"
        		            "        {{gBiasDeclaration}}\n"
        		            " ) {\n"
        		            "    int globalId = get_global_id(0);\n"
        		            "\n"
        		            "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
        		            "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
        		            "\n"
        		            "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
        		            "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
        		            "\n"
        		            "    float thiswchange = 0;\n"
        		            "{{gBiasInit}}"
    						"    #pragma unroll\n"
        		            "    for (int n = 0; n < {{gBatch}}; n++) {\n"
        		            "        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
        		            "            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
        		            "            for (int outCol = 0; outCol < {{gOutputSize}}; outCol++) {\n"
        		            "                int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
        		            "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
        		            "                    && upstreamCol < {{gInputSize}};\n"
        		            "                if (proceed) {\n"
        		            "                    int resultIndex = (( n * {{gNumFilters}}\n"
        		            "                              + outPlane) * {{gOutputSize}}\n"
        		            "                              + outRow) * {{gOutputSize}}\n"
        		            "                              + outCol;\n"
        		            "                    float error = gradOutput[resultIndex];\n"
        		            "                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
        		            "                                     + upstreamPlane) * {{gInputSize}}\n"
        		            "                                     + upstreamRow) * {{gInputSize}}\n"
        		            "                                     + upstreamCol;\n"
        		            "                    float upstreamResult = images[upstreamDataIndex];\n"
        		            "                    float thisimagethiswchange = upstreamResult * error;\n"
        		            "                    thiswchange += thisimagethiswchange;\n"
        					"{{gBiasComputation}}"
        		            "                }\n"
        		            "            }\n"
        		            "        }\n"
        		            "    }\n"
        		            "    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n"
        		            "{{gBiasUpdate}}"
        		            "}\n"
        		            "";
    if (dim.test){
    	const char * kernelSource =  temp.c_str();
		string operation="BackpropWeightsNaive"+std::to_string(dim.numFilters);
		kernel = cl->buildKernelFromString(operation, kernelSource, "backprop_floats", options, "cl/backpropweights.cl");
	}
    buildKernelBackward(kernelSource2);
}

void BackpropWeightsNaive::buildKernelBackward( string kernelSource) {

	setup=false;
    TemplatedKernel builder(cl);

       setupBuilderBackward(&builder);

        //string identifier2="BackpropWeightsNaive2"+std::to_string(dim.numFilters);

    	string identifier2="BackpropWeightsNaive2";
    		 identifier2=identifier2+"nbFilter=";
    		 identifier2=identifier2+std::to_string(dim.numFilters);
    		 identifier2=identifier2+"_InputSize="+std::to_string(dim.inputSize);
    		 identifier2=identifier2+"_batchsize="+std::to_string(dim.batchsize);
    		 identifier2=identifier2+"_OutputSize="+std::to_string(dim.outputSize);
    		 identifier2=identifier2+"_conv="+BoolToString(dim.isConv);
    		 identifier2=identifier2+"_normalize="+BoolToString(dim.needToNormalize);
    		 identifier2=identifier2+"_maxpool="+BoolToString(dim.useMaxPooling);

        if ((dim.filterSize==1)&&(dim.outputSize==1)&&(dim.padZeros == false)){
        	kernelSource=
        			            "void kernel {{gHintCompiler}} backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
        			            "         global const float *gradOutput, global const float *images\n"
        			            "{{gdeclareGradWeight}}"
        			            "        {{gBiasDeclaration}}\n"
        			            " ) {\n"
        			            "    int globalId = get_global_id(0);\n"
        			            "\n"
        			            "\n"
        			            "    int filter2Id = globalId;\n"
        			            "    int outPlane = filter2Id / {{gInputPlanes}};\n"
        			            "    int upstreamPlane = filter2Id % {{gInputPlanes}};\n"
        			            "\n"
        			            "    float thiswchange = 0;\n"
        			            "{{gBiasInit}}"
        						"    #pragma unroll\n"
        			            "    for (int n = 0; n < {{gBatch}}; n++) {\n"
        			            "       float error = gradOutput[( n * {{gNumFilters}}+ outPlane)];\n"
        			            "       float upstreamResult = images[( n * {{gInputPlanes}}+ upstreamPlane)];\n"
        			            "       float thisimagethiswchange = upstreamResult * error;\n"
        			            "       thiswchange += thisimagethiswchange;\n"
        						"{{gBiasComputation}}"
        			            "    }\n"
        			            "    {{gradCompute}}"
								"    {{updateRule}}"
        			            "{{gBiasUpdate}}"
        			            "}\n"
        			            "\n"
        			            "\n"
        			            "\n"
        			            "";
         }else{
        	 if ((dim.outputSize==1)&&(dim.padZeros == false))
            	 kernelSource=
            			"void kernel {{gHintCompiler}} backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
     		            "         global const float *gradOutput, global const float *images\n"
     		            "{{gdeclareGradWeight}}"
     		            "        {{gBiasDeclaration}}\n"
     		            " ) {\n"
     		            "    int globalId = get_global_id(0);\n"
     		            "\n"
     		            "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
     		            "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
     		            "\n"
     		            "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
     		            "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
     		            "\n"
     		            "    float thiswchange = 0;\n"
     		            "{{gBiasInit}}"
            			"    #pragma unroll\n"
     		            "    for (int n = 0; n < {{gBatch}}; n++) {\n"
     		            "            int upstreamRow = filterRow;\n"
     		            "                int upstreamCol = filterCol;\n"
     		            "                    float error = gradOutput[( n * {{gNumFilters}}+ outPlane)];\n"
     		            "                    float upstreamResult = images[((( n * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol)];\n"
     		            "                    float thisimagethiswchange = upstreamResult * error;\n"
     		            "                    thiswchange += thisimagethiswchange;\n"
     					"{{gBiasComputation}}"
     		            "    }\n"
     		            "    {{gradCompute}}"
						"    {{updateRule}}"
     		            "{{gBiasUpdate}}"
     		            "}\n"
     		            "";

         }
        string declareGradWeightString = ",global float *gradWeights\n";
		 string gradComputeString = "gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n";

if (dim.isConv){
	gradComputeString="gradWeights[ globalId ] = learningRateMultiplier * shareArray[pos*{{gBatch}}];\n";
	//LOGI("backward norm %d",dim.needToNormalize);

	 string imageV_with_possible_normalization= "";
	 if (dim.needToNormalize){
		 imageV_with_possible_normalization="(imageV+translate)*scale";//+"+to_string(dim.translate)+")*"+to_string(dim.scale);//
	 }else
		 imageV_with_possible_normalization="imageV";
	 string decaration_var_with_possible_normalization= "";
	 	 if (dim.needToNormalize)
	 		decaration_var_with_possible_normalization=" ,float translate, float scale";

	int remainer= dim.outputSize%4;
	int divider = dim.outputSize/4;
	string remainerString="";
	string remainerString2="";
	string remainerString3="";
	string addVariable="";
	if (remainer!=0){
			addVariable="    float4 thiswchangeV2= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"{{gBiasInit2}}";
			remainerString=
				"int outCol="+to_string(divider*4)+";\n"
				"       int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
				"       float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
				"       float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
				"       selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
				"       selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
				"       selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
				"       selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
				"       float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
				"       float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
				"       thiswchangeV2+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
				"{{gBiasComputationV2}}";

			switch(remainer) {
				case 1 : {
					remainerString2="+dot((float4)(1.0f,0.0f,0.0f,0.0f),thiswchangeV2)";
					remainerString3="+dot((float4)(1.0f,0.0f,0.0f,0.0f),thisbiaschangeV2)";
					break;
				}
				case 2 : {
					remainerString2="+dot((float4)(1.0f,1.0f,0.0f,0.0f),thiswchangeV2)";
					remainerString3="+dot((float4)(1.0f,1.0f,0.0f,0.0f),thisbiaschangeV2)";
					break;
				}
				case 3 : {
					remainerString2="+dot((float4)(1.0f,1.0f,1.0f,0.0f),thiswchangeV2)";
					remainerString3="+dot((float4)(1.0f,1.0f,1.0f,0.0f),thisbiaschangeV2)";
					break;
				}
			}
		}


	 kernelSource =
		"void kernel {{gHintCompiler}}backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
		"         global const float *gradOutput, global const float *images\n"
		"{{gdeclareGradWeight}}"
		"        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		" ) {\n"
		"    __local float shareArray[{{gBatch}}];\n"
		"    __local float shareArray2[{{gBatch}}];\n"
		"    int local_index = get_local_id(0);\n"
		"    int cpt=local_index%{{gBatch}};\n"
		"    int pos=local_index/{{gBatch}};\n"
		"    int globalId0 = get_global_id(0);\n"
		"    int globalId = get_global_id(0)/({{gBatch}});\n"
		"\n"
		"    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		"    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		"\n"
		"    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		"    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		"\n"
		"    float thiswchange = 0;\n"
		"    float4 thiswchangeV= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"{{gBiasInit}}"
		"    "+addVariable+"\n"
		"    #pragma unroll\n"
		"    for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		"       int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		"       #pragma unroll\n"
		"       for (int outCol = 0; outCol < "+to_string(divider*4)+"; outCol=outCol+4) {\n"
		"           int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		"           float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
		"           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
		"           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
		"           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
		"           selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
		"           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
		"           float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
		"           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
		"{{gBiasComputationV}}"
		"       }\n"
		"       "+remainerString +"\n"
		"    }\n"
		"    shareArray[pos*{{gBatch}}+cpt]=dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV)"+remainerString2 +";\n"
		"    shareArray2[pos*{{gBatch}}+cpt]=dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +";\n"
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    for(unsigned int s = {{gHalfBatch}}; s > 0; s >>= 1){\n"
		"      if(cpt < s){\n"
		"        shareArray[pos*{{gBatch}}+cpt] += shareArray[pos*{{gBatch}}+cpt+ s];\n"
		"        shareArray2[pos*{{gBatch}}+cpt] += shareArray2[pos*{{gBatch}}+cpt + s];\n"
		"      }\n"
		"      barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    }\n"
		"    if(cpt == 0){\n"
		"	   {{gradCompute}}"
		"      {{updateRule}}"
		"{{gBiasUpdate}}"
		"}\n"
		"}\n"
		"";


}

#if MEASURE_BACKWARD_PROP==1
		(&builder)->set("gdeclareGradWeight", declareGradWeightString);
#endif
#if MEASURE_BACKWARD_PROP==0
		(&builder)->set("gdeclareGradWeight", "");
#endif


#if MEASURE_BACKWARD_PROP==1
		(&builder)->set("gradCompute", gradComputeString);
#endif
#if MEASURE_BACKWARD_PROP==0
		(&builder)->set("gradCompute", "");
#endif

        this->kernel2 = builder.buildKernel(
           		identifier2,
               "backprop_floats",
               kernelSource.c_str(),
               "backprop_floats",
               false
        );
    }

void BackpropWeightsNaive::setHintCompiler(int batchSize, TemplatedKernel *builder){
	int possibleGlobalSize = batchSize*dim.filtersSize;
	int possibleWorkgroupsize =  batchSize;//kernel2->get_kernel_work_group_size();//cl->getMaxWorkgroupSize();

	possibleGlobalSize = ((possibleGlobalSize + possibleWorkgroupsize - 1) / possibleWorkgroupsize) * possibleWorkgroupsize;

	string hintCompilerString="__attribute__((vec_type_hint(";
	if (dim.isConv)
		hintCompilerString+="float4";
	else{
		hintCompilerString+="float";
	}

	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);
}
void BackpropWeightsNaive::setupBuilderBackward(TemplatedKernel *builder) {

	string updateWeight=" ";
	string updateBiasWeights=" ";
	string defineVariableUpdates=" ";
	string updateBiasWeight=" ";

	setupUpdateWeight(updateWeight,updateBiasWeights, defineVariableUpdates,updateBiasWeight);

	builder->set("updateVariable",defineVariableUpdates);
	if (dim.isConv){
		updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * shareArray[pos*{{gBatch}}];\n";
	}else
		updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * thiswchange;\n";

	builder->set("updateRule",updateWeight);

	setHintCompiler(dim.batchsize, builder);
	builder->set("gHalfBatch",dim.batchsize>>1);
	builder->set("gBatch",dim.batchsize);
	builder->set("gInputSizeSquared",square(dim.inputSize));
	builder->set("gInputSize",dim.inputSize);
	builder->set("gInputPlanes",dim.inputPlanes);
	builder->set("gMargin",dim.padZeros ? dim.filterSize >> 1 : 0);
	builder->set("gOutputSize",dim.outputSize);
	builder->set("gFilterSize",dim.filterSize);
	builder->set("gFilterSizeSquared",dim.filterSize*dim.filterSize);
	builder->set("gNumFilters",dim.numFilters);

	string gradBiasComputeString="";

	if (dim.biased){
		#if MEASURE_BACKWARD_PROP==1
			builder->set("gBiasDeclaration",", global float *gradBiasWeights");
		#endif
		#if MEASURE_BACKWARD_PROP==0
			builder->set("gBiasDeclaration","");
		#endif

		//builder->set("gBiasDeclaration",", global float *gradBiasWeights");
		builder->set("gBiasInit","    float thisbiaschange = 0;\n");
		string biasUpdateString=
							"    bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
							"    if (writeBias) {\n"
							"{{gradBiasCompute}}"
							"        {{updateBiasWeights}}"
							"    }\n";
		if (dim.isConv){
			builder->set("gBiasInit","    float4 thisbiaschangeV = (float4)(0.0f,0.0f,0.0f,0.0f);\n");
			builder->set("gBiasComputationV","           thisbiaschangeV += errorV;\n");
			builder->set("gBiasInit2","    float4 thisbiaschangeV2 = (float4)(0.0f,0.0f,0.0f,0.0f);\n");
			builder->set("gBiasComputationV2","       thisbiaschangeV2 += errorV;\n");
			//builder->set("gBiasComputation","           thisbiaschange += error;\n");
			biasUpdateString=
									"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "      if (writeBias) {\n"
						            "{{gradBiasCompute}}"
									"          {{updateBiasWeights}}"
						            "      }\n";

			gradBiasComputeString="        gradBiasWeights[outPlane] = learningRateMultiplier * shareArray2[pos*{{gBatch}}];\n";
		}else{
			builder->set("gBiasComputation","                    thisbiaschange += error;\n");
			if ((dim.filterSize==1)&&(dim.outputSize==1)&&(dim.padZeros == false))
				biasUpdateString=
								"    if (upstreamPlane == 0) {\n"
								"{{gradBiasCompute}}"
								"        {{updateBiasWeights}}"
								"    }\n";
				gradBiasComputeString="        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n";
		}
		builder->set("gBiasUpdate",biasUpdateString);
		builder->set("updateBiasWeights",updateBiasWeight);
	}

	#if MEASURE_BACKWARD_PROP==1
		builder->set("gradBiasCompute", gradBiasComputeString);
	#endif
	#if MEASURE_BACKWARD_PROP==0
		builder->set("gradBiasCompute", "");
	#endif

}

void BackpropWeightsNaive::setupUpdateWeight(string &updateWeight,string &updateBiasWeights, string &defineVariableUpdates, string & updateBiasWeight){

	if (1){//only  SGD

		if (dim.isConv){
			updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * shareArray[pos*{{gBatch}}];\n";
		}else
			updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * thiswchange;\n";

		if (dim.biased){
			defineVariableUpdates="const float momentum, const float learning_rate, global float* weight, global float * pastTimeStepVector, global float* bias, global float * pastTimeStepBiasVector,";
			if (dim.isConv){
				updateBiasWeight="bias[ outPlane ]=bias[ outPlane ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * shareArray2[pos*{{gBatch}}];\n";
			}else
				updateBiasWeight="bias[ outPlane ]=bias[ outPlane ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * thisbiaschange;\n";

		}else
			defineVariableUpdates="const float momentum, const float learning_rate, global float* weight, global float * pastTimeStepVector,";
	}
}
