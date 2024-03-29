// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "../../EasyCL/util/StatefulTimer.h"

#include "../layer/LayerMaker.h"
#include "../fc/FullyConnectedLayer.h"
#include "SoftMaxLayer.h"

using namespace std;

#define TEST_SOFTMAX 0
#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayer::SoftMaxLayer(Layer *previousLayer, SoftMaxMaker *maker) :
    LossLayer(previousLayer, maker),
        perPlane(maker->_perPlane),
        ptrcl(maker->cl),
        imageSize(previousLayer->getOutputSize()),
        numPlanes(previousLayer->getOutputPlanes()),
        imageSizeSquared(previousLayer->getOutputSize() * previousLayer->getOutputSize()),
        output(0),
        gradInput(0),
        allocatedSize(0),
        batchSize(0),
        kernel(0),
        setup1(false)
         {
	if (dynamic_cast<FullyConnectedLayer*>(previousLayer))
	    	batchsize=((dynamic_cast<FullyConnectedLayer*>(previousLayer))->convolutionalLayer->dim.batchsize);
	else{
		LOGI("error SoftMaxLayer");
		batchsize=128;//error
	}

	setup=false;

}
VIRTUAL SoftMaxLayer::~SoftMaxLayer() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: ~SoftMaxLayer");
#endif


    if(gradInput != 0) {
        delete[] gradInput;
    }
    if(output != 0) {
        delete[] output;
    }
    if (kernel!= 0) {
    	delete kernel;
    }

	//delete outputWrapper;
	delete[] lossFloat;
	delete lossWrapper;
	delete[] nbRight;
	delete 	nbRightWrapper;
	delete[] labelData;
	delete labelWrapper;
	delete gradInputWrapper;
}
VIRTUAL std::string SoftMaxLayer::getClassName() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: string SoftMaxLayer::getClassName");
#endif


    return "SoftMaxLayer";
}
VIRTUAL float *SoftMaxLayer::getOutput() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getOutput");
#endif


    return output;
}
VIRTUAL float *SoftMaxLayer::getGradInput() {
#if 1//DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getGradInput");
#endif


    return gradInput;
}
VIRTUAL void SoftMaxLayer::setBatchSize(int batchSize) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: setBatchSize");
#endif

    this->batchSize = batchSize;
    if(batchSize <= this->allocatedSize) {
        return;
    }
//    if(output != 0) {
//        delete[] output;
//    }
	if (not setup1){
		gradInput = new float[ previousLayer-> getOutputNumElements() ];
		allocatedSize = batchSize;
		setup1=true;
	}

}
VIRTUAL int SoftMaxLayer::getBatchSize() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getBatchSize");
#endif


    return this->batchSize;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLossFromLabels(int const *labels) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: calcLossFromLabels %d",perPlane);
#endif
	if (not setup){
		lossFloat=new float[1];
		lossWrapper = ptrcl->wrap(1,lossFloat);
		lossWrapper->createOnDevice();
		lossWrapper->copyToDevice();
		nbRight=new int[1];
		nbRightWrapper = ptrcl->wrap(1,nbRight);
		nbRightWrapper->createOnDevice();
		nbRightWrapper->copyToDevice();
		labelData =new int[batchSize];
		labelWrapper = ptrcl->wrap(batchSize,labelData);
		labelWrapper->createOnDevice();
		gradInputWrapper=ptrcl->wrap(previousLayer-> getOutputNumElements(),gradInput);
		gradInputWrapper->createOnDevice();
		globalSize = batchSize;
		workgroupsize = std::min(globalSize, ptrcl->getMaxWorkgroupSize());
		createKernel();
		setup=true;
		inputWrapper = previousLayer->getOutputWrapper();

		kernel->input(inputWrapper);
		//kernel->output(outputWrapper);
		kernel->input(labelWrapper);
		kernel->output(lossWrapper);
		kernel->output(nbRightWrapper);
		kernel->output(gradInputWrapper);
	}


	inputWrapper = previousLayer->getOutputWrapper();
	runSoftmax_forward( inputWrapper/*,  outputWrapper*/,labels) ;

    return lossFloat[0];
}

VIRTUAL bool SoftMaxLayer::providesGradInputWrapper() const {
#if DEEPCL_VERBOSE == 1
	LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: providesGradInputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *SoftMaxLayer::getGradInputWrapper() {
#if DEEPCL_VERBOSE == 1
	LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapper;
}

// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLoss(float const *expectedValues) {
#if 1// DEEPCL_VERBOSE == 1
LOGI( "-------------ERROR --->  DeepCL/src/loss/SoftMaxLayer.cpp: calcLoss");
#endif


    StatefulTimer::timeCheck("start SoftMaxLayer calcLoss");
    float loss = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    if(expectedValues[ imageOffset + i ] != 0) {
                        float thisloss = - expectedValues[ imageOffset + i ] * log(output[ imageOffset + i ]);
                        loss += thisloss;
                    }
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                float thisloss = - expectedValues[imageOffset + plane] * log(output[imageOffset + plane]);
                loss += thisloss;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLoss");
    return loss;
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcGradInputFromLabels(int const *labels) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: calcGradInputFromLabels");
#endif

//olivier:this is the CPU version. Everything is already done on the GPU (runSoftmax_forward)
//	float *temp =new float [batchSize*numPlanes];
//	for (int i =0; i<batchSize*numPlanes;i++)
//		temp[i]=gradInput[i];
//
////    cout << "softmaxlayer::calcerrors" << endl;
//    StatefulTimer::timeCheck("start SoftMaxLayer calcGradInputfromlabels");
//    if(perPlane) {
//        for(int n = 0; n < batchSize; n++) {
//            for(int plane = 0; plane < numPlanes; plane++) {
//                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
//                int label = labels[n * numPlanes + plane];
//                for(int i = 0; i < imageSizeSquared; i++) {
//                    gradInput[imageOffset + i] = output[imageOffset + i];
//                }
//                gradInput[imageOffset + label] -= 1;
//            }
//        }
//    } else {
//        // force imagesize of 1 for now
//        if(imageSize != 1) {
//            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
//        }
//        for(int n = 0; n < batchSize; n++) {
//            int imageOffset = n * numPlanes * imageSizeSquared;
//            int label = labels[n];
//            for(int plane = 0; plane < numPlanes; plane++) {
//                gradInput[imageOffset + plane] = output[imageOffset + plane];
//            }
//            if(label >= numPlanes) {
//                throw runtime_error("Label " + toString(label) + " exceeds number of softmax planes " + toString(numPlanes) );
//            } else if(label < 0) {
//                throw runtime_error("Label " + toString(label) + " negative");
//            }
//            gradInput[imageOffset + label] -= 1;
//        }
//    }
//    StatefulTimer::timeCheck("end SoftMaxLayer calcGradInputfromlabels");
//    float error =0.0f;
//	for (int i =0; i<batchSize*numPlanes;i++)
//		error=abs(temp[i]-gradInput[i]);
//
//delete[] temp;
//	  LOGI("error=%f",error);
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcGradInput(float const *expectedValues) {
#if 1 //DEEPCL_VERBOSE == 1
LOGI( "-------------ERROR --->  DeepCL/src/loss/SoftMaxLayer.cpp: calcGradInput");
#endif


//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcGradInput");
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    int resultIndex = imageOffset + i;
                    gradInput[resultIndex] = output[resultIndex] - expectedValues[resultIndex];
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                int resultIndex = imageOffset + plane;
                gradInput[resultIndex] = output[resultIndex] - expectedValues[resultIndex];
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcGradInput");
}
VIRTUAL int SoftMaxLayer::getNumLabelsPerExample() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getNumLabelsPerExample");
#endif


    if(perPlane) {
        return numPlanes;
    } else {
        return imageSizeSquared;
    }
}
VIRTUAL int SoftMaxLayer::getPersistSize(int version) const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL int SoftMaxLayer::calcNumRightFromLabels(int const*labels) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: calcNumRightFromLabels");
#endif
//
//int numRight = 0;
//    if(perPlane) {
//        for(int n = 0; n < batchSize; n++) {
//            for(int plane = 0; plane < numPlanes; plane++) {
//                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
//                int label = labels[n * numPlanes + plane];
//                float thisMax = output[imageOffset + 0];
//                int iMax = 0;
//                for(int i = 1; i < imageSizeSquared; i++) {
//                    if(output[imageOffset + i] > thisMax) {
//                        thisMax = output[imageOffset + i];
//                        iMax = i;
//                    }
//                }
//                if(label == iMax) {
////                    cout << "n " << n << " plane " << plane << " label " << label << endl;
//                    numRight++;
//                }
//            }
//        }
//    } else {
//        // force imagesize of 1 for now
//        if(imageSize != 1) {
//            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
//        }
//        for(int n = 0; n < batchSize; n++) {
//            int imageOffset = n * numPlanes * imageSizeSquared;
//            int label = labels[n];
//            float thisMax = output[imageOffset + 0];
//            int iMax = 0;
//            for(int i = 1; i < numPlanes; i++) {
//                if(output[imageOffset + i] > thisMax) {
//                    thisMax = output[imageOffset + i];
//                    iMax = i;
//                }
//            }
//            if(label == iMax) {
//                numRight++;
//            }
//        }
//    }
//
//    LOGI("%d vs %d",nbRight[0],numRight);


    return nbRight[0];
}
// for forward, we just need to apply the softmax activation. "just" :-P
VIRTUAL void SoftMaxLayer::forward() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: forward perPlane %d %d %d",perPlane,imageSizeSquared,numPlanes);
#endif

// olivier the operation is performed by calcLossFromLabels
//float *input = previousLayer->getOutput(); // just retrieve as host-side array for now
//   if(perPlane) {
//       for(int n = 0; n < batchSize; n++) {
//           for(int plane = 0; plane < numPlanes; plane++) {
//               int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
//               float maxValue = input[imageOffset + 0];
//               for(int i = 1; i < imageSizeSquared; i++) {
//                   maxValue = std::max(maxValue, input[imageOffset + i]);
//               }
//               float denominator = 0;
//               for(int i = 0; i < imageSizeSquared; i++) {
//                   denominator += exp(input[imageOffset + i] - maxValue);
//               }
//               for(int i = 0; i < imageSizeSquared; i++) {
//                   output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
//               }
//           }
//       }
//   } else {
//       // force imagesize of 1 for now
//       if(imageSize != 1) {
//           throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
//       }
//       for(int n = 0; n < batchSize; n++) {
//           int imageOffset = n * numPlanes * imageSizeSquared;
//           // first get the max
//           float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
//           for(int plane = 1; plane < numPlanes; plane++) {
//               maxValue = std::max(maxValue, input[imageOffset + plane]);
//           }
//           // calculate sum, under this max
//           float denominator = 0;
//           for(int plane = 0; plane < numPlanes; plane++) {
//               denominator += exp(input[imageOffset + plane] - maxValue);
//           }
//           // now calc the softmaxes:
//           for(int plane = 0; plane < numPlanes; plane++) {
//               output[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
//           }
//       }
//   }
//   for (int i =0; i<15;i++)
//	   LOGI("output[%d]=%f",i,output[i]);



    StatefulTimer::timeCheck("end SoftMaxLayer forward");
}
VIRTUAL void SoftMaxLayer::getLabels(int *labels) { // need to allocate labels array first, and have called 'forward' first
#if 1 //DEEPCL_VERBOSE == 1
LOGI( "-------------ERROR --->  DeepCL/src/loss/SoftMaxLayer.cpp: getLabels");
#endif
    if(perPlane) {
        throw std::runtime_error("getLabels doesnt work with 'perPlane' option currently, though it wouldnt be hard to add, so ask if you need");
    }
    if(imageSize != 1) {
        throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
    }
    for(int n = 0; n < batchSize; n++) {
        float *outputStack = output + n * numPlanes;
        float highestProb = outputStack[0];
        int bestPlane = 0;
        for(int plane = 1; plane < numPlanes; plane++) {
            if(outputStack[plane] > highestProb) {
                bestPlane = plane;
                highestProb = outputStack[plane];
            }
        }
        labels[n] = bestPlane;
    }
}
// this seems to be handled by calcGradInput? So, just to a nop?
// (cos this layer kind of combines loss layer and a 'normal' propagation layer)
// certainly, we dont have any weights to update, and we already handled error
// propagation in 'calcGradInput' method above
//VIRTUAL void SoftMaxLayer::backward(float learningRate) {
//    cout << "softmaxlayer::backproperrors" << endl;
    // nop, do nothing :-)
//}
VIRTUAL std::string SoftMaxLayer::asString() const {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SoftMaxLayer.cpp: string SoftMaxLayer::asString");
#endif


    return "SoftMaxLayer{ perPlane=" + toString(perPlane) + " numPlanes=" + toString(numPlanes)
        + " imageSize=" + toString(imageSize) + " }";
}

void SoftMaxLayer::createKernel(){

	buildKernelSoftmax_forward();

}

void SoftMaxLayer::buildKernelSoftmax_forward() {
    TemplatedKernel builder(ptrcl);


        setupBuilderSoftmax_forward(&builder);


        string identifier2="softmax_forward"+std::to_string(numPlanes);
        string test=getKernelTemplateSoftmax_forward();
        this->kernel = builder.buildKernel(
           		identifier2,
               "softmax",
               test,
               "softmax_forward",
               false
        );
    }

void SoftMaxLayer::runSoftmax_forward(    CLWrapper *inputWrapper/*,CLWrapper *outputWrapper*/,int const *labels) {

#if TEST_SOFTMAX==1
clock_t startTimer1, stopTimer1,startTimer2, stopTimer2,startTimer3, stopTimer3;
	startTimer2=clock();
#endif
//	for(int i=0;i<batchSize;i++)
//		labelData[i]=labels[i];

	labelWrapper->copyToDevice(labels);



#if TEST_SOFTMAX==1
	LOGI("globalSize %d workgroupsize %d",globalSize,workgroupsize);
	startTimer1=clock();
#endif

	kernel->run_1d(globalSize, workgroupsize);
	ptrcl->finish();
	#if TEST_SOFTMAX==1
	stopTimer1 = clock();
	startTimer3=clock();
	#endif
//	nbRightWrapper->copyToHost();
//	lossWrapper->copyToHost();
	#if TEST_SOFTMAX==1

		stopTimer3 = clock();
		LOGI("read  took %g ms\n\n", 1000.0* (double)(stopTimer3 - startTimer3)/(double)CLOCKS_PER_SEC) ;
		LOGI("Softmax  took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		LOGI("loss %f", lossFloat[0]) ;

		stopTimer2 = clock();
		LOGI("Softmax  took %g ms\n\n", 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC) ;
	#endif
	//gradInputWrapper->copyToHost();

}

CLWrapper * SoftMaxLayer::getLossWrapper(){
	return lossWrapper;
}

CLWrapper * SoftMaxLayer::getNbRightWrapper(){
	return nbRightWrapper;
}

void SoftMaxLayer::setupBuilderSoftmax_forward(TemplatedKernel *builder) {

	int possibleGlobalSize =batchsize;// batchSize;//crash because batchSize value = 0
	//LOGI("batch %d",batchSize);
	int possibleWorkgroupsize = std::min(possibleGlobalSize, ptrcl->getMaxWorkgroupSize());
	string hintCompilerString="__attribute__((vec_type_hint(";
	hintCompilerString+="float";
	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);

	builder->set("numPlanes",numPlanes);
	builder->set("batchSize",possibleGlobalSize);

}

STATIC std::string SoftMaxLayer::getKernelTemplateSoftmax_forward() {

	const char * kernelSource =
			"void kernel {{gHintCompiler}} softmax_forward(const __global float * restrict input , __constant int * labels , __global float * loss , __global int * nbRight, __global float *gradInput\n"
			"){\n"
					"  local float localTemp[{{batchSize}}];\n"
					"  local bool localBool[{{batchSize}}];\n"
			        "  const int globalId = get_global_id(0);\n"
					"  int labelIdx = labels[globalId];\n"
					"  int imageOffset =globalId*{{numPlanes}};\n"
					"  float denominator=0;\n"
					"  float maxValue = input[imageOffset];\n"
					"  #pragma unroll\n"
					"  for(int plane=0 ; plane<{{numPlanes}} ; plane++)\n"
		            "    denominator+=exp(input[imageOffset+plane]-maxValue);\n"
					"  int selectorID= 0;\n"
					"  float maxValue2 = exp(input[imageOffset]-maxValue)/denominator;\n\n"
					"  gradInput[imageOffset]=maxValue2-(float)select(0,1,(labelIdx==0));\n\n"
					"  #pragma unroll\n"
					"  for(int plane=1;plane<{{numPlanes}};plane++)\n{"
					"    float temp=exp(input[imageOffset+plane]-maxValue)/denominator;\n"
					"    gradInput[imageOffset+plane]=temp-(float)select(0,1,(plane==labelIdx));\n"
					"    selectorID=select(selectorID,plane,(isgreater(temp,maxValue2)));\n"
					"    maxValue2=fmax(maxValue2,temp);\n"
					"  }\n"
					"  localTemp[globalId]=-log(exp(input[imageOffset+labelIdx]-maxValue)/denominator);\n\n"
					"  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n\n"
					"  if(globalId==0){\n"
					"    loss[0]=0;\n"
					"    for(int i=0;i<{{batchSize}};i++)\n"
					"      loss[0]+=localTemp[i];\n"
			        "  }\n\n\n"
					"  localBool[globalId]=select(0,1,(labelIdx==selectorID));\n"
					"  barrier(CLK_LOCAL_MEM_FENCE);\n"
					"  if(globalId==0){\n"
					"    nbRight[0]=0;\n"
					"    for(int i=0;i<{{batchSize}};i++)\n"
					"      nbRight[0]+=select(0,1,localBool[i]);"
			        "  }\n"
					"}\n"
			        "\n"
			        "";

    return kernelSource;
}

//STATIC std::string SoftMaxLayer::getKernelTemplateSoftmax_forward() {
//
//	const char * kernelSource =
//			"void kernel {{gHintCompiler}} softmax_forward(const __global float * restrict input , __global float * output , __constant int * labels , __global float * loss , __global int * nbRight, __global float *gradInput\n"
//			"){\n"
//					"  local float localTemp[{{batchSize}}];\n"
//					"  local bool localBool[{{batchSize}}];\n"
//			        "  const int globalId = get_global_id(0);\n"
//					"  int labelIdx = labels[globalId];\n"
//					"  int imageOffset =globalId*{{numPlanes}};\n"
//					"  float denominator=0;\n"
//					"  float maxValue = input[imageOffset+0];\n"
//					"  #pragma unroll\n"
//					"  for(int plane=0 ; plane<{{numPlanes}} ; plane++)\n"
//		            "    denominator+=exp(input[imageOffset+plane]-maxValue);\n"
//					"  #pragma unroll\n"
//					"  for(int plane=0;plane<{{numPlanes}};plane++)\n{"
//					"    output[imageOffset+plane]=exp(input[imageOffset+plane]-maxValue)/denominator;\n"
//					"    gradInput[imageOffset+plane]=output[imageOffset+plane]-(float)select(0,1,(plane==labelIdx));"
//					"  }\n"
//					"  localTemp[globalId]=-log(exp(input[imageOffset+labelIdx]-maxValue)/denominator);\n\n"
//					"  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);\n\n"
//					"  if(globalId==0){\n"
//					"    loss[0]=0;\n"
//					"    for(int i=0;i<{{batchSize}};i++)\n"
//					"      loss[0]+=localTemp[i];"
//			        "  }\n\n\n"
//					"  int selectorID= 0;\n"
//					"  float maxValue2 = output[imageOffset];\n\n"
//					"  #pragma unroll\n"
//					"  for(int plane=1;plane<{{numPlanes}};plane++){\n"
//					"    selectorID=select(selectorID,plane,(isgreater(output[imageOffset+plane],maxValue2)));\n"
//					"    maxValue2=fmax(maxValue2,output[imageOffset+plane]);\n"
//					"  }\n"
//					"  localBool[globalId]=select(0,1,(labelIdx==selectorID));\n"
//					"  barrier(CLK_LOCAL_MEM_FENCE);\n"
//					"  if(globalId==0){\n"
//					"    nbRight[0]=0;\n"
//					"    for(int i=0;i<{{batchSize}};i++)\n"
//					"      nbRight[0]+=select(0,1,localBool[i]);"
//			        "  }\n"
//					"}\n"
//			        "\n"
//			        "";
//
//    return kernelSource;
//}
//
//
