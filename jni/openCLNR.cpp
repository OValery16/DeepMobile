////
////  openCLNR.cpp
////  OpenCL Example1
////
////  Created by Rasmusson, Jim on 18/03/13.
////
////  Copyright (c) 2013, Sony Mobile Communications AB
////  All rights reserved.
////
////  Redistribution and use in source and binary forms, with or without
////  modification, are permitted provided that the following conditions are met:
////
////     * Redistributions of source code must retain the above copyright
////       notice, this list of conditions and the following disclaimer.
////
////     * Redistributions in binary form must reproduce the above copyright
////       notice, this list of conditions and the following disclaimer in the
////       documentation and/or other materials provided with the distribution.
////
////     * Neither the name of Sony Mobile Communications AB nor the
////       names of its contributors may be used to endorse or promote products
////       derived from this software without specific prior written permission.
////
////  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
////  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
////  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
////  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
////  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
////  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
////  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
////  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
////  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
////  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//#define __CL_ENABLE_EXCEPTIONS
//
//#include "openCLNR.h"
////#include "clew/clew.h"
//#include "DeepCL/EasyCL/EasyCL.h"
//#include "DeepCL/EasyCL/CLKernel.h"
//
//#include <DeepCL/clMathLibraries/clBLAS/src/clBLAS.h>
//#include <DeepCL/src/DeepCL.h>
//#include <DeepCL/src/conv/Forward.h>
//
//#include "DeepCL/src/util/Timer.h"
//#include "DeepCL/src/net/NeuralNet.h"
////#include "AccuracyHelper.h"
////#include "DeepCL/src/util/StatefulTimer.h"
//#include "DeepCL/src/batch/NetLearner.h"
//#include "DeepCL/src/trainers/SGD.h"
//#include "DeepCL/src/layer/Layer.h"
//#include "DeepCL/EasyCL/EasyCL.h"
//#include "DeepCL/src/net/NeuralNetMould.h"
//#include "DeepCL/src/layer/LayerMakers.h"
//#include "DeepCL/src/batch/EpochMaker.h"
//#include "DeepCL/src/batch/Batcher2.h"
//#include "DeepCL/src/clblas/ClBlasInstance.h"
//
//#include <random>
//#include <algorithm>
//#include "DeepCL/src/net/NeuralNet.h"
//#include "DeepCL/src/conv/Backward.h"
//#include "DeepCL/src/activate/ActivationFunction.h"
//#include "DeepCL/src/loss/LossLayer.h"
//#include "DeepCL/src/forcebackprop/ForceBackpropLayerMaker.h"
//#include "DeepCL/src/layer/LayerMakers.h"
//#include "DeepCL/src/net/NeuralNetMould.h"
//#include "DeepCL/src/conv/ConvolutionalLayer.h"
//#include "DeepCL/src/input/InputLayer.h"
//#include "DeepCL/src/trainers/SGD.h"
//#include "DeepCL/src/clblas/ClBlasInstance.h"
//
//#include "DeepCL/test/WeightRandomizer.h"
//
//
//#include "DeepCL/test/AccuracyHelper.h"
//#include "trainEngine/train.h"
//
//
////#define M  4
////#define N  3
////#define K  5
//
//static const clblasOrder order = clblasRowMajor;
//
//static const cl_float alpha = 1;
//
//static const clblasTranspose transA = clblasNoTrans;
////static const cl_float A[M*K] = {
////    11, 12, 13, 14, 15,
////    21, 22, 23, 24, 25,
////    31, 32, 33, 34, 35,
////    41, 42, 43, 44, 45,
////};
////static const size_t lda = K;        /* i.e. lda = K */
//
//static const clblasTranspose transB = clblasNoTrans;
////static const cl_float B[K*N] = {
////    11, 12, 13,
////    21, 22, 23,
////    31, 32, 33,
////    41, 42, 43,
////    51, 52, 53,
////};
////static const size_t ldb = N;        /* i.e. ldb = N */
//
//static const cl_float beta = 0;
//
////static cl_float C[M*N] = {
////    11, 12, 13,
////    21, 22, 23,
////    31, 32, 33,
////    41, 42, 43,
////};
////static const size_t ldc = N;        /* i.e. ldc = N */
////
////static cl_float result[M*N];
//
//static const size_t off  = 0;//1;
//static const size_t offA = 0;//K + 1;   /* K + off */
//static const size_t offB = 0;//N + 1;   /* N + off */
//static const size_t offC = 0;//N + 1;   /* N + off */
//
//
//inline std::string loadProgram(std::string input)
//{
//	std::ifstream stream(input.c_str());
//	if (!stream.is_open()) {
//		LOGE("Cannot open input file\n");
//		exit(1);
//	}
//	return std::string( std::istreambuf_iterator<char>(stream),
//						(std::istreambuf_iterator<char>()));
//}
//
//static const char *getKernel() {
//    // [[[cog
//    // import stringify
//    // stringify.stringify("source", "test/testeasycl.cl")
//    // ]]]
//    // generated using cog, from test/testeasycl.cl:
//    const char * source =
//    "kernel void test(global float *in, global float *out) {\n"
//    "    const int globalid = get_global_id(0);\n"
//    "    out[globalid] = in[globalid] + 7;\n"
//    "}\n"
//    "\n"
//    "kernel void testuchars(global unsigned char *in, global unsigned char *out) {\n"
//    "    const int globalid = get_global_id(0);\n"
//    "    out[globalid] = in[globalid] + 7;\n"
//    "}\n"
//    "\n"
//    "kernel void test_int(global int *in, global int *out) {\n"
//    "    const int globalid = get_global_id(0);\n"
//    "    out[globalid] = in[globalid] + 7;\n"
//    "}\n"
//    "\n"
//    "kernel void test_stress(global const int *in, global int *out) {\n"
//    "    const int globalid = get_global_id(0);\n"
//    "    int sum = 0;\n"
//    "    int n = 0;\n"
//    "   // make it do some work....\n"
//    "//    while(n < 1000000) {\n"
//    "    while(n < 10001) {\n"
//    "        sum = (sum + in[n % 47]) % (103070 * 7);\n"
//    "        n++;\n"
//    "    }\n"
//    "    out[globalid] = sum;\n"
//    "}\n"
//    "\n"
//    "kernel void test_read(const int one,  const int two, global int *out) {\n"
//    "    const int globalid = get_global_id(0);\n"
//    "    int sum = 0;\n"
//    "    int n = 0;\n"
//    "    while(n < 100000) {\n"
//    "        sum = (sum + one) % 1357 * two;\n"
//    "        n++;\n"
//    "    }\n"
//    "//    out[globalid+2048] = sum;\n"
//    "//    out[globalid] = sum;\n"
//    "//    out[0] = 44;\n"
//    "    out[globalid] = sum;\n"
//    "   // out[0] = globalid > out[0] ? globalid : out[0];\n"
//    "//    out[globalid] = 8827;\n"
//    "}\n"
//    "\n"
//    "";
//    // [[[end]]]
//    return source;
//}
//
//static void
//printResult(const char* str,float *result,int ldc)
//{
//    size_t i, j, nrows;
//
//    LOGI("%s:\n", str);
//
//    nrows = (sizeof(result) / sizeof(cl_float)) / ldc;
//    for (i = 0; i < 5/*nrows*/; i++) {
//        for (j = 0; j < 5/*ldc*/; j++) {
//        	LOGI("%f ", (float)result[i * ldc + j]);
//        }
//        LOGI("\n");
//    }
//}
//
//
//int normTest(){
//	size_t N2 = 1024*4;
//
//	float *X= (float*) calloc(N2, sizeof(float));
//			for(int i=0;i<N2;i++){
//				X[i]=0.25;
//			}
//
//	static const int incx2 = 1;
//	static float NRM2;
//
//
//	cl_int err;
//	    cl_platform_id platform = 0;
//	    cl_device_id device = 0;
//	    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
//	    cl_context ctx = 0;
//	    cl_command_queue queue = 0;
//	    cl_mem bufX, bufNRM2, scratchBuff;
//	    cl_event event = NULL;
//	    int ret = 0;
//		int lenX = 1 + (N2-1)*abs(incx2);
//
//	    /* Setup OpenCL environment. */
//	    err = clGetPlatformIDs(1, &platform, NULL);
//	    if (err != CL_SUCCESS) {
//	        printf( "clGetPlatformIDs() failed with %d\n", err );
//	        return 1;
//	    }
//
//	    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//	    if (err != CL_SUCCESS) {
//	        printf( "clGetDeviceIDs() failed with %d\n", err );
//	        return 1;
//	    }
//
//	    props[1] = (cl_context_properties)platform;
//	    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
//	    if (err != CL_SUCCESS) {
//	        printf( "clCreateContext() failed with %d\n", err );
//	        return 1;
//	    }
//
//	    queue = clCreateCommandQueue(ctx, device, 0, &err);
//	    if (err != CL_SUCCESS) {
//	        printf( "clCreateCommandQueue() failed with %d\n", err );
//	        clReleaseContext(ctx);
//	        return 1;
//	    }
//
//	    /* Setup clblas. */
//	    err = clblasSetup();
//	    if (err != CL_SUCCESS) {
//	        printf("clblasSetup() failed with %d\n", err);
//	        clReleaseCommandQueue(queue);
//	        clReleaseContext(ctx);
//	        return 1;
//	    }
//
//	    /* Prepare OpenCL memory objects and place vectors inside them. */
//	    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenX*sizeof(cl_float)), NULL, &err);
//	    // Allocate 1 element space for NRM2
//	    bufNRM2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (sizeof(cl_float)), NULL, &err);
//	    // Allocate minimum of N elements
//	    scratchBuff = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (2*N2*sizeof(cl_float)), NULL, &err);
//
//	    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, (lenX*sizeof(cl_float)), X, 0, NULL, NULL);
//
//	    /* Call clblas function. */
//	    err = clblasSnrm2(N2, bufNRM2, 0, bufX, 0, incx2, scratchBuff,
//	                                    1, &queue, 0, NULL, &event);
//	    if (err != CL_SUCCESS) {
//	    	LOGI("clblasSnrm2() failed with %d\n", err);
//	        ret = 1;
//	    }
//	    else {
//	        /* Wait for calculations to be finished. */
//	        err = clWaitForEvents(1, &event);
//
//	        /* Fetch results of calculations from GPU memory. */
//	        err = clEnqueueReadBuffer(queue, bufNRM2, CL_TRUE, 0, sizeof(cl_float),
//	                                    &NRM2, 0, NULL, NULL);
//	        LOGI("Result Euclidean Norm: %f\n", NRM2);
//	    }
//
//	    /* Release OpenCL memory objects. */
//	    clReleaseMemObject(bufX);
//	    clReleaseMemObject(bufNRM2);
//	    clReleaseMemObject(scratchBuff);
//
//	    /* Finalize work with clblas. */
//	    clblasTeardown();
//
//	    /* Release OpenCL working objects. */
//	    clReleaseCommandQueue(queue);
//	    clReleaseContext(ctx);
//free(X);
//	    return ret;
//}
//
//int matmultTest(){
//	int a =1024;
//
//	float *A= (float*) calloc(a*a, sizeof(float));
//		for(int i=0;i<a*a;i++){
//			A[i]=1;
//		}
//	float *B= (float*) calloc(a*a, sizeof(float));
//		for(int i=0;i<a*a;i++){
//			B[i]=1;
//		}
//	float *C= (float*) calloc(a*a, sizeof(float));
//		for(int i=0;i<a;i++){
//			C[i]=1.5;
//		}
//	int M=a;
//	int N=a;
//	int K=a;
//	static const size_t lda = K;
//	static const size_t ldb = N;
//	static const size_t ldc = N;
//	float *result= (float*) calloc(a*a, sizeof(float));
//
//	cl_int err;
//	    cl_platform_id platform = 0;
//	    cl_device_id device = 0;
//	    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
//	    cl_context ctx = 0;
//	    cl_command_queue queue = 0;
//	    cl_mem bufA, bufB, bufC;
//	    cl_event event = NULL;
//	    int ret = 0;
//
//	    /* Setup OpenCL environment. */
//	    err = clGetPlatformIDs(1, &platform, NULL);
//	    if (err != CL_SUCCESS) {
//	    	LOGI( "clGetPlatformIDs() failed with %d\n", err );
//	        return 1;
//	    }
//
//	    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//	    if (err != CL_SUCCESS) {
//	    	LOGI( "clGetDeviceIDs() failed with %d\n", err );
//	        return 1;
//	    }
//
//	    props[1] = (cl_context_properties)platform;
//	    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
//	    if (err != CL_SUCCESS) {
//	    	LOGI( "clCreateContext() failed with %d\n", err );
//	        return 1;
//	    }
//
//	    queue = clCreateCommandQueue(ctx, device, 0, &err);
//	    if (err != CL_SUCCESS) {
//	    	LOGI( "clCreateCommandQueue() failed with %d\n", err );
//	        clReleaseContext(ctx);
//	        return 1;
//	    }
//
//	    /* Setup clblas. */
//	    err = clblasSetup();
//	    if (err != CL_SUCCESS) {
//	    	LOGI("clblasSetup() failed with %d\n", err);
//	        clReleaseCommandQueue(queue);
//	        clReleaseContext(ctx);
//	        return 1;
//	    }
//
//	    /* Prepare OpenCL memory objects and place matrices inside them. */
//	    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(float),
//	                          NULL, &err);
//	    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(float),
//	                          NULL, &err);
//	    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(float),
//	                          NULL, &err);
//
//
//	    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
//	        M * K * sizeof(*A), A, 0, NULL, NULL);
//	    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
//	        K * N * sizeof(*B), B, 0, NULL, NULL);
//	    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
//	        M * N * sizeof(*C), C, 0, NULL, NULL);
//
//	    /* Call clblas extended function. Perform gemm for the lower right sub-matrices */
//	    err = clblasSgemm(order, transA, transB, M - off, N - off, K - off,
//	                         alpha, bufA, offA, lda,
//	                         bufB, offB, ldb, beta,
//	                         bufC, offC, ldc,
//	                         1, &queue, 0, NULL, &event);
//
//	    if (err != CL_SUCCESS) {
//	    	LOGI("clblasSgemmEx() failed with %d\n", err);
//	        ret = 1;
//	    }
//	    else {
//	        /* Wait for calculations to be finished. */
//	        err = clWaitForEvents(1, &event);
//
//	        /* Fetch results of calculations from GPU memory. */
//	        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
//	                                  M * N * sizeof(*result),
//	                                  result, 0, NULL, NULL);
//
//	        /* At this point you will get the result of SGEMM placed in 'result' array. */
//	        puts("");
//	        printResult("clblasSgemmEx result",result,ldc);
//	    }
//
//
//	    /* Release OpenCL memory objects. */
//	    clReleaseMemObject(bufC);
//	    clReleaseMemObject(bufB);
//	    clReleaseMemObject(bufA);
//
//	    /* Finalize work with clblas. */
//	    clblasTeardown();
//
//	    /* Release OpenCL working objects. */
//	    clReleaseCommandQueue(queue);
//	    clReleaseContext(ctx);
//	    free(A);
//	    free(B);
//	    free(C);
//	    return ret;
//}
//
//void compareSpecific(int instance0, int instance1, int numIts, int batchSize, LayerDimensions dim) {
//    cout << "batchsize=" << batchSize << " " << dim << endl;
//    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//    ClBlasInstance clblasInstance;
//
//    int inputNumElements = dim.inputCubeSize * batchSize;
//    int errorsSize = dim.outputCubeSize * batchSize;
//    int weightsSize = dim.filtersSize;
//    int errorsForUpstreamSize = dim.inputCubeSize * batchSize;
//
//    float *input = new float[inputNumElements];
//    float *errors = new float[errorsSize];
//    float *weights = new float[weightsSize];
//    float *errorsForUpstream0 = new float[errorsForUpstreamSize];
//    float *errorsForUpstream1 = new float[errorsForUpstreamSize];
//
//    WeightRandomizer::randomize(0, input, inputNumElements, -0.1f, 0.1f);
//    WeightRandomizer::randomize(1, errors, errorsSize, -0.1f, 0.1f);
//    WeightRandomizer::randomize(2, weights, weightsSize, -0.1f, 0.1f);
//
//    CLWrapper *inputWrapper = cl->wrap(inputNumElements, input);
//    CLWrapper *errorsWrapper = cl->wrap(errorsSize, errors);
//    CLWrapper *weightsWrapper = cl->wrap(weightsSize, weights);
//    CLWrapper *errorsForUpstreamWrapper0 = cl->wrap(errorsForUpstreamSize, errorsForUpstream0);
//    CLWrapper *errorsForUpstreamWrapper1 = cl->wrap(errorsForUpstreamSize, errorsForUpstream1);
//
//    inputWrapper->copyToDevice();
//    errorsWrapper->copyToDevice();
//    weightsWrapper->copyToDevice();
//    errorsForUpstreamWrapper0->createOnDevice();
//    errorsForUpstreamWrapper1->createOnDevice();
//
//    Backward *bp0 = Backward::instanceSpecific(instance0, cl, dim);
//    Backward *bp1 = Backward::instanceSpecific(instance1, cl, dim);
//
//    for(int it=0; it < numIts; it++ ) {
//        bp0->backward(batchSize,
//                inputWrapper, errorsWrapper, weightsWrapper,
//                errorsForUpstreamWrapper0);
//        bp1->backward(batchSize,
//                inputWrapper, errorsWrapper, weightsWrapper,
//                errorsForUpstreamWrapper1);
//
//        errorsForUpstreamWrapper0->copyToHost();
//        errorsForUpstreamWrapper1->copyToHost();
//
//        int outputNumElements = errorsForUpstreamSize;
//        cout << dim << endl;
//        bool same = true;
//        for(int i = 0; i < max(20, outputNumElements); i++) {
//            if(i < outputNumElements) {
//                if(abs(errorsForUpstream0[i] - errorsForUpstream1[i]) < 0.000001 || abs(errorsForUpstream0[i] - errorsForUpstream1[i]) <= 0.001 * max(abs(errorsForUpstream0[i]), abs(errorsForUpstream1[i]))) {
//                    if(it == 0 && i < 20) {
//                        cout << "output[" << i << "]=" << errorsForUpstream0[i] << " " << errorsForUpstream1[i];
//                        cout << " SAME";
//                    }
//                } else {
//                    cout << "output[" << i << "]=" << errorsForUpstream0[i] << " " << errorsForUpstream1[i];
//                    cout << " DIFF";
//                    same = false;
//                }
//            } else {
//                 if(it == 0 && i < 20) {
//                     cout << "     ";
//                 }
//            }
//            if(it == 0 && i < 20) {
//                cout << "  || " << errorsForUpstream1[100+i] ;
//                cout << "  || " << errorsForUpstream1[200+i] ;
//                cout << "  || " << errorsForUpstream1[300+i] ;
//                cout << "  || " << errorsForUpstream1[400+i] ;
//                cout << "  || " << errorsForUpstream1[500+i] ;
//                cout << "  || " << errorsForUpstream1[600+i] ;
//                cout << "  || " << errorsForUpstream1[700+i] << endl;
//            }
//        }
//        //EXPECT_EQ(true, same);
//    }
//
//    delete inputWrapper;
//    delete errorsWrapper;
//    delete weightsWrapper;
//    delete errorsForUpstreamWrapper0;
//    delete errorsForUpstreamWrapper1;
//
//    delete[] errorsForUpstream0;
//    delete[] errorsForUpstream1;
//    delete bp0;
//    delete bp1;
//    delete cl;
//    delete[] input;
//    delete[] errors;
//    delete[] weights;
//}
//
//
//
//void test2(){
//	Timer timer;
//	    float data[] = { 0.5f, 0.5f, 0.5f,
//	                    -0.5f, 0.5f, 0.5f,
//	                    0.5f, 0.5f, 0.5f,
//
//	                   0.5f, 0.5f, 0.5f,
//	                   0.5f, -0.5f, 0.5f,
//	                   0.5f, 0.5f, 0.5f,
//
//	                    -0.5f, -0.5f, -0.5f,
//	                    -0.5f, 0.5f, -0.5f,
//	                    -0.5f, -0.5f, -0.5f,
//
//	                   -0.5f, -0.5f, -0.5f,
//	                   0.5f, -0.5f, -0.5f,
//	                   -0.5f, -0.5f, -0.5f
//	 };
//
//	    int *labels = new int[4];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    labels[2] = 0;
//	    labels[3] = 1;
//	    float *expectedOutput = new float[8];
//	    expectedOutput[0] = 0.5f;
//	    expectedOutput[1] = -0.5f;
//	    expectedOutput[2] = -0.5f;
//	    expectedOutput[3] = 0.5f;
//	    expectedOutput[4] = 0.5f;
//	    expectedOutput[5] = -0.5f;
//	    expectedOutput[6] = -0.5f;
//	    expectedOutput[7] = 0.5f;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(3) );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
//	    net->addLayer( ActivationMaker::instance()->tanh() );
//	    net->addLayer( SquareLossMaker::instance() );;
//	    float weights1[] = {-0.171115f, 0.28369f, 0.201354f, -0.496124f, 0.391512f, 0.120458f, 0.396952f, -0.1356f, -0.319595f, 0.251043f, 0.318859f, 0.220892f, -0.480651f, -0.51708f, 0.2173f, 0.365935f, 0.304687f, -0.712624f};
//	    float bias1[] = {0.375101f, 0.00130748f};
//	    net->initWeights(1, weights1);
//	    net->initBias(1, bias1 );
//	    float const*output = 0;
//
//	    SGD *sgd = SGD::instance( cl, 0.4f, 0 );
//	    InputData inputData( net->getInputCubeSize(), data );
//	    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
//	    LearnBatcher2 learnBatcher( net, sgd, 4, 4, &inputData, &expectedData );
//	    for( int epoch = 0; epoch < 15; epoch++ ) {
//	        learnBatcher.run( epoch );
//	//        batchLearner.runEpochFromExpected( sgd, 4, 4, data, expectedOutput );
//	//        net->printWeightsAsCode();
//	//        net->printBiasAsCode();
//	        //if( epoch % 5 == 0 ) {
//	        	float loss = net->calcLoss(expectedOutput);
//	        	LOGI("loss, E, %f",loss);
//	            //cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	            output = net->getOutput();
//	            AccuracyHelper::printAccuracy( 4, 2, labels, output );
//	        //}
//	    }
//	//    net->print();
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
//	    AccuracyHelper::printAccuracy( 4, 2, labels, output );
//	    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
////	    EXPECT_EQ( numCorrect, 4 );
////	    EXPECT_GE( 0.0001f, loss );
//
//	    delete sgd;
//	    delete net;
//	delete cl;
//}
//void test3(){//3rd simple conv test
//	Timer timer;
//	    float *data = new float[2];
//	    data[0] = 0.5f;
//	    data[1] = -0.5f;
//	    int *labels = new int[2];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    float *expectedOutput = new float[4];
//	    expectedOutput[0] = 1;
//	    expectedOutput[1] = 0;
//	    expectedOutput[2] = 0;
//	    expectedOutput[3] = 1;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = new NeuralNet(cl);
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
//	//    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//	    float weights1[] = {-0.380177f, -1.5738f};
//	    float bias1[] = {0.5f, 0.0606055f};
//	    net->initWeights( 1, weights1, bias1 );
//
//	    SGD *sgd = SGD::instance( cl, 0.2f, 0 );
//	    InputData inputData( net->getInputCubeSize(), data );
//	    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
//	    LearnBatcher2 learnBatcher( net, sgd, 2, 2, &inputData, &expectedData );
//	    for( int epoch = 0; epoch < 40; epoch++ ) {
//	        learnBatcher.run( epoch );
//	//    BatchLearner batchLearner( net );
//	//    SGD *sgd = SGD::instance( cl, 0.2f, 0 );
//	//    for( int epoch = 0; epoch < 40; epoch++ ) {
//	//        batchLearner.runEpochFromExpected( sgd, 2, 2, data, expectedOutput );
//	        LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        if( epoch % 5 == 0 ) {
//	            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	//            net->print();
//	    //        net->printWeightsAsCode();
//	    //        net->printBiasAsCode();
//	            float const*output = net->getOutput();
//	            AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	        }
//	    }
//	//    net->print();
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( 2, 2, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
////	    EXPECT_EQ( numCorrect, 2 );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
////	    EXPECT_GE( 0.001f, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//}
//
//void test5(){//3rd simple conv test
//	Timer timer;
//	    float data[] = { 0.5f, 0.5f, 0.5f,
//	                    -0.5f, 0.5f, 0.5f,
//	                    0.5f, 0.5f, 0.5f,
//
//	                   0.5f, 0.5f, 0.5f,
//	                   0.5f, -0.5f, 0.5f,
//	                   0.5f, 0.5f, 0.5f,
//
//	                    -0.5f, -0.5f, -0.5f,
//	                    -0.5f, 0.5f, -0.5f,
//	                    -0.5f, -0.5f, -0.5f,
//
//	                   -0.5f, -0.5f, -0.5f,
//	                   0.5f, -0.5f, -0.5f,
//	                   -0.5f, -0.5f, -0.5f
//	 };
//
//	    int *labels = new int[4];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    labels[2] = 0;
//	    labels[3] = 1;
//	    float *expectedOutput = new float[8];
//	    expectedOutput[0] = 1;
//	    expectedOutput[1] = 0;
//	    expectedOutput[2] = 0;
//	    expectedOutput[3] = 1;
//	    expectedOutput[4] = 1;
//	    expectedOutput[5] = 0;
//	    expectedOutput[6] = 0;
//	    expectedOutput[7] = 1;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(3) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(3)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
//	    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//	    float const*output = 0;
//	    double _weights1[] = {0.0113327, 0.280063, -0.0584702, -0.503431, -0.37286, -0.457257, 0.29226, -0.360089, -0.273977, 0.530185, -0.460167, 0.489126, 0.141883, 0.179525, -0.18084, 0.412117, 0.0866731, -0.247958};
//	    vector<float> __weights1( _weights1, _weights1 + sizeof( _weights1 ) / sizeof(double) );
//	    float *weights1 = &__weights1[0];
//	    float bias1[] = {0.0418723f, 0.158733f};
//	    net->getLayer(1)->setWeights( weights1, bias1 );
//	//    BatchLearner batchLearner( net );
//	    SGD *sgd = SGD::instance( cl, 0.1f, 0 );
//	    for( int epoch = 0; epoch < 50; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(4)
//	            ->numExamples(4)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//	        LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        if( epoch % 5 == 0 ) {
//	            output = net->getOutput();
//	    //        net->printWeightsAsCode();
//	    //        net->printBiasAsCode();
//	            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	            AccuracyHelper::printAccuracy( 4, 2, labels, output );
//	        }
//	    }
//	//    net->print();
//	    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    AccuracyHelper::printAccuracy( 4, 2, labels, output );
//	    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getOutput() );
//	    LOGI("accuracy: %d/4",numCorrect);
//	    //cout << "accuracy: " << numCorrect << "/" << 4 << endl;
//	    //EXPECT_EQ( numCorrect, 4 );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
//	    //EXPECT_GE( 0.000001, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//}
//
//void test6(){//3rd simple conv test
//	Timer timer;
//	    float data[] = { 0.5f, 0.5f, 0.5f,
//	                    -0.5f, 0.5f, 0.5f,
//	                    0.5f, 0.5f, 0.5f,
//
//	                   0.5f, 0.5f, 0.5f,
//	                   0.5f, -0.5f, 0.5f,
//	                   0.5f, 0.5f, 0.5f,
//
//	                    -0.5f, -0.5f, -0.5f,
//	                    -0.5f, 0.5f, -0.5f,
//	                    -0.5f, -0.5f, -0.5f,
//
//	                   -0.5f, -0.5f, -0.5f,
//	                   0.5f, -0.5f, -0.5f,
//	                   -0.5f, -0.5f, -0.5
//	 };
//
//	    int *labels = new int[4];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    labels[2] = 0;
//	    labels[3] = 1;
//	    float *expectedOutput = new float[8];
//	    expectedOutput[0] = 1;
//	    expectedOutput[1] = 0;
//	    expectedOutput[2] = 0;
//	    expectedOutput[3] = 1;
//	    expectedOutput[4] = 1;
//	    expectedOutput[5] = 0;
//	    expectedOutput[6] = 0;
//	    expectedOutput[7] = 1;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(3) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(3)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
//	    net->addLayer( SquareLossMaker::instance() );;
//	    float const*output = 0;
//	    float weights1[] = {0.715867f, -0.428623f, -0.281465f, -0.736675f, -0.224507f, 0.335028f, -0.384762f, -0.213304f, 0.679177f, -0.170055f, 0.335075f, -0.572057f, -0.175718f, -0.410962f, -0.175277f, 0.536131f, -0.0568329f, -0.00297278f};
//	    float bias1[] = {0.5f, 0.5f};
//	    net->initWeights( 1, weights1, bias1 );
//	    SGD *sgd = SGD::instance( cl, 0.09f, 0 );
//	    for( int epoch = 0; epoch < 20; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(4)
//	            ->numExamples(4)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//	//        net->printWeightsAsCode();
//	//        net->printBiasAsCode();
//	        LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        if( epoch % 5 == 0 ) {
//	            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	            output = net->getOutput();
//	            AccuracyHelper::printAccuracy( 4, 2, labels, output );
//	        }
//	    }
//	//    net->print();
//	    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    AccuracyHelper::printAccuracy( 4, 2, labels, output );
//	    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
////	    EXPECT_EQ( numCorrect, 4 );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
////	    EXPECT_GE( 0.001f, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//}
//
//void test7(){//3rd simple conv test
//	Timer timer;
//	    float *data = new float[2];
//	    data[0] = 0.5f;
//	    data[1] = -0.5f;
//	    int *labels = new int[2];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    float *expectedOutput = new float[4];
//	    expectedOutput[0] = 0.5f;
//	    expectedOutput[1] = -0.5f;
//	    expectedOutput[2] = -0.5f;
//	    expectedOutput[3] = 0.5f;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(1) );
//	    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(1) );
//	//    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//
//	    float weights1[] = {-0.303866f, -1.66244f};
//	    float weights3[] = {0.426358f, -0.841404f, -0.420361f, 0.841048f};
//	    float bias1[] = {-0.324465f, 0.731219f};
//	    float bias3[] = {0.600115f, -0.599876f};
//	    net->initWeights( 1, weights1, bias1 );
//	    net->initWeights( 3, weights3, bias3 );
//
//	    SGD *sgd = SGD::instance( cl, 0.1f, 0.0f );
//	    for( int epoch = 0; epoch < 40; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(2)
//	            ->numExamples(2)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//	        LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        //cout << "epoch " << epoch << " loss, E, " << net->calcLoss(expectedOutput) << endl;
//	//        net->print();
//	//        float const*output = net->getOutput();
//	//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	    }
//	    net->print();
//	    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( 2, 2, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
//	    //EXPECT_EQ( numCorrect, 2 );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
//	    //EXPECT_GE( 0.0001f, loss );
//
//	        cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    net->print();
//	    net->getLayer(1)->getWeights();
//	    net->getLayer(3)->getWeights();
//	    //NetTestHelper::printWeightsAsCode( net );
//	    //NetTestHelper::printBiasAsCode( net );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//}
//
//void test8(){//3rd simple conv test
//	Timer timer;
//	    float *data = new float[2];
//	    data[0] = 0.5f;
//	    data[1] = -0.5f;
//	    int *labels = new int[2];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    float *expectedOutput = new float[4];
//	    expectedOutput[0] = 0.5f;
//	    expectedOutput[1] = -0.5f;
//	    expectedOutput[2] = -0.5f;
//	    expectedOutput[3] = 0.5f;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
//	    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
//	//    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//	/*float weights1[] = {1.12739f, 1.21476f};
//	float weights2[] = {-0.352846f, 0.534554f, -1.13343f, -0.191175f};
//	float bias1[] = {0.971267f, 1.42629f};
//	float bias2[] = {-0.071288f, 0.443919f};
//	    net->initWeights(1, weights1, bias1 );
//	    net->initWeights(3, weights2, bias2 );*/
//	    SGD *sgd = SGD::instance( cl, 0.2f, 0 );
//	    for( int epoch = 0; epoch < 30; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(2)
//	            ->numExamples(2)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//	        LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        if( epoch % 5 == 0 ) {
//	//           net->printWeightsAsCode();
//	//            net->printBiasAsCode();
//	        cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	        }
//	//        net->print();
//	//        float const*output = net->getOutput();
//	//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	    }
//	//    net->print();
//
//	    StatefulTimer::dump(true);
//
//	    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( 2, 2, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
//	    LOGI("accuracy: %d/%d",numCorrect,2);
//	    //EXPECT_EQ( numCorrect, 2 );
//
//	    float loss = net->calcLoss(expectedOutput);
//
//	    cout << "loss, E, " << loss << endl;
//	    //EXPECT_GE( 0.0001f, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//
//}
//
//void test9(){//3rd simple conv test
//	Timer timer;
//	    int imageSize = 5;
//	    int N = 3;
//	    int numInPlanes = 1;
//	    int numOutPlanes = 3;
//	    float data[] = {
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	};
//	    int inputNumElements = imageSize * imageSize * numInPlanes * N;
//	    for( int i = 0; i < inputNumElements; i++ ) {
//	        data[i] -= 0.5f;
//	    }
//	    int labels[] = { 0, 1, 2 };
//	    int outputNumElements = numOutPlanes * N;
//	    float *expectedOutput = new float[outputNumElements];
//	    for( int n = 0; n < N; n++ ) {
//	        for( int plane = 0; plane < numOutPlanes; plane++ ) {
//	            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
//	        }
//	        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
//	    }
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(2)->biased() );
//	    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(4)->biased() );
//	//    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//	//    net->print();
//	    SGD *sgd = SGD::instance( cl, 0.01f, 0 );
//	    for( int epoch = 0; epoch < 1000/* 1000*/; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(N)
//	            ->numExamples(N)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//	        if (epoch<40)
//	        	LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        if( epoch % 100 == 0 ) cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	//        net->print();
//	//        float const*output = net->getOutput();
//	//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	    }
//	//    net->print();
//	    LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << N << endl;
//	    LOGI("accuracy: %d/%d",numCorrect,N);
//	    //EXPECT_EQ( numCorrect, N );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
//	    //EXPECT_GE( 0.01f, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//
//}
//
//void test10(){//3rd simple conv test
//	   Timer timer;
//	    int imageSize = 5;
//	    int N = 6;
//	    int numInPlanes = 1;
//	    int numOutPlanes = 3;
//	    float data[] = {
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	};
//	    int inputNumElements = imageSize * imageSize * numInPlanes * N;
//	    for( int i = 0; i < inputNumElements; i++ ) {
//	        data[i] -= 0.5f;
//	    }
//	    int labels[] = { 0, 1, 2, 0, 1, 2 };
//	    int outputNumElements = numOutPlanes * N;
//	    float *expectedOutput = new float[outputNumElements];
//	    for( int n = 0; n < N; n++ ) {
//	        for( int plane = 0; plane < numOutPlanes; plane++ ) {
//	            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
//	        }
//	        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
//	    }
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(2)->biased() );
//	    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(4)->biased() );
//	//    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//	//    net->print();
//	double _weights1[] = {-0.69664, 0.58017, 0.140447, -0.205859, 0.0198638, 0.0110593, -0.388923, -0.844424, -0.472903, 0.453888, -0.616155, -0.454998};
//	double _weights2[] = {0.207138, -0.106497, -0.1228, -0.162173, 0.1822, -0.100027, 0.0447708, 0.165723, -0.0147989, 0.109204, -0.0334504, 0.00452646, 0.198443, -0.23725, 0.105671, 0.192242, -0.0268933, 0.150674, 0.160054, -0.116846, 0.222009,
//	0.226935, 0.113873, -0.153742, 0.0273874, -0.216493, 0.177896, 0.155068, -0.0809009, 0.0305763, 0.198926, -0.115796, -0.179839, -0.133567, -0.0386595, -0.166771, -0.11833, -0.219205, -0.0115777, 0.122457, 0.0984342,
//	0.0616336, 0.130647, 0.192949, 0.143467, -0.130633, -0.221122, -0.154317, 0.11901, 0.00502961, 0.213079, -0.0373076, -0.0461127, -0.156646, -0.148074, -0.105763, -0.140191, 0.136911, -0.217382, 0.17574, -0.0312263,
//	0.0931478, 0.0789604, -0.00794073, -0.218235, 0.0418423, 0.234828, 0.225359, -0.191966, 0.241517, 0.182442, -0.216337, -0.228462, -0.140195, 0.0493267, 0.0383108, -0.0124946, -0.093023, 0.0322872, 0.0855678, -0.0466207,
//	-0.025329, -0.198314, -0.0189797, 0.147109, -0.200046, 0.20127, 0.169828, -0.173335, -0.100567, -0.195165, -0.0657755, -0.224493, -0.208405, 0.154131, 0.12547, -0.161635, -0.248707, 0.13305, -0.00289013, 0.228017,
//	0.0528438, 0.0157539, 0.161637, -0.199882, 0.171727, 0.171146, -0.237469, -0.226088, 0.2026, -0.131614, 0.0631847, -0.0949208, -0.137853, -0.177839, -0.237589, -0.229862, 0.202094, 0.0531539, -0.0467284, 0.125544,
//	-0.0750956, 0.225228, 0.255915, 0.076901, -0.0596187, 0.16937, -0.104811, -0.0815879, -0.196806, 0.0526821, 0.136622, -0.12163, 0.170657, -0.0956968, -0.00985565, 0.0455411, 0.0242914, 0.107953, -0.0594324, 0.124928,
//	0.0875922, -0.100952, 0.155045};
//	vector<float> vweights1( _weights1, _weights1 + sizeof(_weights1) / sizeof(_weights1[0] ) );
//	float *weights1 = &vweights1[0];
//	vector<float> vweights2( _weights2, _weights2 + sizeof(_weights2) / sizeof(_weights2[0] ) );
//	float *weights2 = &vweights2[0];
//	float bias1[] = {0.0998941f, -0.365008f, 0.188937f};
//	float bias2[] = {0.232961f, 0.141537f, 0.159074f};
//	    net->initWeights(1, weights1, bias1 );
//	    net->initWeights(3, weights2, bias2 );
//	    SGD *sgd = SGD::instance( cl, 0.04f, 0 );
//	    for( int epoch = 0; epoch < 500; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(N)
//	            ->numExamples(N)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//
//	        LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	        if( epoch % 100 == 0 ) {
//	            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	//        net->print();
//	//           net->printWeightsAsCode();
//	//            net->printBiasAsCode();
//	        }
//	//        float const*output = net->getOutput();
//	//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	    }
//	//    net->print();
//	    LOGI("loss, E, %f",net->calcLoss(expectedOutput));
//	    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << N << endl;
//	    LOGI("accuracy: %d/%d",numCorrect,N);
//	    //EXPECT_EQ( numCorrect, N );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    cout << "loss, E, " << loss << endl;
//	    //EXPECT_GE( 0.01f, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//
//}
//
//void test11(){//3rd simple conv test
//    Timer timer;
//    int imageSize = 5;
//    int N = 6;
//    int numInPlanes = 1;
//    int numOutPlanes = 3;
//    float data[] = {
//                    1,0,1,0,1,
//                    0,1,0,1,0,
//                    1,0,1,0,1,
//                    0,1,0,1,0,
//                    1,0,1,0,1,
//
//                    1,0,1,0,1,
//                    1,0,1,0,1,
//                    1,0,1,0,1,
//                    1,0,1,0,1,
//                    1,0,1,0,1,
//
//                    1,1,1,1,1,
//                    0,0,0,0,0,
//                    1,1,1,1,1,
//                    0,0,0,0,0,
//                    1,1,1,1,1,
//
//                    0,1,0,1,0,
//                    1,0,1,0,1,
//                    0,1,0,1,0,
//                    1,0,1,0,1,
//                    0,1,0,1,0,
//
//                    0,1,0,1,0,
//                    0,1,0,1,0,
//                    0,1,0,1,0,
//                    0,1,0,1,0,
//                    0,1,0,1,0,
//
//                    0,0,0,0,0,
//                    1,1,1,1,1,
//                    0,0,0,0,0,
//                    1,1,1,1,1,
//                    0,0,0,0,0,
//};
//    int inputNumElements = imageSize * imageSize * numInPlanes * N;
//    for( int i = 0; i < inputNumElements; i++ ) {
//        data[i] -= 0.5f;
//    }
//    int labels[] = { 0, 1, 2, 0, 1, 2 };
//    int outputNumElements = numOutPlanes * N;
//    float *expectedOutput = new float[outputNumElements];
//    for( int n = 0; n < N; n++ ) {
//        for( int plane = 0; plane < numOutPlanes; plane++ ) {
//            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
//        }
//        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
//    }
//    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//    ClBlasInstance blasInstance;
//    NeuralNet *net = NeuralNet::maker(cl)->instance();
//    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
////    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
//    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
//    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
////    net->addLayer( ActivationMaker::instance()->relu() );
//    net->addLayer( SquareLossMaker::instance() );
////    net->print();
//double _weights1[] = {-0.171255, 0.374466, -0.224289, -0.196481, 0.162787, 0.418841, 0.230909, 0.23731, -0.244594, -0.469993, 0.221895, -0.0145731, 0.163359, 0.276707, -0.533498, -0.376532, 0.275129, -0.298299, -0.162541, -0.497442, 0.0331104,
//0.140816, 0.339377, -0.466528, -0.260578, -0.373026, -0.0151962};
//double _weights2[] = {0.11266, 0.199489, 0.193306, -0.0574513, 0.266716, -0.271093, 0.0622974, 0.276959, 0.234103, -0.0329131, 0.111828, 0.255213, 0.0546736, -0.14267, -0.195783, 0.140402, -0.225388, 0.143696, 0.00776717, -0.216402, 0.13755,
//-0.0404622, 0.321655, -0.218655, -0.140874, 0.0361279, 0.227149, -0.0224601, -0.0438027, 0.0945921, 0.264248, -0.212632, 0.125262, 0.303234, 0.265334, 0.0165108, -0.119786, 0.0967013, -0.316602, 0.0735333, -0.298583,
//-0.131285, 0.158645, 0.0816884, 0.0191159, 0.233569, -0.0288674, 0.166787, 0.0839494, -0.232928, 0.32289, 0.259277, 0.28396, 0.0585126, 0.0419515, -0.315813, 0.32489, -0.208887, -0.157422, 0.223066, 0.235666,
//-0.286893, -0.00949466, -0.0232266, 0.000597281, -0.28573, 0.23746, -0.12194, 0.211189, 0.114797, 0.334012, 0.195305, 0.0269026, 0.191523, -0.0801473, 0.323508, 0.214993, -0.0651319, 0.268872, -0.270865, 0.0842015
//};
//vector<float> __weights1( _weights1, _weights1 + sizeof( _weights1 ) / sizeof(double) );
//vector<float> __weights2( _weights2, _weights2 + sizeof( _weights2  ) / sizeof(double) );
//float *weights1 = &__weights1[0];
//float *weights2 = &__weights2[0];
//float bias1[] = {0.224118f, -0.246188f, -0.22282f};
//float bias2[] = {-0.0863176f, -0.227985f, -0.147554f};
//    net->initWeights(1, weights1, bias1 );
//    net->initWeights(3, weights2, bias2 );
//    SGD *sgd = SGD::instance( cl, 0.04f, 0 );
//    for( int epoch = 0; epoch < 300; epoch++ ) {
//        net->epochMaker(sgd)
//            ->batchSize(N)
//            ->numExamples(N)
//            ->inputData(data)
//            ->expectedOutputs(expectedOutput)
//            ->run(epoch);
//        LOGI("loss, E, : %f",net->calcLoss(expectedOutput));
//        if( epoch % 100 == 0 ) {
//            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
////        net->print();
////           net->printWeightsAsCode();
////            net->printBiasAsCode();
//        }
////        float const*output = net->getOutput();
////        AccuracyHelper::printAccuracy( 2, 2, labels, output );
//    }
////    net->print();
//    //cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//    LOGI("loss, E, : %f",net->calcLoss(expectedOutput));
//    float const*output = net->getOutput();
//    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );
//
//    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
//    cout << "accuracy: " << numCorrect << "/" << N << endl;
//    LOGI("accuracy: %d/%d",numCorrect,N);
////    EXPECT_EQ( numCorrect, N );
//
//    float loss = net->calcLoss(expectedOutput);
//    cout << "loss, E, " << loss << endl;
//   // EXPECT_GE( 0.1f, loss );
//
//    delete sgd;
//    delete net;
//    delete cl;
//
//}
//
//void test(){//3rd simple conv test
//	Timer timer;
//	    int imageSize = 5;
//	    int N = 18;
//	    int numInPlanes = 1;
//	    int numOutPlanes = 3;
//	    float data[] = {
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	//1
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	//2
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	//3
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	//4
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	//5
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	//6
//	                    1,0,1,0,1,
//	                    0,1,0,1,0,
//	                    1,0,1,0,1,
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	//7
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    1,0,1,0,1,
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	//8
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	//9
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	                    0,0,0,1,0,
//	                    0,0,1,0,1,
//	                    0,0,0,1,0,
//	//10
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	                    0,1,0,1,0,
//	//11
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	                    0,0,0,0,0,
//	                    1,1,1,1,1,
//	                    0,0,0,0,0,
//
//	//12
//	                    0,0,1,0,1,
//	                    0,0,0,1,0,
//	                    0,0,1,0,1,
//	                    0,0,0,1,0,
//	                    0,0,1,0,1,
//	//13
//	                    0,0,1,0,1,
//	                    0,0,1,0,1,
//	                    0,0,1,0,1,
//	                    0,0,1,0,1,
//	                    0,0,1,0,1,
//	//14
//	                    0,0,1,1,1,
//	                    0,0,0,0,0,
//	                    0,0,1,1,1,
//	                    0,0,0,0,0,
//	                    0,0,1,1,1,
//	//15
//	                    0,1,0,0,0,
//	                    1,0,1,0,0,
//	                    0,1,0,0,0,
//	                    1,0,1,0,0,
//	                    0,1,0,0,0,
//	//16
//	                    0,1,0,0,0,
//	                    0,1,0,0,0,
//	                    0,1,0,0,0,
//	                    0,1,0,0,0,
//	                    0,1,0,0,0,
//	//17
//	                    0,0,0,0,0,
//	                    1,1,1,0,0,
//	                    0,0,0,0,0,
//	                    1,1,1,0,0,
//	                    0,0,0,0,0,
//	};
//	    int inputNumElements = imageSize * imageSize * numInPlanes * N;
//	    for( int i = 0; i < inputNumElements; i++ ) {
//	        data[i] -= 0.5f;
//	    }
//	    int labels[] = { 0, 1, 2, 0, 1, 2,
//	                    0, 1, 2, 0, 1, 2,
//	                    0, 1, 2, 0, 1, 2 };
//	    int outputNumElements = numOutPlanes * N;
//	    float *expectedOutput = new float[outputNumElements];
//	    for( int n = 0; n < N; n++ ) {
//	        for( int plane = 0; plane < numOutPlanes; plane++ ) {
//	            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
//	        }
//	        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
//	    }
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//	//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
//	    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
//	//    net->addLayer( ActivationMaker::instance()->relu() );
//	    net->addLayer( SquareLossMaker::instance() );
//	//    net->print();
//	    SGD *sgd = SGD::instance( cl, 0.02f, 0 );
//	    for( int epoch = 0; epoch < 3000; epoch++ ) {
//	        net->epochMaker(sgd)
//	            ->batchSize(N)
//	            ->numExamples(N)
//	            ->inputData(data)
//	            ->expectedOutputs(expectedOutput)
//	            ->run(epoch);
//	        if( epoch % 100 == 0 ) {
//	        	LOGI("loss, E, : %f",net->calcLoss(expectedOutput));
//	            //cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	//        net->print();
//	//           net->printWeightsAsCode();
//	//            net->printBiasAsCode();
//	        }
//	//        float const*output = net->getOutput();
//	//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	    }
//	    net->print();
//	    LOGI("loss, E, : %f",net->calcLoss(expectedOutput));
//	    //cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
//	    cout << "accuracy: " << numCorrect << "/" << N << endl;
//	    //EXPECT_EQ( numCorrect, N );
//
//	    float loss = net->calcLoss(expectedOutput);
//	    LOGI("accuracy: %d/%d",numCorrect,N);
//	    //cout << "loss, E, " << loss << endl;
//	    //EXPECT_GE( 0.1f, loss );
//
//	    delete sgd;
//	    delete net;
//	    delete cl;
//
//
//}
//
//void testDeepCL2(){
//	test();
//
//
//
//}
//
//void testDeepCL(){
//
//	Timer timer;
//	    const float learningRate = 0.1f;
//	    const int batchSize = 2;
//	    float *data = new float[batchSize];
//	    data[0] = 0.5f;
//	    data[1] = -0.5f;
//	    int *labels = new int[batchSize];
//	    labels[0] = 0;
//	    labels[1] = 1;
//	    float *expectedOutput = new float[4];
//	    expectedOutput[0] = 0.5f;
//	    expectedOutput[1] = -0.5f;
//	    expectedOutput[2] = -0.5f;
//	    expectedOutput[3] = 0.5f;
//	    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
//	    ClBlasInstance blasInstance;
//	    NeuralNet *net = NeuralNet::maker(cl)->instance();
//	    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//	    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(0) );
//	    net->addLayer( ActivationMaker::instance()->tanh() );
//	    net->addLayer( SquareLossMaker::instance() );;
//	    float weights1[] = {0.382147f, -1.77522f};
//	    net->initWeights(1, weights1);
//
//	//    BatchLearner batchLearner( net );
//	    SGD *sgd = SGD::instance( cl, learningRate, 0 );
//	    InputData inputData( net->getInputCubeSize(), data );
//	    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
//	    LearnBatcher2 learnBatcher( net, sgd, batchSize, batchSize,
//	            &inputData, &expectedData );
//	    for( int epoch = 0; epoch < 50; epoch++ ) {
//	        learnBatcher.run( epoch );
//
//	//        batchLearner.runEpochFromExpected( sgd, batchSize, batchSize, data, expectedOutput );
//	        if( epoch % 10 == 0 ) {
//	            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//	            float const*output = net->getOutput();
//	            AccuracyHelper::printAccuracy( 2, 2, labels, output );
//	        }
//	    }
//
//	    float loss = net->calcLoss(expectedOutput);
//	    LOGI("loss, E, %f",loss);
//	    float const*output = net->getOutput();
//	    AccuracyHelper::printAccuracy( 2, 2, labels, output );
//
//	    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
//	    LOGI("accuracy: %d/2",numCorrect);
////	    EXPECT_EQ( numCorrect, 2 );
////	    EXPECT_GE( 0.03, loss );
//
//	    delete sgd;
//	    delete net;
//	delete cl;
//}
//
//
//void openCLNR (unsigned char* bufIn, unsigned char* bufOut, int* info)
//{
//
////	LOGI( "test begin.");
////	EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();
////
////	LOGI( "1.");
////	    CLKernel *kernel = cl->buildKernelFromString("openCLNR", getKernel(), "test", "");
////	    LOGI( "2.");
////	    float in[1024];
////	    for(int i = 0; i < 1024; i++) {
////	        in[i] = 5;
////	    }
////	    LOGI( "3.");
////	    float out[1024];
////	    kernel->input(1024, in);
////	    kernel->output(1024, out);
////	    size_t global = 1024;
////	    size_t local = 1024;
////	    LOGI( "4.");
////	    kernel->run(1, &global, &local);
////	    LOGI( "test complete.");
////	    LOGI( "output[%d]=%f",0,out[0]);
////	    LOGI( "output[%d]=%f",1,out[1]);
////	delete cl;
//	    //matmultTest();
//	    testDeepCL2();
//	//TrainModel t;
//	//t.trainCmd("train numtrain=10 batchsize=2 netdef=1n learningrate=0.002 dataset=mnist");
//
//
//	//normTest();
//	    //
////	    for(int i=0;i<5;i++){
////	    	LOGI( "output[%d]=%d",i,out[i]);
////	    }
//
////	bool clpresent = 0 == clewInit();
////	    if( !clpresent ) {
////	        LOGI( "opencl library not found.");
//////	        return -1;
////	    }
////
////	    cl_int error = 0;
////	    cl_platform_id platform_ids[10];
////	    cl_uint num_platforms;
////	    error = clGetPlatformIDs(10, platform_ids, &num_platforms);
////		if (error != CL_SUCCESS) {
////			LOGI("something went wrong");
////		}
////		LOGI("num platforms: %d",num_platforms);
////	float *inputData= (float*) calloc(3*256, sizeof(float));
////
////	for(int i=0;i<256;i++){
////		inputData[i]=1.5;
////		inputData[i+1]=1.5;
////		inputData[i+2]=1.5;
////	}
////
////	LOGI("\n\nStart openCLNR (i.e., OpenCL on the GPU)");
////
////	int width = info[0];
////	int height = info[1];
////	unsigned int imageSize = width * height * 4 * sizeof(cl_uchar);
////
////	cl_int err = CL_SUCCESS;
////	try {
////
////		std::vector<cl::Platform> platforms;
////		cl::Platform::get(&platforms);
////		if (platforms.size() == 0) {
////			std::cout << "Platform size 0\n";
////			return;
////		}
////
////		cl_context_properties properties[] =
////		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
////		cl::Context context(CL_DEVICE_TYPE_GPU, properties);
////
////		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
////		cl::CommandQueue queue(context, devices[0], 0, &err);
////
////		std::string kernelSource = loadProgram("/data/data/com.sony.openclexample1/app_execdir/similarityMatrix.cl");
////
////		cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(), kernelSource.length()+1));
////		cl::Program program(context, source);
////		const char *options = "-cl-fast-relaxed-math";
////		program.build(devices, options);
////
////		cl::Kernel kernel(program, "ImageBoxFilter", &err);
////		cl::Kernel kernel2(program, "ImageBoxFilter16NSampling", &err);
////
////		cl::Buffer bufferIn = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imageSize, (void *) &bufIn[0], &err);
////		cl::Buffer bufferOut = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, imageSize, (void *) &bufOut[0], &err);
////
////		//Let image constructor select row pitch
////		std::size_t row_pitch = 0, rows = width, cols = height;
////
////		cl::ImageFormat format;
////		format.image_channel_order = CL_BGRA;
////		format.image_channel_data_type = CL_UNORM_INT8;
////		cl::Image2D m_sourceImage = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_BGRA, CL_UNORM_INT8), width, height, 0, &bufIn[0]);
////
////		cl::Image2D m_destImage = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_BGRA, CL_UNORM_INT8), width, height, 0, &bufIn[0]);
////		cl::Image2D m_bufImage = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_BGRA, CL_UNORM_INT8), width, height, 0, &bufOut[0]);
////		cl::Sampler m_sampler = cl::Sampler( context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);
////
////		kernel.setArg(0,m_sourceImage);
////		kernel.setArg(1,m_bufImage);
////		kernel.setArg(2,m_sampler);
////		kernel.setArg(3,width);
////		kernel.setArg(4,height);
////
////		kernel2.setArg(0,m_bufImage);
////		kernel2.setArg(1,m_destImage);
////		kernel2.setArg(2,m_sampler);
////		kernel2.setArg(3,width);
////		kernel2.setArg(4,height);
////
////		size_t maxWorkGroupSize = devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
////		LOGE("maxWorkGroupSize: %zu",maxWorkGroupSize);
////		size_t localWorkSize[2] = { (size_t) sqrtf( maxWorkGroupSize), (size_t) sqrtf(  maxWorkGroupSize) };
////		size_t globalWorkSize[2];
////	    // Compute the next global size that is a multiple of the local size
////	    size_t remndr = width % localWorkSize[0];
////	    if( remndr == 0 )
////	        globalWorkSize[0] = width;
////	    else
////	        globalWorkSize[0] = width + localWorkSize[0] - remndr;
////
////	    remndr = height % localWorkSize[1];
////	    if( remndr == 0 )
////	        globalWorkSize[1] = height;
////	    else
////	        globalWorkSize[1] = height + localWorkSize[1] - remndr;
////
////	    cl::Event event;
////	    clock_t startTimer1, stopTimer1;
////	    		startTimer1=clock();
////
////	    queue.enqueueNDRangeKernel(	kernel,
////				cl::NullRange,
////				cl::NDRange(globalWorkSize[0],globalWorkSize[1]),
////				cl::NDRange(localWorkSize[0],localWorkSize[1]),
////				NULL,
////				&event);
////
////	    queue.enqueueNDRangeKernel(	kernel2,
////	    				cl::NullRange,
////	    				cl::NDRange(globalWorkSize[0],globalWorkSize[1]),
////	    				cl::NDRange(localWorkSize[0],localWorkSize[1]),
////	    				NULL,
////	    				&event);
////
////	    queue.finish();
////	    stopTimer1 = clock();
////	    		double elapse = 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC;
////	    		info[2] = (int)elapse;
////	    		LOGI("OpenCL code on the GPU took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
////
////	    //BYTE* pRGBABits = new BYTE[ width * height * 4 ];
////	    cl::size_t<3> origin;
////	    origin[0] = 0; origin[1] = 0, origin[2] = 0;
////	    cl::size_t<3> region;
////	    region[0] = width; region[1] = height; region[2] = 1;
////
////
////	    queue.enqueueReadImage(m_destImage, CL_TRUE, origin, region, 0, 0, bufOut);
////
////	}
////	catch (cl::Error err) {
////		LOGE("ERROR: %s\n", err.what());
////	}
////	free(inputData);
//	return;
//}
//
//
///*
//void openCLNR (unsigned char* bufIn, unsigned char* bufOut, int* info)
//{
//
//	LOGI("\n\nStart openCLNR (i.e., OpenCL on the GPU)");
//
//	int width = info[0];
//	int height = info[1];
//	unsigned int imageSize = width * height * 4 * sizeof(cl_uchar);
//
//	cl_int err = CL_SUCCESS;
//	try {
//
//		std::vector<cl::Platform> platforms;
//		cl::Platform::get(&platforms);
//		if (platforms.size() == 0) {
//			std::cout << "Platform size 0\n";
//			return;
//		}
//
//		cl_context_properties properties[] =
//		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
//		cl::Context context(CL_DEVICE_TYPE_GPU, properties);
//
//		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
//		cl::CommandQueue queue(context, devices[0], 0, &err);
//
//		std::string kernelSource = loadProgram("/data/data/com.sony.openclexample1/app_execdir/bilateralKernel.cl");
//
//		cl::Program::Sources source(1, std::make_pair(kernelSource.c_str(), kernelSource.length()+1));
//		cl::Program program(context, source);
//		const char *options = "-cl-fast-relaxed-math";
//		program.build(devices, options);
//
//		cl::Kernel kernel(program, "bilateralFilterKernel", &err);
//
//		cl::Buffer bufferIn = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imageSize, (void *) &bufIn[0], &err);
//		cl::Buffer bufferOut = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, imageSize, (void *) &bufOut[0], &err);
//
//		kernel.setArg(0,bufferIn);
//		kernel.setArg(1,bufferOut);
//		kernel.setArg(2,width);
//		kernel.setArg(3,height);
//
//		cl::Event event;
//
//		clock_t startTimer1, stopTimer1;
//		startTimer1=clock();
//
//		//one time
//		queue.enqueueNDRangeKernel(	kernel,
//				cl::NullRange,
//				cl::NDRange(width,height),
//				cl::NullRange,
//				NULL,
//				&event);
//
//		//swap in and out buffer pointers and run a 2nd time
//		kernel.setArg(0,bufferOut);
//		kernel.setArg(1,bufferIn);
//		queue.enqueueNDRangeKernel(	kernel,
//				cl::NullRange,
//				cl::NDRange(width,height),
//				cl::NullRange,
//				NULL,
//				&event);
//
//		//swap in and out buffer pointers and run a 3rd time
//		kernel.setArg(0,bufferIn);
//		kernel.setArg(1,bufferOut);
//		queue.enqueueNDRangeKernel(	kernel,
//				cl::NullRange,
//				cl::NDRange(width,height),
//				cl::NullRange,
//				NULL,
//				&event);
//
//		queue.finish();
//
//		stopTimer1 = clock();
//		double elapse = 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC;
//		info[2] = (int)elapse;
//		LOGI("OpenCL code on the GPU took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
//
//		queue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, imageSize, bufOut);
//	}
//	catch (cl::Error err) {
//		LOGE("ERROR: %s\n", err.what());
//	}
//	return;
//}
//
//*/
