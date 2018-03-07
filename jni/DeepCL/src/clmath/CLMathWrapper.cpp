// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/CLFloatWrapper.h"
#include "../util/stringhelper.h"
#include "GpuOp.h"
#include "CLMathWrapper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL CLMathWrapper::~CLMathWrapper() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: ~CLMathWrapper");
#endif


    delete gpuOp;
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator=(const float scalar) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: operator=");
#endif


//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    Op2Equal op;
    gpuOp->apply2_inplace(N, wrapper, scalar, &op);
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator*=(const float scalar) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: operator*=");
#endif


//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    Op2Mul op;
    gpuOp->apply2_inplace(N, wrapper, scalar, &op);
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator+=(const float scalar) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: operator+=");
#endif


//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    Op2Add op;
    gpuOp->apply2_inplace(N, wrapper, scalar, &op);
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator*=(const CLMathWrapper &two) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: operator*=");
#endif


//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    if(two.N != N) {
        throw runtime_error("CLMathWrapper::operator+, array size mismatch, cannot assign " + toString(two.N) + 
            " vs " + toString(N) );
    }
    Op2Mul op;
    gpuOp->apply2_inplace(N, wrapper, ((CLMathWrapper &)two).wrapper, &op);
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator+=(const CLMathWrapper &two) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: operator+=");
#endif


//    cout << "CLMathWrapper.operator+=()" << endl;
    if(two.N != N) {
        throw runtime_error("CLMathWrapper::operator+, array size mismatch, cannot assign " + toString(two.N) + 
            " vs " + toString(N) );
    }
    Op2Add op;
    gpuOp->apply2_inplace(N, wrapper, ((CLMathWrapper &)two).wrapper, &op);
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator=(const CLMathWrapper &rhs) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: operator=");
#endif


//    cout << "CLMathWrapper.operator=()" << endl;
    if(rhs.N != N) {
        throw runtime_error("CLMathWrapper::operator= array size mismatch, cannot assign " + toString(rhs.N) + 
            " vs " + toString(N) );
    }
    Op2Equal op;
    gpuOp->apply2_inplace(N, wrapper, ((CLMathWrapper &)rhs).wrapper, &op);
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::sqrt() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: sqrt");
#endif


    Op1Sqrt op;
    gpuOp->apply1_inplace(N, wrapper, &op);
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::inv() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: inv");
#endif


    Op1Inv op;
    gpuOp->apply1_inplace(N, wrapper, &op);
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::squared() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: squared");
#endif


    Op1Squared op;
    gpuOp->apply1_inplace(N, wrapper, &op);
    return *this;
}
VIRTUAL void CLMathWrapper::runKernel(CLKernel *kernel) {   
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: runKernel");
#endif


    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    cl->finish();
}
CLMathWrapper::CLMathWrapper(CLWrapper *wrapper) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/clmath/CLMathWrapper.cpp: CLMathWrapper");
#endif


    CLFloatWrapper *floatWrapper = dynamic_cast< CLFloatWrapper * >(wrapper);
    if(floatWrapper == 0) {
        throw runtime_error("CLMathWrapper only works on CLFloatWrapper objects");
    }
    this->cl = floatWrapper->getCl();
    this->wrapper = floatWrapper;
    this->N = floatWrapper->size();
    this->gpuOp = new GpuOp(cl);
}

