//
//  openCLNR.h
//  OpenCL Example1
//
//  Created by Rasmusson, Jim on 18/03/13.
//
//  Copyright (c) 2013, Sony Mobile Communications AB
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of Sony Mobile Communications AB nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef TESTSINGLEBATCH_H
#define TESTSINGLEBATCH_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>


#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "sonyOpenCLexample1.h"

class TestArgs{
public:
    static TestArgs instance(){ TestArgs args; return args; };
    // [[[cog
    // floats= ['learningRate']
    // ints = ['batchSize','imageSize','numLayers','filterSize','numFilters', 'numEpochs', 'poolingSize',
    // 'numCats', 'softMax' ]
    // import cog_fluent
    // cog_fluent.go1b('TestArgs', ints = ints, floats = floats)
    // ]]]
    // generated, using cog:
    int batchSize;
    int imageSize;
    int numLayers;
    int filterSize;
    int numFilters;
    int numEpochs;
    int poolingSize;
    int numCats;
    int softMax;
    float learningRate;
    TestArgs() {
        batchSize = 0;
        imageSize = 0;
        numLayers = 0;
        filterSize = 0;
        numFilters = 0;
        numEpochs = 0;
        poolingSize = 0;
        numCats = 0;
        softMax = 0;
        learningRate = 0;
    }
    TestArgs BatchSize( int _batchSize ) {
        this->batchSize = _batchSize;
        return *this;
    }
    TestArgs ImageSize( int _imageSize ) {
        this->imageSize = _imageSize;
        return *this;
    }
    TestArgs NumLayers( int _numLayers ) {
        this->numLayers = _numLayers;
        return *this;
    }
    TestArgs FilterSize( int _filterSize ) {
        this->filterSize = _filterSize;
        return *this;
    }
    TestArgs NumFilters( int _numFilters ) {
        this->numFilters = _numFilters;
        return *this;
    }
    TestArgs NumEpochs( int _numEpochs ) {
        this->numEpochs = _numEpochs;
        return *this;
    }
    TestArgs PoolingSize( int _poolingSize ) {
        this->poolingSize = _poolingSize;
        return *this;
    }
    TestArgs NumCats( int _numCats ) {
        this->numCats = _numCats;
        return *this;
    }
    TestArgs SoftMax( int _softMax ) {
        this->softMax = _softMax;
        return *this;
    }
    TestArgs LearningRate( float _learningRate ) {
        this->learningRate = _learningRate;
        return *this;
    }
    // [[[end]]]
};

void essai();

#endif
