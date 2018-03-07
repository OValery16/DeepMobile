// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../../../sonyOpenCLexample1.h"
#include "../net/Trainable.h"

#include "../DeepCLDllExport.h"
#include "../trainers/trainer.h"
#include <boost/iostreams/device/mapped_file.hpp>

class EpochResult;
class NetAction;
class Trainer;
#include "../trainers/TrainingContext.h"

#define VIRTUAL virtual
#define STATIC static

/// \brief Runs an epoch for a single set of already-loaded data
///
/// This class is responsible for running all batches once (ie one 'epoch'), but
/// for a single set of already-loaded data
/// If you want to have a class that loads the data in chunks, then you'll need
/// OnDemandBatcher
/// Note however that, what OnDemandBatcher does is, it loads in some data, and
/// then it calls this Batcher class.  so they work together
PUBLICAPI
class DeepCL_EXPORT Batcher {
protected:
    Trainable *net;
    int batchSize;
    int N;
    float const* data;
    int const* labels;
    bool trainOrTest;

    /////////////////
    float const* dataTest;
    int const* labelTest;
    boost::iostreams::mapped_file file3;
    boost::iostreams::mapped_file file4;

////////////////////

    int numBatches;
    int inputCubeSize;

    bool epochDone;
    int nextBatch;
    int numRight;
    float loss;

    BatchResult epochResult;

public:
    virtual void internalTick(int epoch, float const*batchData, int const*batchLabels) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels,bool trainOrTest);
    VIRTUAL ~Batcher();
    PUBLICAPI void reset();
    PUBLICAPI int getNextBatch();
    PUBLICAPI VIRTUAL float getLoss();
    PUBLICAPI VIRTUAL int getNumRight();
    PUBLICAPI VIRTUAL int getN();
    PUBLICAPI VIRTUAL bool getEpochDone();
    VIRTUAL void setBatchState(int nextBatch, int numRight, float loss);
    VIRTUAL void setN(int N);
    PUBLICAPI bool tick(int epoch);
    PUBLICAPI EpochResult run(int epoch);

    // [[[end]]]
};

class DeepCL_EXPORT LearnBatcher : public Batcher {
public:
    Trainer *trainer; // NOT delete
    TrainingContext *context; // NOT delete

    LearnBatcher(Trainer *trainer, 
        Trainable *net, int batchSize, int N, float *data, int const*labels);
    virtual void internalTick(int epoch, float const*batchData, int const*batchLabels);
};

//class DeepCL_EXPORT LearnFromExpectedBatcher : public Batcher {
//public:
//    Trainer *trainer; // NOT delete
//    TrainingContext *context; // NOT delete

//    LearnFromExpectedBatcher(Trainer *trainer, 
//        Trainable *net, int batchSize, int N, float *data, float *expectedOutputs);
//    virtual void internalTick(int epoch, float const*batchData, float *expectedOutputs);
//};

class DeepCL_EXPORT NetActionBatcher : public Batcher {
public:
    NetAction * netAction;
    NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction * netAction);
    virtual void internalTick(int epoch, float const*batchData, int const*batchLabels);
};


class DeepCL_EXPORT ForwardBatcher : public Batcher {
public:
    ForwardBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels);
    virtual void internalTick(int epoch, float const*batchData, int const*batchLabels);
};


