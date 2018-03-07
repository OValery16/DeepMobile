// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../net/NeuralNet.h"
#include "../util/stringhelper.h"
#include "Trainer.h"
#include "../net/MultiNet.h"
#include "../batch/NetAction.h"
#include "TrainerStateMaker.h"
#include "TrainerState.h"
#include "../layer/Layer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


Trainer::Trainer(EasyCL *cl) :
    cl(cl),
    learningRate(0) {
}
VIRTUAL Trainer::~Trainer() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: ~Trainer");
#endif


}
VIRTUAL void Trainer::setLearningRate(float learningRate) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: setLearningRate");
#endif


    this->learningRate = learningRate;
}
VIRTUAL std::string Trainer::asString() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: string Trainer::asString");
#endif


    return "Trainer{ learningRate=" + toString(learningRate) + " }";
}
VIRTUAL BatchResult Trainer::train(Trainable *trainable, 
        TrainingContext *context,
        float const*input, float const*expectedOutput) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: train");
#endif
    MultiNet *multiNet = dynamic_cast< MultiNet *>(trainable);
    float loss = 0;
    if(multiNet != 0) {
		#if DEEPCL_VERBOSE == 1
    		LOGI( "DeepCL/src/trainers/Trainer.cpp netnb=%d", multiNet->getNumNets());
		#endif
        for(int i = 0; i < multiNet->getNumNets(); i++) {
            Trainable *child = multiNet->getNet(i);
            BatchResult result = this->train(child, context, input, expectedOutput);
            loss += result.loss;
        }
    } else {
		#if DEEPCL_VERBOSE == 1
    		LOGI( "DeepCL/src/trainers/Trainer.cpp creation of net");
		#endif
        NeuralNet *net = dynamic_cast< NeuralNet * > (trainable);
        return this->trainNet(net, context, input, expectedOutput);
    }
    return BatchResult(loss, 0);
}
VIRTUAL BatchResult Trainer::trainFromLabels(Trainable *trainable,
    TrainingContext *context,
    float const*input, int const*labels) {
	//olivier: computing path goes here
	#if DEEPCL_VERBOSE == 1
	LOGI( "DeepCL/src/trainers/Trainer.cpp: trainFromLabels");
	#endif
    MultiNet *multiNet = dynamic_cast< MultiNet *>(trainable);
    float loss = 0;
    int numRight = 0;
    if(multiNet != 0) {
        for(int i = 0; i < multiNet->getNumNets(); i++) {
            Trainable *child = multiNet->getNet(i);
            BatchResult result = this->trainFromLabels(child, context, input, labels);
            loss += result.loss;
            numRight += result.numRight;
        }
    } else {
        NeuralNet *net = dynamic_cast< NeuralNet * > (trainable);
        return this->trainNetFromLabels(net, context, input, labels);
    }
    return BatchResult(loss, numRight);
}
VIRTUAL void Trainer::_bindState(NeuralNet *net, TrainerStateMaker *stateMaker) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: _bindState");
#endif


    // go through network layers, and assign TrainerState objects
    for(int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++) {
        Layer *layer = net->getLayer(layerIdx);
        if(layer->needsTrainerState()) {
            TrainerState *state = layer->getTrainerState();
            if(!stateMaker->created(state) ) {
                layer->setTrainerState(stateMaker);
            }
        }
    }
}

