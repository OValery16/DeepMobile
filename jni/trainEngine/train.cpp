// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


//#include <iostream>
//#include <algorithm>

#include "../DeepCL/src/DeepCL.h"
//#include "test/Sampler.h"  // TODO: REMOVE THIS
//#include "../DeepCL/src/clblas/ClBlasInstance.h"
//#include "../sonyOpenCLexample1.h"
#include "train.h"


 #include <unistd.h>


using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    # format:
    # (name, type, description, default, ispublicapi)
    options = [
        ('gpuIndex', 'int', 'gpu device index; default value is gpu if present, cpu otw.', -1, True),
        ('dataDir', 'string', 'directory to search for train and validate files', '../data/mnist', True),
        ('trainFile', 'string', 'path to training data file',"train-images-idx3-ubyte", True),
        ('dataset', 'string', 'choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]','', True),
        ('validateFile', 'string', 'path to validation data file',"t10k-images-idx3-ubyte", True),
        ('numTrain', 'int', 'num training examples',-1, True),
        ('numTest', 'int', 'num test examples]',-1, True),
        ('batchSize', 'int', 'batch size',128, True),
        ('numEpochs', 'int', 'number epochs',12, True),
        ('netDef', 'string', 'network definition',"rt2-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n", True),
        ('loadWeights', 'int', 'load weights from file at startup?', 0, True),
        ('weightsFile', 'string', 'file to write weights to','weights.dat', True),
        ('writeWeightsInterval', 'float', 'write weights every this many minutes', 0, True),
        ('normalization', 'string', '[stddev|maxmin]', 'stddev', True),
        ('normalizationNumStds', 'float', 'with stddev normalization, how many stddevs from mean is 1?', 2.0, True),
        ('dumpTimings', 'int', 'dump detailed timings each epoch? [1|0]', 0, True),
        ('multiNet', 'int', 'number of Mcdnn columns to train', 1, True),
        ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, True),
        ('fileReadBatches', 'int', 'how many batches to read from file each time? (for loadondemand=1)', 50, True),
        ('normalizationExamples', 'int', 'number of examples to read to determine normalization parameters', 10000, True),
        ('weightsInitializer', 'string', 'initializer for weights, choices: original, uniform (default: original)', 'original', True),
        ('initialWeights', 'float', 'for uniform initializer, weights will be initialized randomly within range -initialweights to +initialweights, divided by fanin, (default: 1.0f)', 1.0, False),
        ('trainer', 'string', 'which trainer, sgd, anneal, nesterov, adagrad, rmsprop, or adadelta (default: sgd)', 'sgd', True),
        ('learningRate', 'float', 'learning rate, a float value, used by all trainers', 0.002, True),
        ('rho', 'float', 'rho decay, in adadelta trainer. 1 is no decay. 0 is full decay (default 0.9)', 0.9, False),
        ('momentum', 'float', 'momentum, used by sgd and nesterov trainers', 0.0, True),
        ('weightDecay', 'float', 'weight decay, 0 means no decay; 1 means full decay, used by sgd trainer', 0.0, True),
        ('anneal', 'float', 'multiply learningrate by this amount each epoch, used by anneal trainer, default 1.0', 1.0, False)
    ]
*///]]]
// [[[end]]]

ConfigTraining::ConfigTraining() {

    gpuIndex = -1;
    dataDir = "/data/data/com.sony.openclexample1/app_execdir";
    trainFile = "train-images-idx3-ubyte";
    dataset = "";
    validateFile = "t10k-images-idx3-ubyte";
    numTrain = -1;
    numTest = -1;
    batchSize = 128;
    numEpochs = 12;
    netDef = "rt2-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n";
    loadWeights = 0;
    weightsFile = "/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//weightstToDelete.dat";//;
    writeWeightsInterval = 0.0f;
    normalization = "stddev";
    normalizationNumStds = 2.0f;
    dumpTimings = 0;
    multiNet = 1;
    loadOnDemand = 0;
    fileReadBatches = 128;
    normalizationExamples = 10000;
    weightsInitializer = "original";
    initialWeights = 1.0f;
    trainer = "sgd";
    learningRate = 0.002f;
    rho = 0.9f;
    momentum = 0.0f;
    weightDecay = 0.0f;
    anneal = 1.0f;
        // [[[end]]]

    }
string ConfigTraining::getTrainingString() {
        string configString = "";
        configString += "netDef=" + netDef; // lets just force that at least
                   // need same network structure, otherwise weights wont
                   // really make sense at all.  Evreything else is up to the
                   // end-user plausibly?
        return configString;
    }

string ConfigTraining::getOldTrainingString() {
        string configString = "";
        configString += "netDef=" + netDef + " trainFile=" + trainFile;
        return configString;
    }

TrainModel::TrainModel(){

//    if(config.gpuIndex >= 0) {
//        cl = EasyCL::createForIndexedGpu(config.gpuIndex);
//    } else {
        cl = EasyCL::createForFirstGpuOtherwiseCpu();
//    }

}

void TrainModel::go(ConfigTraining config) {
	    Timer timer;
	    GenericLoaderv2 *trainLoader=0;
	    GenericLoaderv2 *testLoader=0;
	    int Ntrain;
	    int Ntest;
	    int numPlanes;
	    int imageSize;

	    float *trainData = 0;
	    float *testData = 0;
	    int *trainLabels = 0;
	    int *testLabels = 0;

	    int trainAllocateN = 0;
	    int testAllocateN = 0;

	    if(config.dumpTimings) {
	        StatefulTimer::setEnabled(true);
	    }
	    cout << "Statefultimer enabled: " << StatefulTimer::enabled << endl;
	    LOGI("loading training data ...");
	//    int totalLinearSize;

	    Ntrain = 128;//2048;//2048;//2000;//                  2048;//trainLoader->getN();
	    numPlanes = 1;//3;//                  trainLoader->getPlanes();
	    imageSize = 28;//32;//                  trainLoader->getImageSize();
	    trainAllocateN=2048;//2048;//60000;//2048;//2000;//                  2048;
	    // GenericLoader::getDimensions(, &Ntrain, &numPlanes, &imageSize);
	    //Ntrain = config.numTrain == -1 ? Ntrain : config.numTrain;
	//    long allocateSing)Ntrain * numPlanes * imageSize * imageSize;
	    //cout << "Ntrain " << Ntrain << " numPlanes " << numPlanes << " imageSize " << imageSize << endl;
	    LOGI("Ntrain %d numPlanes %d imageSize %d",Ntrain,numPlanes,imageSize);
	    if(config.loadOnDemand) {
	        trainAllocateN = config.batchSize; // can improve this later
	    } else {
	        trainAllocateN = Ntrain;
	    }
	    string test=config.dataDir + "/" + config.trainFile;
	    LOGI("string %s",test.c_str());
		#if MEMORY_MAP_FILE_LOADING==0
	    	trainLoader=new GenericLoaderv2(config.dataDir + "/" + config.trainFile);

	   	    Ntrain = trainLoader->getN();
	   	    numPlanes = trainLoader->getPlanes();
	   	    imageSize = trainLoader->getImageSize();
	   	    // GenericLoader::getDimensions(, &Ntrain, &numPlanes, &imageSize);
	   	    Ntrain = config.numTrain == -1 ? Ntrain : config.numTrain;
	   	//    long allocateSize = (long)Ntrain * numPlanes * imageSize * imageSize;
	   	    //cout << "Ntrain " << Ntrain << " numPlanes " << numPlanes << " imageSize " << imageSize << endl;
	   	    LOGI("Ntrain %d numPlanes %d imageSize %d",Ntrain,numPlanes,imageSize);
	   	    if(config.loadOnDemand) {
	   	        trainAllocateN = config.batchSize; // can improve this later
	   	    } else {
	   	        trainAllocateN = Ntrain;
	   	    }
			LOGI("1");
			trainData = new float[ (long)trainAllocateN * numPlanes * imageSize * imageSize ];
			trainLabels = new int[trainAllocateN];
			if(!config.loadOnDemand && Ntrain > 0) {
				LOGI("config.loadOnDemand %d",config.loadOnDemand);
				trainLoader->load(trainData, trainLabels, 0, Ntrain);
			}
			LOGI("loading validation data ...");
		#endif




		#if NO_POSTPROCESSING==0
	    	testLoader=new GenericLoaderv2(config.dataDir + "/" + config.validateFile);
			Ntest = testLoader->getN();
			numPlanes = testLoader->getPlanes();
			imageSize = testLoader->getImageSize();
			Ntest = config.numTest == -1 ? Ntest : config.numTest;
			if(config.loadOnDemand) {
				testAllocateN = config.batchSize; // can improve this later
			} else {
				testAllocateN = Ntest;
			}
			testData = new float[ (long)testAllocateN * numPlanes * imageSize * imageSize ];
			testLabels = new int[testAllocateN];
			if(!config.loadOnDemand && Ntest > 0) {
				testLoader->load(testData, testLabels, 0, Ntest);
			}

	    cout << "Ntest " << Ntest << " Ntest" << endl;

	    timer.timeCheck("after load images");
		#endif
	    const int inputCubeSize = numPlanes * imageSize * imageSize;
	    float translate;
	    float scale;
#if MEMORY_MAP_FILE_LOADING==0
        int normalizationExamples = config.normalizationExamples > Ntrain ? Ntrain : config.normalizationExamples;
	    if(!config.loadOnDemand) {
	        if(config.normalization == "stddev") {
	            float mean, stdDev;
	            NormalizationHelper::getMeanAndStdDev(trainData, normalizationExamples * inputCubeSize, &mean, &stdDev);
	            cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
	            translate = - mean;
	            scale = 1.0f / stdDev / config.normalizationNumStds;
	        } else if(config.normalization == "maxmin") {
	            float mean, stdDev;
	            NormalizationHelper::getMinMax(trainData, normalizationExamples * inputCubeSize, &mean, &stdDev);
	            translate = - mean;
	            scale = 1.0f / stdDev;
	        } else {
	            cout << "Error: Unknown normalization: " << config.normalization << endl;
	            return;
	        }
	    } else {
	        if(config.normalization == "stddev") {
	            float mean, stdDev;
	            NormalizeGetStdDev normalizeGetStdDev(trainData, trainLabels);
	            BatchProcessv2::run(&trainLoader, 0, config.batchSize, normalizationExamples, inputCubeSize, &normalizeGetStdDev);
	            normalizeGetStdDev.calcMeanStdDev(&mean, &stdDev);
	            cout << " image stats mean " << mean << " stdDev " << stdDev << endl;
	            translate = - mean;
	            scale = 1.0f / stdDev / config.normalizationNumStds;
	        } else if(config.normalization == "maxmin") {
	            NormalizeGetMinMax normalizeGetMinMax(trainData, trainLabels);
	            BatchProcessv2::run(&trainLoader, 0, config.batchSize, normalizationExamples, inputCubeSize, &normalizeGetMinMax);
	            normalizeGetMinMax.calcMinMaxTransform(&translate, &scale);
	        } else {
	            cout << "Error: Unknown normalization: " << config.normalization << endl;
	            return;
	        }
	    }
#endif


	    /////////////////////////
#if MEMORY_MAP_FILE_LOADING==1
	#if 0
			ofstream myfile ("/data/data/com.sony.openclexample1/preloadingData/normalization.txt");
			  if (myfile.is_open())
			  {
				myfile << to_string(translate)<<"\n"<< to_string(scale);
				myfile.close();
			  }
	#endif

			string line;
			ifstream myfile ("/data/data/com.sony.openclexample1/preloadingData/normalization.txt");
			if (myfile.is_open())
			{
				getline (myfile,line);
				translate=::atof(line.c_str());
				getline (myfile,line);
				scale=::atof(line.c_str());
				myfile.close();
			}
#endif



	    NeuralNet *net;
	    net = new NeuralNet(cl);

	    WeightsInitializer *weightsInitializer = 0;
	    if(toLower(config.weightsInitializer) == "original") {
	    	LOGI( "DeepCL/src/main/train.cpp: weights Initialization --> original");
	        weightsInitializer = new OriginalInitializer();
	    } else if(toLower(config.weightsInitializer) == "uniform") {
	    	LOGI( "DeepCL/src/main/train.cpp: weights Initialization --> uniform");
	        weightsInitializer = new UniformInitializer(config.initialWeights);
	    } else {
	        cout << "Unknown weights initializer " << config.weightsInitializer << endl;
	        return;
	    }
	//    net->inputMaker<unsigned char>()->numPlanes(numPlanes)->imageSize(imageSize)->insert();
	    net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
	    net->addLayer(NormalizationLayerMaker::instance()->translate(translate)->scale(scale)->batch(config.batchSize));
	    if(!NetdefToNet::createNetFromNetdef(config.batchSize,net, config.netDef, weightsInitializer)) {
	    	LOGE( "DeepCL/src/main/train.cpp: neural network creation failed");
	        return;
	    }
	    // apply the trainer
	    Trainer *trainer = 0;
	    if(toLower(config.trainer) == "sgd") {
	    	LOGI( "DeepCL/src/main/train.cpp: trainer --> sgd");
	        SGD *sgd = new SGD(cl);
	        sgd->setLearningRate(config.learningRate);
	        sgd->setMomentum(config.momentum);
	        sgd->setWeightDecay(config.weightDecay);
	        trainer = sgd;
	    } else if(toLower(config.trainer) == "anneal") {
	    	LOGI( "DeepCL/src/main/train.cpp: trainer --> anneal");
	        Annealer *annealer = new Annealer(cl);
	        annealer->setLearningRate(config.learningRate);
	        annealer->setAnneal(config.anneal);
	        trainer = annealer;
	    } else if(toLower(config.trainer) == "nesterov") {
	    	LOGI( "DeepCL/src/main/train.cpp: trainer --> nesterov");
	        Nesterov *nesterov = new Nesterov(cl);
	        nesterov->setLearningRate(config.learningRate);
	        nesterov->setMomentum(config.momentum);
	        trainer = nesterov;
	    } else if(toLower(config.trainer) == "adagrad") {
	    	LOGI( "DeepCL/src/main/train.cpp: trainer --> adagrad");
	        Adagrad *adagrad = new Adagrad(cl);
	        adagrad->setLearningRate(config.learningRate);
	        trainer = adagrad;
	    } else if(toLower(config.trainer) == "rmsprop") {
	    	LOGI( "DeepCL/src/main/train.cpp: trainer --> rmsprop");
	        Rmsprop *rmsprop = new Rmsprop(cl);
	        rmsprop->setLearningRate(config.learningRate);
	        trainer = rmsprop;
	    } else if(toLower(config.trainer) == "adadelta") {
	    	LOGI( "DeepCL/src/main/train.cpp: trainer --> adadelta");
	        Adadelta *adadelta = new Adadelta(cl, config.rho);
	        trainer = adadelta;
	    } else {
	    	LOGE( "DeepCL/src/main/train.cpp: trainer --> unknown");
	        //cout << "trainer " << config.trainer << " unknown." << endl;
	        return;
	    }
	    //cout << "Using trainer " << trainer->asString() << endl;
	//    trainer->bindTo(net);
	//    net->setTrainer(trainer);
	    net->setBatchSize(config.batchSize);
	    net->print();

	    bool afterRestart = false;
	    int restartEpoch = 0;
	    int restartBatch = 0;
	    float restartAnnealedLearningRate = 0;
	    int restartNumRight = 0;
	    float restartLoss = 0;
	    if(config.loadWeights && config.weightsFile != "") {
	    	LOGI( "DeepCL/src/main/train.cpp:  loadingweights");
	        //cout << "loadingweights" << endl;
	        afterRestart = WeightsPersister::loadWeights(config.weightsFile, config.getTrainingString(), net, &restartEpoch, &restartBatch, &restartAnnealedLearningRate, &restartNumRight, &restartLoss);
	        if(!afterRestart && FileHelper::exists(config.weightsFile)) {


	            // try old trainingstring
	            afterRestart = WeightsPersister::loadWeights(config.weightsFile, config.getOldTrainingString(), net, &restartEpoch, &restartBatch, &restartAnnealedLearningRate, &restartNumRight, &restartLoss);
	        }
	        if(!afterRestart && FileHelper::exists(config.weightsFile)) {



	            cout << "Weights file " << config.weightsFile << " exists, but doesnt match training options provided." << endl;
	            cout << "Continue loading anyway (might crash, or weights might be completely inappropriate)? (y/n)" << endl;
	            string response;
	            cin >> response;
	            if(response != "y") {
	                cout << "Please either check the training options, or choose a weights file that doesnt exist yet" << endl;
	                return;
	            }
	        }
	        cout << "reloaded epoch=" << restartEpoch << " batch=" << restartBatch << " numRight=" << restartNumRight << " loss=" << restartLoss << endl;
	    }

	    timer.timeCheck("before learning start");
	    if(config.dumpTimings) {
	        StatefulTimer::dump(true);
	    }
	    StatefulTimer::timeCheck("START");

	    Trainable *trainable = net;
	    MultiNet *multiNet = 0;
	    if(config.multiNet > 1) {
	        multiNet = new MultiNet(config.multiNet, net);
	        trainable = multiNet;
	    }
	    NetLearnerBase *netLearner = 0;
	    if(config.loadOnDemand) {
	        netLearner = new NetLearnerOnDemandv2(trainer, trainable,
	            trainLoader, Ntrain,
	            testLoader, Ntest,
	            config.fileReadBatches, config.batchSize
	        );
	    } else {
	        netLearner = new NetLearner(trainer, trainable,
	            Ntrain, trainData, trainLabels,
	            Ntest, testData, testLabels,
	            config.batchSize
	        );
	    }
	//    netLearner->setTrainer(trainer);
	    netLearner->reset();
	    netLearner->setSchedule(config.numEpochs, afterRestart ? restartEpoch : 0);
	    if(afterRestart) {
	        netLearner->setBatchState(restartBatch, restartNumRight, restartLoss);
	    }
	    netLearner->setDumpTimings(config.dumpTimings);
	//    netLearner->setLearningRate(config.learningRate, config.annealLearningRate);
	    Timer weightsWriteTimer;
	    int i=0;

//	    ///////////
//	    clock_t startTimer1, stopTimer1;
//	    startTimer1=clock();
//	    //////////
		struct timeval start, end;
		/*get the start time*/
		gettimeofday(&start, NULL);
	    while(!netLearner->isLearningDone()) {//i<100){//
	//        netLearnerBase->tickEpoch();
	    	i++;
	        netLearner->tickBatch();
//	        //LOGI("loss %f",netLearner->getBatchLoss());
//	        if(netLearner->getEpochDone()) {
//	//            cout << "epoch done" << endl;
//	            if(config.weightsFile != "") {
//	            	LOGI("record epoch= %d", netLearner->getNextEpoch());
//	                // to restore
////	                WeightsPersister::persistWeights(config.weightsFile, config.getTrainingString(), net, netLearner->getNextEpoch(), 0, 0, 0, 0);
////	                weightsWriteTimer.lap();
//	            }
//
//
//	//            Sampler::sampleFloatWrapper("conv weights", net->getLayer(6)->getWeightsWrapper());
//	//            Sampler::sampleFloatWrapper("fc weights", net->getLayer(11)->getWeightsWrapper());
//	//            Sampler::sampleFloatWrapper("conv bias", net->getLayer(6)->getBiasWrapper());
//	//            Sampler::sampleFloatWrapper("fc bias", net->getLayer(11)->getBiasWrapper());
//	            if(config.dumpTimings) {
//	                StatefulTimer::dump(true);
//	            }
//	        } /*else {
//	            if(config.writeWeightsInterval > 0) {
//
//	//                cout << "batch done" << endl;
//	                float timeMinutes = weightsWriteTimer.interval() / 1000.0f / 60.0f;
//	//                cout << "timeMinutes " << timeMinutes << endl;
//	                if(timeMinutes >= config.writeWeightsInterval) {
//	                    int nextEpoch = netLearner->getNextEpoch();
//	                    int nextBatch = netLearner->getNextBatch();
//	                    int batchNumRight = netLearner->getBatchNumRight();
//	                    float batchLoss = netLearner->getBatchLoss();
//	                    cout << "record epoch=" << nextEpoch << " batch=" << nextBatch <<
//	                        "(" << ((float)nextBatch * 100.0f / netLearner->getNTrain() * config.batchSize) << "% of epoch)" <<
//	                        " numRight=" << batchNumRight << "(" << (batchNumRight * 100.0f / nextBatch / config.batchSize) << "%)" <<
//	                        " loss=" << batchLoss << endl;
//	                    WeightsPersister::persistWeights(config.weightsFile, config.getTrainingString(), net,
//	                        nextEpoch, nextBatch, 0, batchNumRight, batchLoss);
//	                    weightsWriteTimer.lap();
//	                }
//	            }
//	        }*/
	    }
	    (dynamic_cast<NeuralNet*>(net))->cl->finish();
	    WeightsPersister::persistWeights(config.weightsFile, config.getTrainingString(), net, netLearner->getNextEpoch(), 0, 0, 0, 0);
	    weightsWriteTimer.lap();
		gettimeofday(&end, NULL);
		/*Print the amount of time taken to execute*/
		LOGI("gettimeofday %f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));


////	    /////////////////
////	    ofstream myfile2 ("/data/data/com.sony.openclexample1/app_execdir/testDate.txt");
//	    time_t now = time(0);
////
////	    // convert now to string form
//	    char* dt = ctime(&now);
//	    LOGI("end %s",dt);
////	    if (myfile2.is_open())
////	    {
////		  myfile2 << dt ;
////		  myfile2.close();
////	    }
////	    ///////////////
//
//
//	    stopTimer1 = clock();
//	    LOGI("OpenCL code on the GPU took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
	    delete cl;
	    LOGI("i=%d",i);
	    LOGI("End of ther training: Delete object");
	    delete weightsInitializer;
	    LOGI("Delete ---- weightsInitializer ");
	    delete trainer;
	    LOGI("Delete ---- trainer ");
	    delete netLearner;
	    LOGI("Delete ---- netLearner ");
	    if(multiNet != 0) {
	        delete multiNet;
	    }
	    LOGI("Delete ---- multiNet ");
	    delete net;
	    LOGI("Delete ---- net ");
#if MEMORY_MAP_FILE_LOADING==0
	    delete trainLoader;
	    if(trainData != 0) {
	        delete[] trainData;
	    }LOGI("Delete ---- trainData ");
	    if(trainLabels != 0) {
	        delete[] trainLabels;
	    }LOGI("Delete ---- trainLabels ");
#endif
#if NO_POSTPROCESSING==0
	    delete testLoader;
	    if(testData != 0) {
	        delete[] testData;
	    }LOGI("Delete ---- testData ");
	    if(testLabels != 0) {
	        delete[] testLabels;
	    }LOGI("Delete ---- testLabels ");
#endif

	    LOGI("Delete ---- trainLoader ");
	    //(dynamic_cast<NeuralNet*>(net))->cl->finish();
	}

void TrainModel::printUsage(char *argv[], ConfigTraining config) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    /* [[[cog
        cog.outl('// generated using cog:')
        cog.outl('cout << "public api, shouldnt change within major version:" << endl;')
        for (name,type,description,_, is_public_api) in options:
            if is_public_api:
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
        cog.outl('cout << "" << endl; ')
        cog.outl('cout << "unstable, might change within major version:" << endl; ')
        for (name,type,description,_, is_public_api) in options:
            if not is_public_api:
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
    *///]]]
    // generated using cog:
    cout << "public api, shouldnt change within major version:" << endl;
    cout << "    gpuindex=[gpu device index; default value is gpu if present, cpu otw.] (" << config.gpuIndex << ")" << endl;
    cout << "    datadir=[directory to search for train and validate files] (" << config.dataDir << ")" << endl;
    cout << "    trainfile=[path to training data file] (" << config.trainFile << ")" << endl;
    cout << "    dataset=[choose datadir,trainfile,and validatefile for certain datasets [mnist|norb|kgsgo|cifar10]] (" << config.dataset << ")" << endl;
    cout << "    validatefile=[path to validation data file] (" << config.validateFile << ")" << endl;
    cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
    cout << "    numtest=[num test examples]] (" << config.numTest << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
    cout << "    netdef=[network definition] (" << config.netDef << ")" << endl;
    cout << "    loadweights=[load weights from file at startup?] (" << config.loadWeights << ")" << endl;
    cout << "    weightsfile=[file to write weights to] (" << config.weightsFile << ")" << endl;
    cout << "    writeweightsinterval=[write weights every this many minutes] (" << config.writeWeightsInterval << ")" << endl;
    cout << "    normalization=[[stddev|maxmin]] (" << config.normalization << ")" << endl;
    cout << "    normalizationnumstds=[with stddev normalization, how many stddevs from mean is 1?] (" << config.normalizationNumStds << ")" << endl;
    cout << "    dumptimings=[dump detailed timings each epoch? [1|0]] (" << config.dumpTimings << ")" << endl;
    cout << "    multinet=[number of Mcdnn columns to train] (" << config.multiNet << ")" << endl;
    cout << "    loadondemand=[load data on demand [1|0]] (" << config.loadOnDemand << ")" << endl;
    cout << "    filereadbatches=[how many batches to read from file each time? (for loadondemand=1)] (" << config.fileReadBatches << ")" << endl;
    cout << "    normalizationexamples=[number of examples to read to determine normalization parameters] (" << config.normalizationExamples << ")" << endl;
    cout << "    weightsinitializer=[initializer for weights, choices: original, uniform (default: original)] (" << config.weightsInitializer << ")" << endl;
    cout << "    trainer=[which trainer, sgd, anneal, nesterov, adagrad, rmsprop, or adadelta (default: sgd)] (" << config.trainer << ")" << endl;
    cout << "    learningrate=[learning rate, a float value, used by all trainers] (" << config.learningRate << ")" << endl;
    cout << "    momentum=[momentum, used by sgd and nesterov trainers] (" << config.momentum << ")" << endl;
    cout << "    weightdecay=[weight decay, 0 means no decay; 1 means full decay, used by sgd trainer] (" << config.weightDecay << ")" << endl;
    cout << "" << endl;
    cout << "unstable, might change within major version:" << endl;
    cout << "    initialweights=[for uniform initializer, weights will be initialized randomly within range -initialweights to +initialweights, divided by fanin, (default: 1.0f)] (" << config.initialWeights << ")" << endl;
    cout << "    rho=[rho decay, in adadelta trainer. 1 is no decay. 0 is full decay (default 0.9)] (" << config.rho << ")" << endl;
    cout << "    anneal=[multiply learningrate by this amount each epoch, used by anneal trainer, default 1.0] (" << config.anneal << ")" << endl;
    // [[[end]]]
}

TrainModel::~TrainModel(){
	LOGI( "easyCL oject destroyed");
//	delete cl;
    //delete cl;
}

int TrainModel::trainCmd(std::string argument){

	//prepareFiles("/data/data/com.sony.openclexample1/app_execdir//train-images-idx3-ubyte",2048, 1,"/data/data/com.sony.openclexample1/memMapFileData.raw","/data/data/com.sony.openclexample1/memMapFileLabel.raw");

    istringstream iss(argument);
    vector<string> tokens;
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter(tokens));
    for(int i =0;i<tokens.size();i++){
    	LOGI ("%s\n",tokens[i].c_str());
    }
    char** argList = new char*[tokens.size()];
    for(int i = 0; i < tokens.size(); ++i)
    {
    	argList[i] = new char[tokens[i].length()+1];
    	memcpy ( argList[i], tokens[i].c_str(),  tokens[i].length() );
    	argList[i][tokens[i].length()]='\0';
    }
	for(int j=0;j<tokens.size();j++){
		  LOGI ("%s\n",argList[j]);

	  }
//	clock_t startTimer1, stopTimer1;
//	startTimer1=clock();

	struct timeval start, end;
	gettimeofday(&start, NULL);

	int i=prepareConfig(tokens.size(), argList);
	gettimeofday(&end, NULL);

	LOGI("All code took %f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));

//	stopTimer1 = clock();
//	LOGI("All code took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

	LOGI ("finish2");
    for (int j=0; j<tokens.size(); j++)
    	delete argList[j];



    delete[] argList;


	return 1;
}


int TrainModel::prepareConfig(int parameterNb, char *argList[]) {
	ostringstream s1;
    ConfigTraining config;
    if(parameterNb == 2 && (string(argList[1]) == "--help" || string(argList[1]) == "--?" || string(argList[1]) == "-?" || string(argList[1]) == "-h")) {
        printUsage(argList, config);
    }
    LOGI ("1");
    for(int i = 1; i < parameterNb; i++) {
        vector<string> splitkeyval = split(argList[i], "=");
        if(splitkeyval.size() != 2) {
        	s1.clear();
        	s1.str("");
        	s1 << "Usage: " << argList[0] << " [key]=[value] [[key]=[value]] ..." ;
        	LOGI("%s",s1.str().c_str());
          exit(1);
        } else {
            string key = splitkeyval[0];
            string value = splitkeyval[1];
            if(false) {
            } else if(key == "gpuindex") {
                config.gpuIndex = atoi(value);
            } else if(key == "datadir") {
                config.dataDir = (value);
            } else if(key == "trainfile") {
                config.trainFile = (value);
            } else if(key == "dataset") {
                config.dataset = (value);
                LOGI ("1");
            } else if(key == "validatefile") {
                config.validateFile = (value);
            } else if(key == "numtrain") {
                config.numTrain = atoi(value);
            } else if(key == "numtest") {
                config.numTest = atoi(value);
            } else if(key == "batchsize") {
                config.batchSize = atoi(value);
            } else if(key == "numepochs") {
                config.numEpochs = atoi(value);
            } else if(key == "netdef") {
                config.netDef = (value);
            } else if(key == "loadweights") {
                config.loadWeights = atoi(value);
            } else if(key == "weightsfile") {
                config.weightsFile = (value);
            } else if(key == "writeweightsinterval") {
                config.writeWeightsInterval = atof(value);
            } else if(key == "normalization") {
                config.normalization = (value);
            } else if(key == "normalizationnumstds") {
                config.normalizationNumStds = atof(value);
            } else if(key == "dumptimings") {
                config.dumpTimings = atoi(value);
            } else if(key == "multinet") {
                config.multiNet = atoi(value);
            } else if(key == "loadondemand") {
                config.loadOnDemand = atoi(value);
            } else if(key == "filereadbatches") {
                config.fileReadBatches = atoi(value);
            } else if(key == "normalizationexamples") {
                config.normalizationExamples = atoi(value);
            } else if(key == "weightsinitializer") {
                config.weightsInitializer = (value);
            } else if(key == "initialweights") {
                config.initialWeights = atof(value);
            } else if(key == "trainer") {
                config.trainer = (value);
            } else if(key == "learningrate") {
                config.learningRate = atof(value);
            } else if(key == "rho") {
                config.rho = atof(value);
            } else if(key == "momentum") {
                config.momentum = atof(value);
            } else if(key == "weightdecay") {
                config.weightDecay = atof(value);
            } else if(key == "anneal") {
                config.anneal = atof(value);
            } else {
            	s1.clear();
            	s1.str("");
            	s1 << "Error: key '" << key << "' not recognised";
            	LOGI("%s",s1.str().c_str());
                //printUsage(argList, config);
                return -1;
            }
        }
    }

    //when using sdcard this method make it crash string dataset = toLower(config.dataset);

    LOGI ("dataset %s",config.dataset.c_str());
    if(config.dataset != "") {
        if(config.dataset == "mnist") {
        	LOGI ("mnist");
            config.dataDir = "/data/data/com.sony.openclexample1/app_execdir";
            config.trainFile = "train-images-idx3-ubyte";
            config.validateFile = "t10k-images-idx3-ubyte";
            config.loadOnDemand = 0;
        } else if(config.dataset == "norb") {
            config.dataDir = "../data/norb";
            config.trainFile = "training-shuffled-dat.mat";
            config.validateFile = "testing-sampled-dat.mat";
        } else if(config.dataset == "cifar10") {
            config.dataDir = "../data/cifar10";
            config.trainFile = "train-dat.mat";
            config.validateFile = "test-dat.mat";
        } else if(config.dataset == "kgsgo") {
            config.dataDir = "../data/kgsgo";
            config.trainFile = "kgsgo-train10k-v2.dat";
            config.validateFile = "kgsgo-test-v2.dat";
            config.loadOnDemand = 1;
        } else if(config.dataset == "kgsgoall") {
            config.dataDir = "../data/kgsgo";
            config.trainFile = "kgsgo-trainall-v2.dat";
            config.validateFile = "kgsgo-test-v2.dat";
            config.loadOnDemand = 1;
        } else {
            cout << "dataset " << config.dataset << " not known.  please choose from: mnist, norb, cifar10, kgsgo" << endl;
            return -1;
        }
        s1.clear();
        s1.str("");
        s1 << "Using dataset " << config.dataset << ":\n";
        s1 << "   datadir: " << config.dataDir << ":\n";
        s1 << "   trainfile: " << config.trainFile << ":\n";
        s1 << "   validatefile: " << config.validateFile << ":\n";
        LOGI("%s",s1.str().c_str());
    }
    try {
    	LOGI ("3");
        go(config);
        LOGI ("finish");

    } catch(runtime_error e) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
    LOGI ("finish1");
    return 1;
}

int TrainModel::prepareFiles(string pathOriginalFile,int trainingExample, int inputChannel,string pathNewFileData,string pathNewFileLabel) {
	LOGI("###############");
	GenericLoaderv2 *trainLoader=new GenericLoaderv2(pathOriginalFile);
    int Ntrain = trainingExample;
	int numPlanes = inputChannel;
	int imageSize = trainLoader->getImageSize();

	LOGI("Ntrain %d numPlanes %d imageSize %d",Ntrain,numPlanes,imageSize);
	float * trainData = new float[ (long)Ntrain * numPlanes * imageSize * imageSize ];
	int * trainLabels = new int[Ntrain];

	trainLoader->load(trainData, trainLabels, 0, Ntrain);
	LOGI("11111111111111111111111111");

	FILE *file0 = fopen(pathNewFileData.c_str(), "wb");
	 //float buffer0[2000*3*32*32];
//	 for (int i=0; i<2000*32*32; ++i)
//	    buffer0[i]=(float)trainData[i];
	 fwrite (trainData/*buffer0*/ , sizeof(float), (long)Ntrain * numPlanes * imageSize * imageSize/*sizeof(buffer0)*/, file0);
	 fclose (file0);
	 //LOGI("%zu vs %ld", sizeof(buffer0) ,(long)Ntrain * numPlanes * imageSize * imageSize);

//	 file0 = fopen(pathNewFileData.c_str(), "r");
//	 FILE *file3 = fopen("/data/data/com.sony.openclexample1/preloadingData/testTrainData.raw", "r");
//	 compare_two_binary_files(file0, file3);
//	 fclose (file3);

	//fclose (file0);

	FILE *file2 = fopen(pathNewFileLabel.c_str(), "wb");
	    	//for (int i=0; i<10; ++i)

//	    	 int buffer[2000];
//	    	 for (int i=0; i<2000; ++i)
//	    		 buffer[i]=(int)trainLabels[i];
	    	  //float buffer[] = { 1.0f , 1.0f , 2.0f, 1.0f , 1.0f , 9.0f};
	    	  //file2 = fopen ("myfile.bin", "wb");
	    	  fwrite (trainLabels/*buffer */, sizeof(int), (long)Ntrain/*sizeof(buffer)*/, file2);


	    	  fclose (file2);

//	for (int i=0; i<2048*28*28; ++i){
//		if (trainData[i]!=0.0f)
//			LOGI("%f",trainData[i]);
//	}


    delete [] trainData;
    delete [] trainLabels;
/*
//    ////////////////////////////
//    LOGI("###############");
//    float const* dataTest =0;
//    float const* dataTest2 =0;
//    boost::iostreams::mapped_file file5;
//    boost::iostreams::mapped_file file4;
//    int numberOfElements = 2048*28*28;
//	int numberOfBytes = numberOfElements*sizeof(float);
//	file5.open("/data/data/com.sony.openclexample1/memMapFileData.raw",boost::iostreams::mapped_file::mapmode::readonly, numberOfBytes);
//	file4.open("/data/data/com.sony.openclexample1/preloadingData/testTrainData.raw",boost::iostreams::mapped_file::mapmode::readonly, numberOfBytes);
//	if(file5.is_open()) {
//		dataTest = (const float *)file5.const_data();
//	} else {
//				LOGI("could not map the file filename.raw");
//			}
//	if(file4.is_open()) {
//			dataTest2 = (const float *)file4.const_data();
//		} else {
//					LOGI("could not map the file filename.raw");
//				}
//
//	for(int i=0;i<2048*28*28;i++){
//		if (dataTest[i]!=dataTest2[i]){
//			LOGI("diff %f %f",dataTest[i],dataTest2[i]);
//		}
//	}
//	file4.close();
//	file5.close();
//
//	LOGI("###############");
*/
}

void  TrainModel::compare_two_binary_files(FILE *fp1, FILE *fp2)
{
    char ch1, ch2;
    int flag = 0;

    while (((ch1 = fgetc(fp1)) != EOF) &&((ch2 = fgetc(fp2)) != EOF))
    {
        /*
          * character by character comparision
          * if equal then continue by comparing till the end of files
          */
        if (ch1 == ch2)
        {
            flag = 1;
            continue;
        }
        /*
          * If not equal then returns the byte position
          */
        else
        {
            fseek(fp1, -1, SEEK_CUR);
            flag = 0;
            break;
        }
    }

    if (flag == 0)
    {
        LOGI("Two files are not equal :  byte poistion at which two files differ is %d\n", ftell(fp1)+1);
    }
    else
    {
        LOGI("Two files are Equal\n ");
    }
}

