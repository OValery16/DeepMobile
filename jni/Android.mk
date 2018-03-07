LOCAL_PATH		:= $(call my-dir)
LOCAL_PATH_EXT	:= $(call my-dir)/../extra_libs/
MY_PATH := $(LOCAL_PATH)
include $(call all-subdir-makefiles)

include $(CLEAR_VARS)

LOCAL_PATH := $(MY_PATH)

include $(CLEAR_VARS)

LOCAL_ARM_MODE  := arm

LOCAL_MODULE    := deepMobile

LOCAL_CFLAGS 	+= -DANDROID_CL
LOCAL_CFLAGS    += -O3 -ffast-math

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../include 

LOCAL_EXPORT_LDLIBS := -latomic
LOCAL_SRC_FILES :=refNR.cpp trainEngine/train.cpp predictEngine/predict.cpp 
LOCAL_SRC_FILES += $(notdir $(LOCAL_PATH)/DeepCL)/src/netdef/NetdefToNet.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/GenericLoaderv2.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/Kgsv2Loader.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/NorbLoader.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/MnistLoader.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/ManifestLoaderv1.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/Loader.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/GenericLoaderv1Wrapper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loaders/GenericLoader.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingBackwardCpu.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingForwardCpu.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingBackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingBackward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingForwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/pooling/PoolingForward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/util/JpegHelper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/util/FileHelper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/util/stringhelper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/util/RandomSingleton.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/net/Trainable.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/net/MultiNet.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/net/NeuralNetMould.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/net/NeuralNet.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/patches/Translator.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/patches/RandomPatches.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/patches/RandomTranslationsMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/patches/PatchExtractor.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/patches/RandomPatchesMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/patches/RandomTranslations.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/ConvolutionalLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/DeepCL.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/clmath/GpuOp.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/clmath/GpuAdd.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/clmath/MultiplyBuffer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/clmath/CLMathWrapper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/clmath/MultiplyInPlace.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/clmath/CopyBuffer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutForward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutBackwardCpu.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutBackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutForwardCpu.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutBackward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutForwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/dropout/DropoutLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/NetAction.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/NetAction2.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/NetLearnerOnDemandv2.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/OnDemandBatcher.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/BatchData.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/EpochMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/Batcher.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/OnDemandBatcherv2.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/BatchLearnerOnDemand.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/Batcher2.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/BatchProcess.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/NetLearner.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/batch/NetLearnerOnDemand.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/input/InputLayerMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/input/InputLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/normalize/NormalizationLayerMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/normalize/NormalizationLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/normalize/clNormalization.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loss/LossLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loss/CrossEntropyLoss.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loss/SoftMaxLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loss/SoftMaxLayerPredict.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loss/SquareLossLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/loss/IAcceptsLabels.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/layer/Layer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/layer/LayerMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/SGDMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/TrainingContext.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/TrainerState.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/AdadeltaStateMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/AdagradState.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/AdadeltaState.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/RmspropState.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/AdagradStateMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/Adagrad.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/Trainer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/NesterovStateMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/Nesterov.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/Adadelta.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/RmspropStateMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/Annealer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/Rmsprop.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/SGDStateMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/TrainerStateMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/NesterovState.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/SGD.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/SGDState.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/trainers/TrainerMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/weights/WeightsPersister.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/weights/WeightsInitializer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/weights/OriginalInitializer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/weights/UniformInitializer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/CppRuntimeBoundary.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/BackpropWeightsNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/ConvolutionalLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/ReduceSegments.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/BackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/ConvolutionalMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/LayerDimensions.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/conv/Forward1.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/fc/FullyConnectedMaker.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/fc/FullyConnectedLayer.cpp
LOCAL_SRC_FILES +=$(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/luac.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ltablib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lbaselib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/loslib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/loadlib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lstrlib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lapi.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lmathlib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ldump.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lzio.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ldblib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lgc.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lmem.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/liolib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lopcodes.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ltm.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lparser.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ltable.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/linit.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lauxlib.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lua.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ldebug.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lundump.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/llex.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/print.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lvm.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lstate.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/ldo.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lfunc.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lstring.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lcode.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/thirdparty/lua-5.1.5/src/lobject.c $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/util/StatefulTimer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/util/easycl_stringhelper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/CLKernel.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/deviceinfo_helper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/DevicesInfo.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/gpuinfo.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/DeviceInfo.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/EasyCL.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/templates/LuaTemplater.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/templates/TemplatedKernel.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/platforminfo_helper.cpp $(notdir $(LOCAL_PATH)/DeepCL)/EasyCL/CLWrapper.cpp
LOCAL_SRC_FILES += $(notdir $(LOCAL_PATH)/)kernelManager/ConfigManager.cpp 

#$(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationForward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationBackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationFunction.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationForwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationBackward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationBackwardCpu.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationForwardCpu.cpp
#$(notdir $(LOCAL_PATH)/DeepCL)/src/forcebackprop/ForceBackpropLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/forcebackprop/ForceBackpropLayerMaker.cpp  
LOCAL_LDLIBS 	:= -llog -ljnigraphics 
LOCAL_SHARED_LIBRARIES := libjpegturboSIMD 

LOCAL_CFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp

LOCAL_LDLIBS := -ljnigraphics -llog -landroid
LOCAL_C_INCLUDES += $(LOCAL_PATH) \
                    $(LOCAL_PATH)/libjpegturbo \
                    $(LOCAL_PATH)/libjpegturbo/android                  
LOCAL_LDLIBS 	+= $(LOCAL_PATH_EXT)libOpenCL.so  

LOCAL_STATIC_LIBRARIES :=boost_iostreams_static
# LOCAL_SHARED_LIBRARIES := libDeepCL libEasyCL

LOCAL_LDLIBS 	:= -llog -ljnigraphics 
LDFLAGS += -pthread
LOCAL_LDLIBS 	+= $(LOCAL_PATH_EXT)libOpenCL.so  

include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_MODE  := arm

LOCAL_MODULE    := openclexample1

LOCAL_CFLAGS 	+= -DANDROID_CL
LOCAL_CFLAGS    += -O3 -ffast-math
LOCAL_LDLIBS := -ljnigraphics -llog -landroid
LOCAL_CFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp
LOCAL_SRC_FILES :=sonyOpenCLexample1.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../include 
LOCAL_STATIC_LIBRARIES :=deepMobile
LOCAL_LDLIBS 	+= $(LOCAL_PATH_EXT)libOpenCL.so  

include $(BUILD_SHARED_LIBRARY)

$(call import-module,boost/1.57.0)
