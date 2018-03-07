#include "DeepCL.h"
#include "../EasyCL/DevicesInfo.h"

#undef STATIC
#define STATIC
#define PUBLIC

//#define DEEPCL_VERBOSE 1
//int DEEPCL_VERBOSE = 1;

using namespace easycl;

//DeepCL::DeepCL() :
//    EasyCL() {
//}
//DeepCL::DeepCL(int gpu) :
//    EasyCL(gpu) {
//}
PUBLIC DeepCL::DeepCL(cl_platform_id platformId, cl_device_id deviceId) :
    EasyCL(platformId, deviceId) {
}
PUBLIC DeepCL::~DeepCL() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: ~DeepCL");
#endif


}
PUBLIC void DeepCL::deleteMe() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: deleteMe");
#endif


    delete this;
}
PUBLIC STATIC DeepCL *DeepCL::createForFirstGpu() {

#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: createForFirstGpu");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedGpu(0, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForFirstGpuOtherwiseCpu() {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: createForFirstGpuOtherwiseCpu");
#endif


    if(DevicesInfo::getNumGpus() >= 1) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: getNumGpus() >= 1) {");
#endif


        return createForFirstGpu();
    } else {
        return createForIndexedDevice(0);
    }
}
PUBLIC STATIC DeepCL *DeepCL::createForIndexedDevice(int device) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: createForIndexedDevice");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedDevice(device, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForIndexedGpu(int gpu) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: createForIndexedGpu");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedGpu(gpu, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForPlatformDeviceIndexes(int platformIndex, int deviceIndex) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: createForPlatformDeviceIndexes");
#endif


    cl_platform_id platformId;
    cl_device_id deviceId;
    DevicesInfo::getIdForIndexedPlatformDevice(platformIndex, deviceIndex, CL_DEVICE_TYPE_ALL, &platformId, &deviceId);
    return new DeepCL(platformId, deviceId);
}
PUBLIC STATIC DeepCL *DeepCL::createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/DeepCL.cpp: createForPlatformDeviceIds");
#endif


    return new DeepCL(platformId, deviceId);
}

