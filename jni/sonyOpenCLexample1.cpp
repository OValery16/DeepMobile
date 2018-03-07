//
//  sonyOpenCLexample1.cpp
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

#include <stdio.h>
#include "sonyOpenCLexample1.h"

#include <android/bitmap.h>
#include <jni.h>


#include "trainEngine/train.h"
#include "predictEngine/predict.h"



#include "kernelManager/kernelManager.h"

#include <sys/stat.h>
#include <dirent.h>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

//#include "train.h"
#define CLMATH_VERBOSE 1
#define DEEPCL_VERBOSE 1


char* ReadFile(char *filename)
{
   char *buffer = NULL;
   int string_size, read_size;
   FILE *handler = fopen(filename, "r");
   LOGI( "fopen");

   if (handler)
   {
	   LOGI( "handler");
       // Seek the last byte of the file
       fseek(handler, 0, SEEK_END);
       LOGI( "Seek");
       // Offset from the first to the last byte, or in other words, filesize
       string_size = ftell(handler);
       // go back to the start of the file
       rewind(handler);

       // Allocate a string that can hold it all
       buffer = (char*) malloc(sizeof(char) * (string_size + 1) );

       // Read it all in one operation
       read_size = fread(buffer, sizeof(char), string_size, handler);

       // fread doesn't set it so put a \0 in the last position
       // and buffer is now officially a string
       buffer[string_size] = '\0';

       if (string_size != read_size)
       {
           // Something went wrong, throw away the memory and set
           // the buffer to NULL
           free(buffer);
           buffer = NULL;
       }

       // Always remember to close the file.
       fclose(handler);
    }

    return buffer;
}

int
mkpath(std::string s,mode_t mode)
{
    size_t pre=0,pos;
    std::string dir;
    int mdret;

    if(s[s.size()-1]!='/'){
        // force trailing / so we can handle everything in loop
        s+='/';
    }

    while((pos=s.find_first_of('/',pre))!=std::string::npos){
        dir=s.substr(0,pos++);
        pre=pos;
        if(dir.size()==0) continue; // if leading / first time is 0 length
        if((mdret=mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}

int64_t getTimeNsec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000000000LL + now.tv_nsec;
}
static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_runOpenCL(JNIEnv* env, jclass clazz, jobject bitmapIn, jobject bitmapOut, jintArray info)
{


	 struct timespec start1;
		struct timespec end1;
		double elapsedSeconds1;
		clock_gettime(CLOCK_MONOTONIC, &start1);


		struct timeval start, end;
		/*get the start time*/
		gettimeofday(&start, NULL);
////	int mkdirretval;
////	    mkdirretval=mkpath("/data/data/com.sony.openclexample1/app_execdir",0755);
////	    if (-1 == mkdirretval)
////	    {
////	        LOGI("Error creating directory!n");
////	        exit(1);
////	    }
//
////	 DIR *theFolder = opendir("/data/data/com.sony.openclexample1/app_execdir/binariesKernel/");
////	    struct dirent *next_file;
////	    char filepath[256];
////
////	    while ( (next_file = readdir(theFolder)) != NULL )
////	    {
////	        // build the path for each file in the folder
////	        sprintf(filepath, "%s/%s", "/data/data/com.sony.openclexample1/app_execdir/binariesKernel/", next_file->d_name);
////	        remove(filepath);
////	    }
//
//	string dirtemp="/data/data/com.sony.openclexample1/app_execdir/configToPrecompile.txt";
//	char * st=ReadFile((char*)dirtemp.c_str());
////
////	dirtemp="/data/data/com.sony.openclexample1/app_execdir/olivierdata/test4/manifest4bis.txt"
//	LOGI("%s",st);
//
//	DIR *dir;
//	struct dirent *ent;
//	if ((dir = opendir ("/data/data/com.sony.openclexample1/app_execdir/test4/")) != NULL) {
//	  /* print all the files and directories within directory */
//	  while ((ent = readdir (dir)) != NULL) {
////		  if ((ent->d_name!="..")&&(ent->d_name!="."))
////		  	  remove( "myfile.txt" );
//	  LOGI ("%s\n", ent->d_name);
//	  }
//	  closedir (dir);
//	} else {
//	  /* could not open directory */
//	  perror ("");
//	  return EXIT_FAILURE;
//	}
//if (0){
	//if (1){

	//t->trainCmd("./train numepochs=5 netdef=8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n learningrate=0.002 dataset=mnist");///data/data/com.sony.openclexample1/app_execdir
	//t->trainCmd("./train numtest=-1 numtrain=10000 datadir=/sdcard1/olivierdata");
//8c5z-relu-mp2-16c5z-relu-mp3-

	//1s8c5z-relu-mp2-1s16c5z-relu-mp3-
//-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n

//7janv t->trainCmd("./train datadir=/data/data/com.sony.openclexample1/app_execdir/test4/ netdef=1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n numepochs=3 batchsize=128 trainfile=manifest.txt validatefile=manifest.txt numtrain=2048  numtest=2048");
//1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n
		LOGI("###############");
	TrainModel* t= new TrainModel();
	//t->trainCmd("./train datadir=/data/data/com.sony.openclexample1/app_execdir/ netdef=1s8c5z-mp2-1s16c5z-mp3-152n-tanh-10n numepochs=3 batchsize=128 trainfile=train-images-idx3-ubyte validatefile=t10k-images-idx3-ubyte numtrain=2048  numtest=2048");
t->trainCmd("./train datadir=/data/data/com.sony.openclexample1/copyZ5/app_execdir/ netdef=1s8c5z-mp2-1s16c5z-mp3-152n-tanh-10n numepochs=3 batchsize=32 trainfile=train-images-idx3-ubyte validatefile=t10k-images-idx3-ubyte numtrain=2048  numtest=2048");


	//t->trainCmd("./train datadir=/data/data/com.sony.openclexample1/app_execdir/ netdef=1s32c5z-relu-mp3-1s32c5z-relu-mp3-1s64c5z-relu-mp3-64n-relu-10n numepochs=1 batchsize=50 trainfile=train-images-idx3-ubyte validatefile=t10k-images-idx3-ubyte numtrain=2000  numtest=2000");


	//t->prepareFiles("/sdcard1/deepLearning/cifar10train//manifestZ5.txt",2000, 3,"/data/data/com.sony.openclexample1/memCIFAR10MapFileData.raw","/data/data/com.sony.openclexample1/memCIFAR10MapFileLabel.raw");
	//t->prepareFiles("/data/data/com.sony.openclexample1/app_execdir//train-images-idx3-ubyte",60000, 1,"/data/data/com.sony.openclexample1/memMapFileData60000.raw","/data/data/com.sony.openclexample1/memMapFileLabel60000.raw");

	delete t;

	clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds(&end1) - TimeSpecToSeconds(&start1);
	LOGI("3)time %f\n\n",elapsedSeconds1);
	/*get the end time*/
	gettimeofday(&end, NULL);
	/*Print the amount of time taken to execute*/
	LOGI("%f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));


        return 0;
}



extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_runNativeC(JNIEnv* env, jclass clazz, jobject bitmapIn, jobject bitmapOut, jintArray info)
{


	PredictionModel* p=new PredictionModel();
	p->predictCmd("./predict batchsize=6 inputfile=/sdcard1/olivierdata/test4/manifest3.txt outputfile=/data/data/com.sony.openclexample1/app_execdir/pred2.txt");
	delete p;


	return 0;
}

extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_runPrecompile(JNIEnv* env, jclass clazz, jobject bitmapIn, jobject bitmapOut, jintArray info)
{

//	string path ="/data/data/com.sony.openclexample1/app_execdir/binariesKernel/list.txt";
//	string binaryFileRep="/data/data/com.sony.openclexample1/app_execdir/binariesKernel/";
//	string configPath ="/data/data/com.sony.openclexample1/app_execdir/co"
//			"nfigToPrecompile.txt";
//	KernelManager *km =new KernelManager(path,binaryFileRep);
//	km->CompileKernels(configPath);
//	delete km;

	return 0;
}

