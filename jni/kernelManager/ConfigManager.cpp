
#include "ConfigManager.h"
#include "../DeepCL/src/util/stringhelper.h"


using namespace std;


ConfigManager::ConfigManager(){


	string filepath="";
	string line;
	ifstream myfileI (kernellList);
	if (myfileI.is_open())
	{
	   while ( getline (myfileI,line) )
	   {
		   vector<string> splitData=split(line, ",");
		   listOfCompiledKernel.insert ({splitData[0],splitData[1]});
        }
    myfileI.close();
    }



}


ConfigManager::ConfigManager(std::string fileDirectory,string binaryFilesRepo){


	string filepath="";
	string line;
	ifstream myfileI (fileDirectory);
	if (myfileI.is_open())
	{
	   while ( getline (myfileI,line) )
	   {
		   vector<string> splitData=split(line, ",");
		   listOfCompiledKernel.insert ({splitData[0],splitData[1]});
        }
    myfileI.close();
    }
	kernellList=fileDirectory;
	binaryRepo=binaryFilesRepo;


}



bool ConfigManager::alreadyCompiledKernel(string kernelname, string option,string operation,string &filepath){

	if (listOfCompiledKernel.empty()!=0){
		LOGI("empty");
		filepath=binaryRepo+operation+"_"+kernelname+".bin";
		return false;
	}
	string key=operation+" "+kernelname+" "+option;
	std::unordered_map<string,string>::const_iterator got = listOfCompiledKernel.find (key);

	if ( got == listOfCompiledKernel.end() ){
//		LOGI("false");
//	    LOGI("not found in the list");
//	    LOGI("find the name of the binary file such as name_number.bin");
	    int i=0;
	    for (auto& x: listOfCompiledKernel) {
	    	if (x.first.find(kernelname)!=std::string::npos){
        		i++;
        	}
	    	string kernelname2=operation+"_"+kernelname+"_"+std::to_string(i);
	    	filepath=binaryRepo+kernelname2+".bin";
	        //std::cout << x.first << ": " << x.second << std::endl;
	      }
	    return false;
	}else{

		  filepath=got->second.c_str();
		  return true;
	  }

	return true;
}

void ConfigManager::addNewCompiledKernel(string kernelname, string options,string operation,string &filepath){

	string key=operation+" "+kernelname+" "+options;
	listOfCompiledKernel.insert ({key,filepath});

	ofstream myfileO;
	myfileO.open (kernellList,std::ofstream::out | std::ofstream::app);
	myfileO <<operation<<" "<<kernelname<<" " <<options <<","<<filepath<<"\n";
	myfileO.close();
}



