//
//  OpenCLActivity.java
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

package com.sony.openclexample1;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DateFormat;
import java.util.Date;

//import android.content.pm.PackageManager;
//import android.support.v4.app.ActivityCompat;
//import android.support.v4.content.ContextCompat;
//import android.support.v7.app.AppCompatActivity;

import android.app.Activity;
import android.content.res.AssetManager;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

public class OpenCLActivity extends Activity {
	protected static final String TAG = "OpenCLActivity";
	
	private static boolean copyAssetFolder(AssetManager assetManager,
            String fromAssetPath, String toPath) {
        try {
            String[] files = assetManager.list(fromAssetPath);
            new File(toPath).mkdirs();
            boolean res = true;
            for (String file : files)
                if (file.contains("."))
                    res &= copyAsset(assetManager, 
                            fromAssetPath + "/" + file,
                            toPath + "/" + file);
                else 
                    res &= copyAssetFolder(assetManager, 
                            fromAssetPath + "/" + file,
                            toPath + "/" + file);
            return res;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean copyAsset(AssetManager assetManager,
            String fromAssetPath, String toPath) {
        InputStream in = null;
        OutputStream out = null;
        try {
          in = assetManager.open(fromAssetPath);
          new File(toPath).createNewFile();
          out = new FileOutputStream(toPath);
          copyFile(in, out);
          in.close();
          in = null;
          out.flush();
          out.close();
          out = null;
          return true;
        } catch(Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
          out.write(buffer, 0, read);
        }
    }
	
	private void copyFile(final String f) {
		InputStream in;
		try {
			in = getAssets().open(f);
			final File of = new File(getDir("execdir",MODE_PRIVATE), f);
			
			final OutputStream out = new FileOutputStream(of);

			final byte b[] = new byte[65535];
			int sz = 0;
			while ((sz = in.read(b)) > 0) {
				out.write(b, 0, sz);
			}
			in.close();
			out.close();
		} catch (IOException e) {       
			e.printStackTrace();
		}
	}
	
	static boolean sfoundLibrary = true;  
	
	static {
	  try {
		  //System.loadLibrary("clew"); 
		  //System.loadLibrary("EasyCL"); 
		  System.loadLibrary("openclexample1");  
	  }
	  catch (UnsatisfiedLinkError e) {
	      sfoundLibrary = false;
	  }
	}
	
	public static native int runOpenCL(Bitmap bmpIn, Bitmap bmpOut, int info[]);
	public static native int runNativeC(Bitmap bmpIn, Bitmap bmpOut, int info[]);
	public static native int runPrecompile(Bitmap bmpIn, Bitmap bmpOut, int info[]);
	final int info[] = new int[3]; // Width, Height, Execution time (ms)

    LinearLayout layout;
    Bitmap bmpOrig, bmpOpenCL, bmpNativeC;
    ImageView imageView;
    TextView textView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        
        
//        String[] perms = {"android.permission.STORAGE"};
//
//        int permsRequestCode = 200; 
//
//        requestPermissions(perms, permsRequestCode);
//        
        
        
        imageView = (ImageView) findViewById(R.id.imageHere);
        textView = (TextView) findViewById(R.id.resultText);
            
//        copyFile("bilateralKernel.cl"); //copy cl kernel file from assets to /data/data/...assets
//        copyFile("similarityMatrix.cl");
//        copyFile("ImageSobelFilter.cl");
//        copyFile("mnistTest2048");
//        copyFile("train-images-idx3-ubyte");
//        copyFile("train-labels-idx1-ubyte");
        
        /////////////////////////////////////////
        
        
        copyFile("t10k-images-idx3-ubyte");
        copyFile("t10k-labels-idx1-ubyte");
        copyFile("configToPrecompile.txt");
        copyAssetFolder(getAssets(), "preloading", 
                "/data/data/com.sony.openclexample1/preloadingData");
        //deleteFiles("/data/data/com.sony.openclexample1/preloadingData/binariesKernel");
        deleteFiles("/data/data/com.sony.openclexample1/preloadingData/kernelcode.txt");
        
        copyAssetFolder(getAssets(), "preloading/binariesKernel", 
                "/data/data/com.sony.openclexample1/preloadingData/binariesKernel");
        
        ///////////////////////////
        
        bmpOrig = BitmapFactory.decodeResource(this.getResources(), R.drawable.brusigablommor);
        info[0] = bmpOrig.getWidth();
        info[1] = bmpOrig.getHeight();
        
        bmpOpenCL = Bitmap.createBitmap(info[0], info[1], Config.ARGB_8888);
        bmpNativeC = Bitmap.createBitmap(info[0], info[1], Config.ARGB_8888);
        textView.setText("Original");
        imageView.setImageBitmap(bmpOrig);
        runPrecompile(bmpOrig, bmpOpenCL, info);
    }
        
    public void showOriginalImage(View v) {
    	textView.setText("Original");
    	imageView.setImageBitmap(bmpOrig);
    }

    public void showOpenCLImage(View v) {
    	String currentDateTimeString = DateFormat.getDateTimeInstance().format(new Date());
    	runOpenCL(bmpOrig, bmpOpenCL, info);
    	// textView is the TextView view that should display it
    	textView.setText(currentDateTimeString);
    	
//    	long startTime = System.currentTimeMillis();
//    	runOpenCL(bmpOrig, bmpOpenCL, info);
    	
//    	new Thread(new Runnable() {
//    	    public void run() {
//    	    	runOpenCL(bmpOrig, bmpOpenCL, info);
//    	    }
//    	  }).start();

    }

    public void showNativeCImage(View v) {
    	long startTime = System.currentTimeMillis();
    	runNativeC(bmpOrig, bmpNativeC, info);
    	long difference = System.currentTimeMillis() - startTime;
    	textView.setText("Bilateral Filter, OpenCL, Processing time is " + difference/*info[2]*/ + " ms");
    	
    	imageView.setImageBitmap(bmpNativeC);
    }
    
    public static void deleteFiles(String path) {

        File file = new File(path);

        if (file.exists()) {
            String deleteCmd = "rm -r " + path;
            Runtime runtime = Runtime.getRuntime();
            try {
                runtime.exec(deleteCmd);
            } catch (IOException e) { }
        }
    }
    
    
}
