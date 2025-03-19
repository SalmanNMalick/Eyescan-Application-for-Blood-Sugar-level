package com.eyescan.facedetectframeprocessor;

import androidx.camera.core.ImageProxy;

import com.eyescan.MainApplication;
import com.facebook.react.bridge.ReadableNativeArray;
import com.facebook.react.bridge.WritableNativeArray;
import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.util.Log;

//import androidx.camera.view.PreviewView;

import java.lang.Double;
import java.util.ArrayList;
import java.lang.Math;
public class FaceDetectFrameProcessorPlugin extends FrameProcessorPlugin {

  int rotation = 0;
  boolean backCam = true;
  ImageProcess imageProcess = new ImageProcess();
  private InstDetect yolov5TFLiteDetector;
  private String modelName; 

  public FaceDetectFrameProcessorPlugin() {
    super("faceDetect");
    Log.i("ReactNative", " _____________________________hi");
    yolov5TFLiteDetector = new InstDetect(MainApplication.getAppContext());
    yolov5TFLiteDetector.initialize();
  }

  @Override
  public Object callback(ImageProxy image, Object[] params) {
    int previewHeight = 1280;
    int previewWidth = 720;
    long start = System.currentTimeMillis();

    // code goes here
    WritableNativeArray array = new WritableNativeArray();
    //Log.i("ReactNative", " width = " + image.getWidth() + ", height = " + image.getHeight());

    for(Object param : params){
      String classStr = param.getClass().toString();
      int i =0;
      ArrayList<Object> paramList = ((ReadableNativeArray)param).toArrayList();
      for(Object ps : paramList){
        Log.i("image", "paramList class to string: " + ps.getClass().toString());
        if(ps.getClass().toString().equals("class java.lang.Double")){
//          Log.i("image", "ps to string: " + ps.toString());
          if(i == 0)
            rotation = (int) Math.round((Double)ps);
          else if (i == 1)
            previewWidth = (int) Math.round((Double)ps);
          else
            previewHeight= (int) Math.round((Double)ps);
        }
        if(ps.getClass().toString().equals("class java.lang.Boolean")){
          backCam = (boolean)ps;
        }
        if(ps.getClass().toString().equals("class java.lang.String")){
          modelName = (String)ps;
        }
        i++;
      }
    }

    Log.i("image",""+previewWidth+'/'+previewHeight);
    Log.i("image","~~~~~~~~~~~~~~~~~~~~~~backCam: " + backCam);

    byte[][] yuvBytes = new byte[3][];
    ImageProxy.PlaneProxy[] planes = image.getPlanes();
    int imageHeight = image.getHeight();
    int imagewWidth = image.getWidth();

    imageProcess.fillBytes(planes, yuvBytes);
    int yRowStride = planes[0].getRowStride();
    final int uvRowStride = planes[1].getRowStride();
    final int uvPixelStride = planes[1].getPixelStride();

    int[] rgbBytes = new int[imageHeight * imagewWidth];
    if(backCam)
      imageProcess.YUV420ToARGB8888( yuvBytes[0], yuvBytes[1], yuvBytes[2], imagewWidth, imageHeight, yRowStride, uvRowStride, uvPixelStride, rgbBytes);
    else
      imageProcess.YUV420ToARGB8888Front( yuvBytes[0], yuvBytes[1], yuvBytes[2], imagewWidth, imageHeight, yRowStride, uvRowStride, uvPixelStride, rgbBytes);

    // 原图bitmap
    Bitmap imageBitmap = Bitmap.createBitmap(imagewWidth, imageHeight, Bitmap.Config.ARGB_8888);
    imageBitmap.setPixels(rgbBytes, 0, imagewWidth, 0, 0, imagewWidth, imageHeight);

    // 图片适应屏幕fill_start格式的bitmap
    double scale = Math.max(
            previewHeight / (double) (rotation % 180 == 0 ? imagewWidth : imageHeight),
            previewWidth / (double) (rotation % 180 == 0 ? imageHeight : imagewWidth)
    );
    Matrix fullScreenTransform = imageProcess.getTransformationMatrix(
            imagewWidth, imageHeight,
            (int) (scale * imageHeight), (int) (scale * imagewWidth),
            rotation % 180 == 0 ? 90 : 0, false
    );

    // 适应preview的全尺寸bitmap
    Bitmap fullImageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imagewWidth, imageHeight, fullScreenTransform, false);
    // 裁剪出跟preview在屏幕上一样大小的bitmap
    int cropWidth = previewWidth + 10;
    int cropHeight = previewHeight - 35;
    int startX = 100;
    int startY = 35;

    if (startX + cropWidth > fullImageBitmap.getWidth() || startY + cropHeight > fullImageBitmap.getHeight()) {
        // This would be an out of bounds crop, so adjust as necessary
        cropWidth = fullImageBitmap.getWidth() - startX;
        cropHeight = fullImageBitmap.getHeight() - startY;
    }

    Bitmap cropImageBitmap = Bitmap.createBitmap(
            fullImageBitmap, startX, startY,
            cropWidth, cropHeight
    );


    // 模型输入的bitmap
    Matrix previewToModelTransform =
            imageProcess.getTransformationMatrix(
                    cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
                    yolov5TFLiteDetector.getInputSize().getWidth(),
                    yolov5TFLiteDetector.getInputSize().getHeight(),
                    0, false);
    Bitmap modelInputBitmap = Bitmap.createBitmap(cropImageBitmap, 0, 0,
            cropImageBitmap.getWidth(), cropImageBitmap.getHeight(),
            previewToModelTransform, false);

    Matrix modelToPreviewTransform = new Matrix();
    previewToModelTransform.invert(modelToPreviewTransform);

    ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(modelInputBitmap, modelName);
    long end = System.currentTimeMillis();
    long costTime = (end - start);
    for (Recognition res : recognitions) {
      RectF location = res.getLocation();
      String label = res.getLabelName();
      float confidence = res.getConfidence();
      modelToPreviewTransform.mapRect(location);
      array.pushString(label);
      array.pushDouble((double)confidence);
      array.pushDouble((double)costTime);
      array.pushDouble((double)location.left);
      array.pushDouble((double)location.top);
      array.pushDouble((double)location.right);
      array.pushDouble((double)location.bottom);
      Log.e("ReactNative", "~*~*~* In Loop: " + String.valueOf(array.size()));
    }


//    Log.e("ReactNative", "~*~*~* Num Detections (arraysize): " + String.valueOf(array.size()) + ", Cost Time: " + String.valueOf(costTime));

    return array;
  }
}