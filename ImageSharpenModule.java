package com.eyescan;
import android.graphics.Matrix;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.tensorflow.lite.support.common.FileUtil;
import java.io.InputStream;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;
import java.io.File;
import java.io.FileOutputStream;
import java.util.stream.IntStream;
import java.util.ArrayList;
import java.util.Collections;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.FileInputStream;
import android.content.res.AssetManager;
import android.content.res.AssetFileDescriptor;
import org.tensorflow.lite.gpu.GpuDelegate;
import java.nio.ByteBuffer;

public class ImageSharpenModule extends ReactContextBaseJavaModule {
    private ReactApplicationContext mContext;
    private static final int SIZE = 256;

    // private Interpreter interpreter;
    public ImageSharpenModule(ReactApplicationContext reactContext) {
        super(reactContext);
        this.mContext = reactContext;
        
    }
    public static Bitmap convertToBitmap(float[][][] image) {
        int height = image.length;
        int width = image[0].length;

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Extracting RGB values from the normalized image array
                int red = (int) (image[y][x][0] * 255.0f);
                int green = (int) (image[y][x][1] * 255.0f);
                int blue = (int) (image[y][x][2] * 255.0f);

                // Creating the pixel color
                int pixelColor = Color.rgb(red, green, blue);

                // Setting the pixel color to the bitmap
                bitmap.setPixel(x, y, pixelColor);
            }
        }

        return bitmap;
    }
    public Bitmap convertFloatArrayToBitmap(float[][][] floatArray) {
        int width = floatArray[0][0].length;
        int height = floatArray[0].length;

        // Create a Bitmap with the same width and height as the float array
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        // Set pixels of the Bitmap based on the values in the float array
        int[] pixels = new int[width * height];
        int pixelIndex = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Convert float values to pixel values (assuming float values are in [0, 1] range)
                int alpha = 255; // Set alpha to 255 (fully opaque)
                int red = (int) (floatArray[0][y][x] * 255); // Assuming floatArray[0] represents red channel
                int green = (int) (floatArray[1][y][x] * 255); // Assuming floatArray[1] represents green channel
                int blue = (int) (floatArray[2][y][x] * 255); // Assuming floatArray[2] represents blue channel
                pixels[pixelIndex++] = (alpha << 24) | (red << 16) | (green << 8) | blue;
            }
        }

        // Set pixels to the Bitmap
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);

        return bitmap;
    }



public Bitmap processImage(String lowImagePath, Interpreter interpreter) {
    Bitmap bitmap = BitmapFactory.decodeFile(lowImagePath);
    int heighttttt=bitmap.getHeight();
    int widthhhh=bitmap.getWidth();
    int change_width = 256;
    int change_height = 256;
    float quantizationFactor = 0.003921568859368563f; // Define quantization factor

    // Resize image
    Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, change_width, change_height, true);
    float[][][] normalizedImage = new float[change_height][change_width][3];
    for (int y = 0; y < change_height; y++) {
        for (int x = 0; x < change_width; x++) {
            int pixel = resizedBitmap.getPixel(x, y);
            int red = (pixel >> 16) & 0xFF;
            int green = (pixel >> 8) & 0xFF;
            int blue = pixel & 0xFF;
            normalizedImage[y][x][0] = ((float) red )* quantizationFactor;
            normalizedImage[y][x][1] = ((float) green )* quantizationFactor;
            normalizedImage[y][x][2] = ((float) blue )* quantizationFactor;
        }
    }
        // Prepare input buffer
        int inputSize = change_width * change_height * 3; // Assuming 3 color 3 (RGB) per pixel
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputSize * Float.BYTES);
        inputBuffer.order(ByteOrder.nativeOrder());

        // Flatten normalized image and copy to input buffer
        for (int y = 0; y < change_height; y++) {
            for (int x = 0; x < change_width; x++) {
                for (int c = 0; c < 3; c++) {
                    inputBuffer.putFloat(normalizedImage[y][x][c]);
                }
            }
        }

        // Reset buffer position to beginning before using it
        inputBuffer.rewind();

        // Prepare output tensor buffer
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(interpreter.getOutputTensor(0).shape(), interpreter.getOutputTensor(0).dataType());

        // Run inference
        interpreter.run(inputBuffer, outputTensorBuffer.getBuffer());

        // Process output tensor
        int SIZE = 256; // Assuming SIZE is 256
        float[][][] predicted = new float[SIZE][SIZE][3];
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                for (int k = 0; k < 3; k++) {
                    predicted[i][j][k] = outputTensorBuffer.getFloatValue(i * SIZE * 3 + j * 3 + k);
                }
            }
        }

        // Convert predicted array to Bitmap
        Bitmap outbitmap = convertToBitmap(predicted);
        Bitmap resizedBbbbitmap = Bitmap.createScaledBitmap(outbitmap, widthhhh, heighttttt, true);

        return resizedBbbbitmap;

    }


    public Bitmap performInference(Bitmap inputImage, Interpreter interpreter) {
        Object[] preprocessedData = preprocessImage(inputImage);
        ByteBuffer inputBuffer = (ByteBuffer) preprocessedData[0];

        // Allocate output buffer with a size that matches the expected output size
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(1280 * 720 * 3 * 4);
        outputBuffer.order(ByteOrder.nativeOrder());

        interpreter.run(inputBuffer, outputBuffer);

        return postprocessOutput(outputBuffer, 1280, 720);
    }


    private Object[] preprocessImage(Bitmap image) {
        Bitmap resizedImage = Bitmap.createScaledBitmap(image, 320, 180, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * resizedImage.getWidth() * resizedImage.getHeight() * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[resizedImage.getWidth() * resizedImage.getHeight()];
        resizedImage.getPixels(intValues, 0, resizedImage.getWidth(), 0, 0, resizedImage.getWidth(), resizedImage.getHeight());

        // Normalize and add pixel values to ByteBuffer
        for (int pixel : intValues) {
            float normalizedR = Color.red(pixel) ;
            float normalizedG = Color.green(pixel) ;
            float normalizedB = Color.blue(pixel) ;

            byteBuffer.putFloat(normalizedR);
            byteBuffer.putFloat(normalizedG);
            byteBuffer.putFloat(normalizedB);
        }
        byteBuffer.rewind(); // Rewind the buffer before returning
        return new Object[]{byteBuffer};
    }

    private Bitmap postprocessOutput(ByteBuffer outputBuffer, int width, int height) {
        Bitmap outputImage = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        int[] intValues = new int[width * height];

        // Rewind the buffer before reading
        outputBuffer.rewind();

        // Read float values from output buffer and convert to RGB
        for (int i = 0; i < width * height; i++) {
            float redFloat = outputBuffer.getFloat();
            float greenFloat = outputBuffer.getFloat();
            float blueFloat = outputBuffer.getFloat();

            // Convert float values back to range [0, 255]
            int red = Math.round(redFloat );
            int green = Math.round(greenFloat );
            int blue = Math.round(blueFloat );

            // Ensure that RGB values are within the valid range [0, 255]
            red = Math.min(255, Math.max(0, red));
            green = Math.min(255, Math.max(0, green));
            blue = Math.min(255, Math.max(0, blue));

            // Set pixel values in the intValues array
            intValues[i] = Color.rgb(red, green, blue);
        }

        // Set pixel values in the output image
        outputImage.setPixels(intValues, 0, width, 0, 0, width, height);

        return outputImage;
    }
    @ReactMethod
    public boolean processAllImages(String folderPath, Callback callback) {
    try {
        if (folderPath.startsWith("file:///")) {
            folderPath = folderPath.substring(7);
        }
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();

        // Create a new folder for copied images
        String allImagesFolderPath = folder.getAbsolutePath() + File.separator + "betterimages";
        File allImagesFolder = new File(allImagesFolderPath);
        allImagesFolder.mkdirs(); // Make sure the directory exists

        // Load the TensorFlow Lite model
        AssetManager assetManager = this.mContext.getAssets();
        AssetFileDescriptor fileDescriptor = assetManager.openFd("esrgan312.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        Interpreter.Options tfliteOptions = new Interpreter.Options();

        Interpreter tfLite = new Interpreter(buffer, tfliteOptions);

        if (listOfFiles != null) {
            for (File file : listOfFiles) {
                if (file.isFile() && isImageFile(file)) {
                    // Process the image
                    // processImage();
                    
                    // Bitmap image = BitmapFactory.decodeFile(file.getAbsolutePath());
                    // Bitmap predictedImage = Bitmap.createScaledBitmap(image, 50, 50, true);
                    Bitmap predictedImage = BitmapFactory.decodeFile(file.getAbsolutePath());
                    int heightt = predictedImage.getHeight();
                    int widtth = predictedImage.getWidth();
                    // Bitmap inputImage = BitmapFactory.decodeFile(imagePath)

                    for (int i = 0; i < 3; i++) {
                    
                    predictedImage = performInference(predictedImage, tfLite);
                    
                    predictedImage =Bitmap.createScaledBitmap(predictedImage, widtth, heightt, true);
        
                    }
                    // predictedImage=sharpenImage(predictedImage);

                    // Bitmap predictedImage = performInference(predictedImage, tfLite);
                    // Bitmap predictedImage = processImage(file.getAbsolutePath(), tfLite);

                    // Generate a unique filename
                    String fileName = file.getName();
                    String[] fileNameParts = fileName.split("\\.");
                    String baseName = fileNameParts[0];
                    String extension = fileNameParts[1];
                    String uniqueFileName = baseName + "_generated_Image16." + extension;

                    File newFile = new File(allImagesFolderPath + File.separator + uniqueFileName);

                    // Save the processed image
                    if (!newFile.exists()) {
                        FileOutputStream fos = new FileOutputStream(newFile);
                        predictedImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
                        fos.flush();
                        fos.close();
                    }
                }
            }
            callback.invoke(null, "Processed images saved successfully");
        }
    } catch (Exception e) {
        callback.invoke(e.toString(), null);
    }
    return true;
}



    @ReactMethod
    public boolean processAllImageProcess(String folderPath, Callback callback) {
    try {
        if (folderPath.startsWith("file:///")) {
            folderPath = folderPath.substring(7);
        }
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();

        // Create a new folder for copied images
        String allImagesFolderPath = folder.getAbsolutePath() + File.separator + "betterimagesnon_AI_GAUS";
        File allImagesFolder = new File(allImagesFolderPath);
        allImagesFolder.mkdirs(); // Make sure the directory exists

        // Load the TensorFlow Lite model
        
        if (listOfFiles != null) {
            for (File file : listOfFiles) {
                if (file.isFile() && isImageFile(file)) {
                    // Process the image
                    // processImage();
                    
                    // Bitmap image = BitmapFactory.decodeFile(file.getAbsolutePath());
                    // Bitmap predictedImage = Bitmap.createScaledBitmap(image, 50, 50, true);
                    Bitmap predictedImage = BitmapFactory.decodeFile(file.getAbsolutePath());
                    int heightt = predictedImage.getHeight();
                    int widtth = predictedImage.getWidth();
                    // Bitmap inputImage = BitmapFactory.decodeFile(imagePath)

                    // for (int i = 0; i < 3; i++) {
                    
                    // predictedImage = performInference(predictedImage, tfLite);
                    
                    // predictedImage =Bitmap.createScaledBitmap(predictedImage, widtth, heightt, true);
        
                    // }
                    predictedImage=sharpenImage(predictedImage);

                    // Bitmap predictedImage = performInference(predictedImage, tfLite);
                    // Bitmap predictedImage = processImage(file.getAbsolutePath(), tfLite);

                    // Generate a unique filename
                    String fileName = file.getName();
                    String[] fileNameParts = fileName.split("\\.");
                    String baseName = fileNameParts[0];
                    String extension = fileNameParts[1];
                    String uniqueFileName = baseName + "_generated_Image16." + extension;

                    File newFile = new File(allImagesFolderPath + File.separator + uniqueFileName);

                    // Save the processed image
                    if (!newFile.exists()) {
                        FileOutputStream fos = new FileOutputStream(newFile);
                        predictedImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
                        fos.flush();
                        fos.close();
                    }
                }
            }
            callback.invoke(null, "Processed images saved successfully");
        }
    } catch (Exception e) {
        callback.invoke(e.toString(), null);
    }
    return true;
}

@ReactMethod
public boolean processAllImageProcess2(String folderPath, Callback callback) {
    try {
        if (folderPath.startsWith("file:///")) {
            folderPath = folderPath.substring(7);
        }
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();

        // Create a new folder for copied images
        String allImagesFolderPath = folder.getAbsolutePath() + File.separator + "betterimagesnon_AI_image_proc_inhance";
        File allImagesFolder = new File(allImagesFolderPath);
        allImagesFolder.mkdirs(); // Make sure the directory exists

        // Load the TensorFlow Lite model
        
        if (listOfFiles != null) {
            for (File file : listOfFiles) {
                if (file.isFile() && isImageFile(file)) {
                    // Process the image
                    // processImage();
                    
                    // Bitmap image = BitmapFactory.decodeFile(file.getAbsolutePath());
                    // Bitmap predictedImage = Bitmap.createScaledBitmap(image, 50, 50, true);
                    Bitmap predictedImage = BitmapFactory.decodeFile(file.getAbsolutePath());
                    int heightt = predictedImage.getHeight();
                    int widtth = predictedImage.getWidth();
                    // Bitmap inputImage = BitmapFactory.decodeFile(imagePath)

                    // for (int i = 0; i < 3; i++) {
                    
                    // predictedImage = performInference(predictedImage, tfLite);
                    
                    // predictedImage =Bitmap.createScaledBitmap(predictedImage, widtth, heightt, true);
        
                    // }
                    predictedImage=sharpenImage2(predictedImage);

                    // Bitmap predictedImage = performInference(predictedImage, tfLite);
                    // Bitmap predictedImage = processImage(file.getAbsolutePath(), tfLite);

                    // Generate a unique filename
                    String fileName = file.getName();
                    String[] fileNameParts = fileName.split("\\.");
                    String baseName = fileNameParts[0];
                    String extension = fileNameParts[1];
                    String uniqueFileName = baseName + "_generated_Image16." + extension;

                    File newFile = new File(allImagesFolderPath + File.separator + uniqueFileName);

                    // Save the processed image
                    if (!newFile.exists()) {
                        FileOutputStream fos = new FileOutputStream(newFile);
                        predictedImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
                        fos.flush();
                        fos.close();
                    }
                }
            }
            callback.invoke(null, "Processed images saved successfully");
        }
    } catch (Exception e) {
        callback.invoke(e.toString(), null);
    }
    return true;
}





    private void copyImage(String sourcePath, String destinationPath, String imageName) {
        try {
            File sourceFile = new File(sourcePath);
            File destinationFile = new File(destinationPath, imageName);
    
            FileInputStream inputStream = new FileInputStream(sourceFile);
            FileOutputStream outputStream = new FileOutputStream(destinationFile);
    
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }
    
            inputStream.close();
            outputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    
    @Override
    public String getName() {
        return "ImageSharpen";
    }

    // @ReactMethod
    public Bitmap sharpenImage(Bitmap bitmap) {
        // try {
        //     if (imagePath.startsWith("file:///")) {
        //         imagePath = imagePath.substring(7);
        //     }
            // Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            Bitmap blurredBitmap = lightweightBlur(bitmap);
            Bitmap sharpenedBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());

            IntStream.range(1, bitmap.getWidth() - 1).parallel().forEach(x -> {
                for (int y = 1; y < bitmap.getHeight() - 1; y++) {
                    int originalPixel = bitmap.getPixel(x, y);
                    int blurredPixel = blurredBitmap.getPixel(x, y);

                    int origRed = Color.red(originalPixel);
                    int origGreen = Color.green(originalPixel);
                    int origBlue = Color.blue(originalPixel);

                    int blurredRed = Color.red(blurredPixel);
                    int blurredGreen = Color.green(blurredPixel);
                    int blurredBlue = Color.blue(blurredPixel);

                    int sharpenedRed = Math.min(255, (int) (1.3 * origRed - 0.5 * blurredRed));
                    int sharpenedGreen = Math.min(255, (int) (1.3 * origGreen - 0.5 * blurredGreen));
                    int sharpenedBlue = Math.min(255, (int) (1.3 * origBlue - 0.5 * blurredBlue));

                    sharpenedBitmap.setPixel(x, y, Color.rgb(sharpenedRed, sharpenedGreen, sharpenedBlue));
                }
            });

         
            // FileOutputStream out = new FileOutputStream(new File(imagePath));
            // sharpenedBitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            // out.close();

            return sharpenedBitmap;
    }



    public Bitmap sharpenImage2(Bitmap bitmap) {
        // try {
        //     if (imagePath.startsWith("file:///")) {
        //         imagePath = imagePath.substring(7);
        //     }
            // Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            Bitmap blurredBitmap = lightweightBlur(bitmap);
            Bitmap sharpenedBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());

            IntStream.range(1, bitmap.getWidth() - 1).parallel().forEach(x -> {
                for (int y = 1; y < bitmap.getHeight() - 1; y++) {
                    int originalPixel = bitmap.getPixel(x, y);
                    int blurredPixel = blurredBitmap.getPixel(x, y);

                    int origRed = Color.red(originalPixel);
                    int origGreen = Color.green(originalPixel);
                    int origBlue = Color.blue(originalPixel);

                    int blurredRed = Color.red(blurredPixel);
                    int blurredGreen = Color.green(blurredPixel);
                    int blurredBlue = Color.blue(blurredPixel);

                    int sharpenedRed = Math.min(255, (int) (1.4 * origRed - 0.5 * blurredRed));
                    int sharpenedGreen = Math.min(255, (int) (1.3 * origGreen - 0.5 * blurredGreen));
                    int sharpenedBlue = Math.min(255, (int) (1.3 * origBlue - 0.5 * blurredBlue));

                    sharpenedBitmap.setPixel(x, y, Color.rgb(sharpenedRed, sharpenedGreen, sharpenedBlue));
                }
            });
            return sharpenedBitmap;
    }

    @ReactMethod
    public void rankImages(String folderPath, Callback callback) {
        try {
            if (folderPath.startsWith("file:///")) {
                folderPath = folderPath.substring(7);
            }
            File folder = new File(folderPath);
            File[] listOfFiles = folder.listFiles();

            ArrayList<ImageRank> imageRanks = new ArrayList<>();

            if (listOfFiles != null) {
                for (File file : listOfFiles) {
                    if (file.isFile() && isImageFile(file)) {
                        int qualityScore = calculateImageQuality(file);
                        int sizeScore = calculateImageSize(file);

                        ImageRank imageRank = new ImageRank(file.getAbsolutePath(), qualityScore, sizeScore);
                        imageRanks.add(imageRank);
                    }
                }
            }

            
            Collections.sort(imageRanks);


            callback.invoke(null, "Images ranked successfully!");

        } catch (Exception e) {
            callback.invoke(e.toString(), null);
        }
    }

    /**
     * @param imageFile
     * @return
     */
    private static int calculateImageQuality(File imageFile)
     {
         try
          {

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(imageFile.getAbsolutePath(), options);

        int imageWidth = options.outWidth;
        int imageHeight = options.outHeight;


        int qualityScore = imageWidth * imageHeight;

        return qualityScore;

    } catch (Exception e) {
        e.printStackTrace();
        return 0;  
    }
}

// public Bitmap processImage(String lowImagePath, Interpreter interpreter) {
// }




private static int calculateImageSize(File imageFile) {
    try {
                
        long fileSize = imageFile.length();
        int sizeScore = (int) fileSize;

        return sizeScore;

    } catch (Exception e) {
        e.printStackTrace();
        return 0;  
    }
}
 

    private static class ImageRank implements Comparable<ImageRank> {
        String imagePath;
        int qualityScore;
        int sizeScore;

        public ImageRank(String imagePath, int qualityScore, int sizeScore) {
            this.imagePath = imagePath;
            this.qualityScore = qualityScore;
            this.sizeScore = sizeScore;
        }

        @Override
        public int compareTo(ImageRank other) {
            // Customize the comparison logic based on your criteria
            // This example compares based on quality score and then size score
            if (this.qualityScore != other.qualityScore) {
                return Integer.compare(this.qualityScore, other.qualityScore);
            } else {
                return Integer.compare(this.sizeScore, other.sizeScore);
            }
        }
    }


    private static int isImageBlurred(File imageFile) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath(), options);
    
        Bitmap grayBitmap = toGrayscale(bitmap);
    
        double variance = calculateVariance(grayBitmap);
    
        return (int) variance ;
    }




    // @ReactMethod
    // public void scanAndDelete(String folderPath, Callback callback) {
    // try {
    //     if (folderPath.startsWith("file:///")) {
    //         folderPath = folderPath.substring(7);
    //     }
    //     File folder = new File(folderPath);
    //     File[] listOfFiles = folder.listFiles();

    //     // Create a new folder for copied images
    //     String allImagesFolderPath = folder.getParent() + File.separator + "AllImages";
    //     File allImagesFolder = new File(allImagesFolderPath);
    //     allImagesFolder.mkdirs();

    //     ArrayList<ImageSize> imageSizes = new ArrayList<>();

    //     if (listOfFiles != null) {
    //         for (File file : listOfFiles) {
    //             if (file.isFile() && isImageFile(file)) {
    //                 // Copy the image to the new folder
    //                 copyImage(file.getAbsolutePath(), allImagesFolderPath, file.getName());

    //                 long fileSize = file.length();
    //                 ImageSize imageSize = new ImageSize(file.getAbsolutePath(), fileSize);
    //                 imageSizes.add(imageSize);
    //             }
    //         }
    //     }

    //     // Sort the images based on size in descending order
    //     Collections.sort(imageSizes, Collections.reverseOrder());

    //     // Log the sizes before deletion
    //     for (ImageSize size : imageSizes) {
    //         System.out.println("Image Path: " + size.imagePath + ", Size: " + size.size);
    //     }

    //     // Keep only the top three images with the largest size
    //     int topImagesCount = Math.min(3, imageSizes.size());
    //     ArrayList<String> topThreeImagePaths = new ArrayList<>();
    //     for (int i = 0; i < topImagesCount; i++) {
    //         topThreeImagePaths.add(imageSizes.get(i).imagePath);
    //     }

    //     // Delete files that are not in the top three
    //     for (File file : listOfFiles) {
    //         if (file.isFile() && isImageFile(file)) {
    //             String filePath = file.getAbsolutePath();
    //             if (!topThreeImagePaths.contains(filePath)) {
    //                 if (file.delete()) {
    //                     System.out.println("Deleted image: " + file.getName());
    //                 } else {
    //                     System.out.println("Failed to delete image: " + file.getName());
    //                 }
    //             }
    //         }
    //     }

    //     callback.invoke(null, "Kept the three largest images and deleted the rest.");

    // } catch (Exception e) {
    //     callback.invoke(e.toString(), null);
    // }
//}


@ReactMethod
public void scanAndDelete(String folderPath, Callback callback) {
    try {
        // Adjust the folder path if it starts with "file:///"
        if (folderPath.startsWith("file:///")) {
            folderPath = folderPath.substring(7);
        }
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();

        // Create a new folder for results
        String resultsFolderPath = folder.getParent() + File.separator + "results";
        File resultsFolder = new File(resultsFolderPath);
        resultsFolder.mkdirs();

        ArrayList<ImageFile> imageFiles = new ArrayList<>();

        if (listOfFiles != null) {
            for (File file : listOfFiles) {
                if (file.isFile() && isImageFile(file)) {
                    long fileSize = file.length();
                    long lastModified = file.lastModified();
                    ImageFile imageFile = new ImageFile(file.getAbsolutePath(), fileSize, lastModified);
                    imageFiles.add(imageFile);
                }
            }
        }

        // Sort the images based on last modified date in descending order
        Collections.sort(imageFiles, (a, b) -> Long.compare(b.lastModified, a.lastModified));

        // Log the file info before copying
        for (ImageFile image : imageFiles) {
            System.out.println("Image Path: " + image.imagePath + ", Size: " + image.size + ", Last Modified: " + image.lastModified);
        }

        // Copy only the top three most recent images to the results folder
        int topImagesCount = Math.min(3, imageFiles.size());
        for (int i = 0; i < topImagesCount; i++) {
            String imagePath = imageFiles.get(i).imagePath;
            File imageFile = new File(imagePath);
            copyImage(imageFile.getAbsolutePath(), resultsFolderPath, imageFile.getName());
        }

        callback.invoke(null, "Copied the three most recent images to the results folder.");

    } catch (Exception e) {
        callback.invoke(e.toString(), null);
    }
}

// Class to hold image file details
class ImageFile {
    String imagePath;
    long size;
    long lastModified;

    ImageFile(String imagePath, long size, long lastModified) {
        this.imagePath = imagePath;
        this.size = size;
        this.lastModified = lastModified;
    }
}

    
    private static class ImageSize implements Comparable<ImageSize> {
        String imagePath;
        long size;
    
        public ImageSize(String imagePath, long size) {
            this.imagePath = imagePath;
            this.size = size;
        }
    
        @Override
        public int compareTo(ImageSize other) {
            // Customize the comparison logic based on size
            return Long.compare(this.size, other.size);
        }
    }

    private static boolean isImageFile(File file) {
        String fileName = file.getName();
        return fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".png");
    }
    private static Bitmap toGrayscale(Bitmap bmpOriginal) {
        int width = bmpOriginal.getWidth();
        int height = bmpOriginal.getHeight();
    
        int[] pixels = new int[width * height];
        bmpOriginal.getPixels(pixels, 0, width, 0, 0, width, height);
    
        int[] grayscalePixels = IntStream.range(0, pixels.length)
                .map(i -> {
                    int color = pixels[i];
                    int red = Color.red(color);
                    int green = Color.green(color);
                    int blue = Color.blue(color);
                    int gray = (int) (0.299 * red + 0.587 * green + 0.114 * blue);
                    return Color.rgb(gray, gray, gray);
                })
                .toArray();
    
        Bitmap bmpGrayscale = Bitmap.createBitmap(grayscalePixels, width, height, Bitmap.Config.ARGB_8888);
        return bmpGrayscale;
    }

    private static double calculateVariance(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
    
        double sum = 0.0;
        double sumSquaredDiff = 0.0;
    
        for (int pixel : pixels) {
            int grayValue = Color.red(pixel);  // Assuming grayscale, change accordingly if not
            sum += grayValue;
            sumSquaredDiff += Math.pow(grayValue, 2);
        }
    
        double mean = sum / (width * height);
        double variance = sumSquaredDiff / (width * height) - Math.pow(mean, 2);
    
        return variance;
    }
    

    private Bitmap lightweightBlur(Bitmap original) {
        Bitmap blurred = Bitmap.createBitmap(original.getWidth(), original.getHeight(), original.getConfig());
        IntStream.range(1, original.getWidth() - 1).parallel().forEach(x -> {
            for (int y = 1; y < original.getHeight() - 1; y++) {
                int surroundingPixels[] = new int[]{
                        original.getPixel(x - 1, y - 1), original.getPixel(x, y - 1), original.getPixel(x + 1, y - 1),
                        original.getPixel(x - 1, y), original.getPixel(x, y), original.getPixel(x + 1, y),
                        original.getPixel(x - 1, y + 1), original.getPixel(x, y + 1), original.getPixel(x + 1, y + 1),
                };
                blurred.setPixel(x, y, averageColor(surroundingPixels));
            }
        });
        return blurred;
    }

    private int averageColor(int[] colors) {
        int r = 0, g = 0, b = 0;
        for (int color : colors) {
            r += Color.red(color);
            g += Color.green(color);
            b += Color.blue(color);
        }
        r /= colors.length;
        g /= colors.length;
        b /= colors.length;
        return Color.rgb(r, g, b);
    }

}











































































































































































