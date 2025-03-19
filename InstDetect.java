package com.eyescan.facedetectframeprocessor;
 
import com.facebook.react.bridge.ReactMethod;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;
import android.util.Size;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


 
public class InstDetect {

    private final Size INPNUT_SIZE = new Size(320, 320);
    private int[] OUTPUT_SIZE;
    //private Boolean IS_INT8 = true;
    private final float DETECT_THRESHOLD = 0.6f;
    private final float IOU_THRESHOLD = 0.55f;
    private final float IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f;
    MetadataExtractor.QuantizationParams eyesInput5SINT8QuantParams;
    MetadataExtractor.QuantizationParams eyesOutput5SINT8QuantParams;
    MetadataExtractor.QuantizationParams faceInput5SINT8QuantParams;
    MetadataExtractor.QuantizationParams faceOutput5SINT8QuantParams;
    private String LABEL_FILE = "";
    private String MODEL_FILE = "";
    private Interpreter tfliteFace;
    private Interpreter tfliteEyes;
    private List<String> faceLabels;
    private List<String> eyesLabels;
    Interpreter.Options options = new Interpreter.Options();
    Context context;

    public InstDetect(Context context) {
        Log.i("FaceDetect", "Hello from init Constructor");
        this.context = context;
    }
 
    @ReactMethod
    public void initialize() {
        try {
            MODEL_FILE = "quantized_model.tflite";
            LABEL_FILE = "coco_label_face.txt";
            faceOutput5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.004016838036477566f, 2);
            faceInput5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);
            ByteBuffer tfliteModel = FileUtil.loadMappedFile(context, MODEL_FILE);
            tfliteFace = new Interpreter(tfliteModel, options);
            //successCallback.invoke("Success reading model: " + MODEL_FILE);
            faceLabels = FileUtil.loadLabels(context, LABEL_FILE);
            //Toast.makeText(context, associatedAxisLabels.get(0), Toast.LENGTH_LONG).show();
            Log.i("YOLO DETECTOR", faceLabels.get(0));
            //successCallback.invoke("Hello from Java ");

            MODEL_FILE = "best-int8.tflite";
            LABEL_FILE = "coco_label.txt";
            eyesInput5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 2);
            eyesOutput5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.017137449234724045f, 1);
            tfliteModel = FileUtil.loadMappedFile(context, MODEL_FILE);
            tfliteEyes = new Interpreter(tfliteModel, options);
            //successCallback.invoke("Success reading model: " + MODEL_FILE);
            eyesLabels = FileUtil.loadLabels(context, LABEL_FILE);
            //Toast.makeText(context, associatedAxisLabels.get(0), Toast.LENGTH_LONG).show();
            Log.i("YOLO DETECTOR", eyesLabels.get(0));
        } catch (IOException e) {
            //errorCallback.invoke("Error reading model or label: " + e.getMessage());
            //Toast.makeText(context, "load model error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }
    }

    public Size getInputSize(){return this.INPNUT_SIZE;}

    public ArrayList<Recognition> detect(Bitmap bitmap, String modelName) {
        Interpreter tflite = null;
        List<String> associatedAxisLabels = null;
        MetadataExtractor.QuantizationParams input5SINT8QuantParams = null;
        MetadataExtractor.QuantizationParams output5SINT8QuantParams = null;
        if(modelName.equals("face")){
            tflite = tfliteFace;
            OUTPUT_SIZE = new int[] { 1, 25200, 6 };
            associatedAxisLabels = faceLabels;
            input5SINT8QuantParams = faceInput5SINT8QuantParams;
            output5SINT8QuantParams = faceOutput5SINT8QuantParams;
        }
        else if(modelName.equals("eyes")){
            tflite = tfliteEyes;
            OUTPUT_SIZE = new int[] { 1, 25200, 7 };
            associatedAxisLabels = eyesLabels;
            input5SINT8QuantParams = eyesInput5SINT8QuantParams;
            output5SINT8QuantParams = eyesOutput5SINT8QuantParams;
        }
        // yolov5s-tflite的输入是:[1, 320, 320,3], 摄像头每一帧图片需要resize,再归一化
        // The input of yolov5s-tflite is: [1, 320, 320,3], each frame of the camera needs to be resized and then normalized
        TensorImage yolov5sTfliteInput;
        ImageProcessor imageProcessor;
        imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPNUT_SIZE.getHeight(), INPNUT_SIZE.getWidth(), ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .add(new QuantizeOp(input5SINT8QuantParams.getZeroPoint(), input5SINT8QuantParams.getScale()))
                        .add(new CastOp(DataType.UINT8))
                        .build();
        yolov5sTfliteInput = new TensorImage(DataType.UINT8);

        yolov5sTfliteInput.load(bitmap);
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput);

        // yolov5s-tflite的输出是:[1, 6300, 85], 可以从v5的GitHub release处找到相关tflite模型, 输出是[0,1], 处理到320.
        // The output of yolov5s-tflite is: [1, 6300, 85], you can find the relevant tflite model from the GitHub release of v5, the output is [0,1], processed to 320
        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(OUTPUT_SIZE, DataType.UINT8);

        // 推理计算
        if (null != tflite) {
            // 这里tflite默认会加一个batch=1的纬度
            // Here tflite will add a batch=1 latitude by default
//            long asd=System.nanoTime();
            tflite.run(yolov5sTfliteInput.getBuffer(), probabilityBuffer.getBuffer());

//             asd=System.nanoTime()-asd;
//            Log.println(Log.DEBUG, "timefor detection", "Ycamereapressed " + asd);
        }

        // 这里输出反量化,需要是模型tflite.run之后执行.
        // The output dequantization here needs to be executed after the model tflite.run.
        TensorProcessor tensorProcessor = new TensorProcessor.Builder()
                .add(new DequantizeOp(output5SINT8QuantParams.getZeroPoint(), output5SINT8QuantParams.getScale()))
                .build();
        probabilityBuffer = tensorProcessor.process(probabilityBuffer);

        // 输出数据被平铺了出来
        // The output data is tiled out
        float[] recognitionArray = probabilityBuffer.getFloatArray();
        // 这里将flatten的数组重新解析(xywh,obj,classes).
        // Here the flatten array is re-parsed (xywh, obj, classes).
        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        for (int i = 0; i < OUTPUT_SIZE[1]; i++) {
            int gridStride = i * OUTPUT_SIZE[2];
            // 由于yolov5作者在导出tflite的时候对输出除以了image size, 所以这里需要乘回去
            // Since the author of yolov5 divided the output by the image size when exporting tflite, it needs to be multiplied here
            float x = recognitionArray[0 + gridStride] * INPNUT_SIZE.getWidth();
            float y = recognitionArray[1 + gridStride] * INPNUT_SIZE.getHeight();
            float w = recognitionArray[2 + gridStride] * INPNUT_SIZE.getWidth();
            float h = recognitionArray[3 + gridStride] * INPNUT_SIZE.getHeight();
            int xmin = (int) Math.max(0, x - w / 2.);
            int ymin = (int) Math.max(0, y - h / 2.);
            int xmax = (int) Math.min(INPNUT_SIZE.getWidth(), x + w / 2.);
            int ymax = (int) Math.min(INPNUT_SIZE.getHeight(), y + h / 2.);
            float confidence = recognitionArray[4 + gridStride];
            float[] classScores = Arrays.copyOfRange(recognitionArray, 5 + gridStride, this.OUTPUT_SIZE[2] + gridStride);
//            if(i % 1000 == 0){
//                Log.i("tfliteSupport","x,y,w,h,conf:"+x+","+y+","+w+","+h+","+confidence);
//            }
//            Log.i("","Laura"+ recognitionArray[6 + gridStride]);

            int labelId = 0;
            float maxLabelScores = 0.f;
            for (int j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxLabelScores) {
                    maxLabelScores = classScores[j];
                    labelId = j;
                }
            }

            Recognition r = new Recognition(
                    labelId,
                    "",
                    maxLabelScores,
                    confidence,
                    new RectF(xmin, ymin, xmax, ymax));
            allRecognitions.add(
                    r);
        }
//        Log.i("tfliteSupport", "recognize data size: "+allRecognitions.size());

        // 非极大抑制输出
        // Non-maximum suppression output
        ArrayList<Recognition> nmsRecognitions = nms(allRecognitions);
        // 第二次非极大抑制, 过滤那些同个目标识别到2个以上目标边框为不同类别的
        // The second non-maximum suppression,
        // filter those objects that recognize more than 2 target borders of different categories for the same target
        ArrayList<Recognition> nmsFilterBoxDuplicationRecognitions = nmsAllClass(nmsRecognitions);

        // 更新label信息
        // Update label information
        for(Recognition recognition : nmsFilterBoxDuplicationRecognitions){
            int labelId = recognition.getLabelId();
            String labelName = associatedAxisLabels.get(labelId);
            recognition.setLabelName(labelName);
        }

        return nmsFilterBoxDuplicationRecognitions;
    }

    /**
     * 非极大抑制
     *
     * @param allRecognitions
     * @return
     */
    protected ArrayList<Recognition> nms(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<Recognition>();
        //getLocation
        // 遍历每个类别, 在每个类别下做nms
        // traverse each category, do nms under each category
        for (int i = 0; i < OUTPUT_SIZE[2]-5; i++) {
            // 这里为每个类别做一个队列, 把labelScore高的排前面
            // Make a queue for each category here, and put the labelScore high in the front
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            6300,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition l, final Recognition r) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(r.getConfidence(), l.getConfidence());
                                }
                            });

            // 相同类别的过滤出来, 且obj要大于设定的阈值
            // Filter out the same category, and obj must be greater than the set threshold
            for (int j = 0; j < allRecognitions.size(); ++j) {
//                if (allRecognitions.get(j).getLabelId() == i) {
                if (allRecognitions.get(j).getLabelId() == i && allRecognitions.get(j).getConfidence() > DETECT_THRESHOLD) {
                    pq.add(allRecognitions.get(j));
//                    Log.i("tfliteSupport", allRecognitions.get(j).toString());
                }
            }

            // nms循环遍历
            // nms loop through

            while (pq.size() > 0) {
                // 概率最大的先拿出来
                // Take out the one with the highest probability first
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsRecognitions.add(max);
                pq.clear();

                for (int k = 1; k < detections.length; k++) {
                    Recognition detection = detections[k];
                    if (boxIou(max.getLocation(), detection.getLocation()) < IOU_THRESHOLD) {
                        pq.add(detection);
                    }
                }
            }

        }
        return nmsRecognitions;
    }

    /**
     * 对所有数据不区分类别做非极大抑制
     *
     * @param allRecognitions
     * @return
     */
    protected ArrayList<Recognition> nmsAllClass(ArrayList<Recognition> allRecognitions) {
        ArrayList<Recognition> nmsRecognitions = new ArrayList<Recognition>();

        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        100,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition l, final Recognition r) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(r.getConfidence(), l.getConfidence());
                            }
                        });

        // 相同类别的过滤出来, 且obj要大于设定的阈值
        // Filter out the same category, and obj must be greater than the set threshold
        for (int j = 0; j < allRecognitions.size(); ++j) {
            if (allRecognitions.get(j).getConfidence() > DETECT_THRESHOLD) {
                pq.add(allRecognitions.get(j));
            }
        }

        while (pq.size() > 0) {
            // 概率最大的先拿出来
            // Take out the one with the highest probability first
            Recognition[] a = new Recognition[pq.size()];
            Recognition[] detections = pq.toArray(a);
            Recognition max = detections[0];
            nmsRecognitions.add(max);
            pq.clear();

            for (int k = 1; k < detections.length; k++) {
                Recognition detection = detections[k];
                if (boxIou(max.getLocation(), detection.getLocation()) < IOU_CLASS_DUPLICATED_THRESHOLD) {
                    pq.add(detection);
                }
            }
        }
        return nmsRecognitions;
    }


    protected float boxIou(RectF a, RectF b) {
        float intersection = boxIntersection(a, b);
        float union = boxUnion(a, b);
        if (union <= 0) return 1;
        return intersection / union;
    }

    protected float boxIntersection(RectF a, RectF b) {
        float maxLeft = a.left > b.left ? a.left : b.left;
        float maxTop = a.top > b.top ? a.top : b.top;
        float minRight = a.right < b.right ? a.right : b.right;
        float minBottom = a.bottom < b.bottom ? a.bottom : b.bottom;
        float w = minRight -  maxLeft;
        float h = minBottom - maxTop;

        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float boxUnion(RectF a, RectF b) {
        float i = boxIntersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    /**
     * 添加NNapi代理
     */
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(true);
//            nnApiOptions.setUseNnapiCpu(true);
            //ANEURALNETWORKS_PREFER_LOW_POWER：倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
            //ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
            //ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            nnApiDelegate = new NnApiDelegate(nnApiOptions);
            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
            Log.i("tfliteSupport", "using nnapi delegate.");
        }
    }

    /**
     * 添加GPU代理
     */
    public void addGPUDelegate() {
        CompatibilityList compatibilityList = new CompatibilityList();
        if(compatibilityList.isDelegateSupportedOnThisDevice()){
            GpuDelegate.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i("tfliteSupport", "\n==================\nusing gpu delegate.\n==================\n");
        } else {
            addThread(4);
        }
    }

    /**
     * 添加线程数
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }
}