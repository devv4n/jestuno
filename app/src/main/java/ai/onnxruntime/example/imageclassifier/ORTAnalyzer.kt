package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.Collections


internal data class Result(
    var detectedIndices: List<Int> = emptyList(),
    var detectedScore: MutableList<Float> = mutableListOf(),
    var processTimeMs: Long = 0
)

class ObjectPool<T>(private val create: () -> T, private val poolSize: Int) {
    private val pool = mutableListOf<T>()

    init {
        repeat(poolSize) {
            pool.add(create())
        }
    }

    @Synchronized
    fun acquire(): T {
        return if (pool.isNotEmpty()) {
            pool.removeAt(pool.size - 1)
        } else {
            create()
        }
    }

    @Synchronized
    fun release(obj: T) {
        if (pool.size < poolSize) {
            pool.add(obj)
        }
    }
}

// Extension function to convert ImageProxy to Bitmap
fun ImageProxy.toBitmapUsingBuffer(): Bitmap {
    val nv21 = yuv420888ToNv21(this)
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 80, out)
    val imageBytes: ByteArray = out.toByteArray()
    val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    return Bitmap.createScaledBitmap(bitmap.rotate(imageInfo.rotationDegrees.toFloat()), 224, 224, false) // Resize to model input size
}

// Function to convert YUV_420_888 to NV21
fun yuv420888ToNv21(image: ImageProxy): ByteArray {
    val width = image.width
    val height = image.height
    val ySize = width * height
    val uvSize = width * height / 4

    val nv21 = ByteArray(ySize + uvSize * 2)

    val yBuffer = image.planes[0].buffer // Y
    val uBuffer = image.planes[1].buffer // U
    val vBuffer = image.planes[2].buffer // V

    var rowStride = image.planes[0].rowStride
    var pos = 0
    if (rowStride == width) {
        yBuffer.get(nv21, 0, ySize)
        pos += ySize
    } else {
        var yPos = 0
        for (i in 0 until height) {
            yBuffer.position(yPos)
            yBuffer.get(nv21, pos, width)
            pos += width
            yPos += rowStride
        }
    }

    rowStride = image.planes[2].rowStride
    val pixelStride = image.planes[2].pixelStride

    if (pixelStride == 2 && rowStride == width && width % 2 == 0) {
        vBuffer.get(nv21, ySize, uvSize)
        uBuffer.get(nv21, ySize + uvSize, uvSize)
    } else {
        for (i in 0 until height / 2) {
            for (j in 0 until width / 2) {
                nv21[ySize + i * width + j * 2] = vBuffer.get(i * rowStride + j * pixelStride)
                nv21[ySize + i * width + j * 2 + 1] = uBuffer.get(i * rowStride + j * pixelStride)
            }
        }
    }
    return nv21
}



internal class ORTAnalyzer(
    private val context: Context,
    private val ortSession: OrtSession,
    private val callBack: (Result) -> Unit,
    private val ortEnv: OrtEnvironment
) : ImageAnalysis.Analyzer {

    private val frameInterval = 5
    private var frameCounter = 0
    private val tensorsList = mutableListOf<FloatBuffer>()
    private val windowSize = 3
    private val inputShape = longArrayOf(1,  8, windowSize.toLong(), 224, 224)
    private val imgDataPool = ObjectPool(::createNewFloatBuffer, poolSize = 10)

    private fun createNewFloatBuffer(): FloatBuffer {
        return FloatBuffer.allocate(3 * 224 * 224)
    }

    private fun softMax(modelResult: FloatArray): FloatArray {
        val labelVals = modelResult.copyOf()
        val max = labelVals.maxOrNull() ?: 0.0f
        var sum = 0.0f

        for (i in labelVals.indices) {
            labelVals[i] = kotlin.math.exp(labelVals[i] - max)
            sum += labelVals[i]
        }

        if (sum != 0.0f) {
            for (i in labelVals.indices) {
                labelVals[i] /= sum
            }
        }

        return labelVals
    }

    override fun analyze(image: ImageProxy) {
        frameCounter++
        if (frameCounter % frameInterval != 0) {
            image.close()
            return
        }

        CoroutineScope(Dispatchers.Default).launch {
            try {
                val imgBitmap = image.toBitmapUsingBuffer()

                val result = Result()

                val imgData = imgDataPool.acquire()
                preProcess(imgBitmap, imgData)
                tensorsList.add(imgData)

                if (tensorsList.size >= windowSize) {
                    val inputTensor = FloatBuffer.allocate(1 * 8 * 3 * 224 * 224)

                    for (i in 0 until 8) {
                        tensorsList.forEach { tensor ->
                            inputTensor.put(tensor)
                        }
                    }
                    inputTensor.rewind()

                    tensorsList.forEach { imgDataPool.release(it) }
                    tensorsList.clear()

                    val inputName = ortSession.inputNames.firstOrNull()
                    if (inputName == null) {
                        Log.e("ORTAnalyzer", "No input name found in the model.")
                        return@launch
                    }

                    val tensor: OnnxTensor = OnnxTensor.createTensor(ortEnv, inputTensor, inputShape)
                    tensor.use { tr ->
                        val startTime = SystemClock.uptimeMillis()

                        val output: OrtSession.Result = ortSession.run(Collections.singletonMap(inputName, tr))
                        output.use { out ->
                            result.processTimeMs = SystemClock.uptimeMillis() - startTime

                            @Suppress("UNCHECKED_CAST")
                            val rawOutput = (out.get(0).value as Array<FloatArray>)[0]
                            val probabilities = softMax(rawOutput)
                            val topIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

                            val prob = probabilities[topIndex]
                            val res = labelData[topIndex]

                            if (showAllLetters) {
                                withContext(Dispatchers.Main) {
                                    Toast.makeText(context, "$res: %.2f%%".format(prob * 100), Toast.LENGTH_SHORT).show()
                                }
                            }

                            if (prob < devModeThreshold) {
                                Log.d("ORTAnalyzer", "Skipped low probability result: $res $prob")
                                return@launch
                            }

                            Log.d("ORTAnalyzer", "Add to result: $res $prob")

                            result.detectedIndices = listOf(topIndex)
                            result.detectedScore.add(probabilities.getOrNull(topIndex) ?: 0f)

                            withContext(Dispatchers.Main) {
                                callBack(result)
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("ORTAnalyzer", "Error during image analysis", e)
            } finally {
                image.close()
            }
        }
    }

    private fun preProcess(bitmap: Bitmap, imgData: FloatBuffer) {
        imgData.clear()
        val stride = 224 * 224
        val bmpData = IntArray(stride)
        bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val mean = floatArrayOf(123.675f, 116.28f, 103.53f)
        val std = floatArrayOf(58.395f, 57.12f, 57.375f)

        val rChannel = FloatArray(stride)
        val gChannel = FloatArray(stride)
        val bChannel = FloatArray(stride)

        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val idx = 224 * i + j
                val pixelValue = bmpData[idx]

                rChannel[idx] = (((pixelValue shr 16) and 0xFF) - mean[0]) / std[0]
                gChannel[idx] = (((pixelValue shr 8) and 0xFF) - mean[1]) / std[1]
                bChannel[idx] = ((pixelValue and 0xFF) - mean[2]) / std[2]
            }
        }

        imgData.put(rChannel)
        imgData.put(gChannel)
        imgData.put(bChannel)

        imgData.rewind()
    }
}