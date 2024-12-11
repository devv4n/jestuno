package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.*
import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainBinding
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// Top-level extension function for rotating a Bitmap
fun Bitmap.rotate(degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private lateinit var binding: ActivityMainBinding

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val labelData: List<String> by lazy { readLabels() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

    private lateinit var tts: TextToSpeech
    private var isSpeaking: Boolean = false
    private val textBuffer: StringBuilder = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        // Initialize OrtEnvironment
        try {
            ortEnv = OrtEnvironment.getEnvironment()
        } catch (e: OrtException) {
            Log.e(TAG, "Failed to initialize OrtEnvironment", e)
            Toast.makeText(this, "Failed to initialize ONNX Runtime.", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        // Initialize TextToSpeech
        tts = TextToSpeech(this, this)

        // Check permissions and start camera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Set button listeners
        binding.button2.setOnClickListener {
            switchCamera()
        }

        binding.speakButton.setOnClickListener {
            isSpeaking = !isSpeaking
            if (isSpeaking) {
                startTextToSpeech()
            } else {
                stopTextToSpeech()
            }
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale("ru", "RU"))
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e(TAG, "Русский язык не поддерживается")
                Toast.makeText(this, "Русский язык не поддерживается TextToSpeech.", Toast.LENGTH_LONG).show()
            }
        } else {
            Log.e(TAG, "Инициализация TextToSpeech не удалась!")
            Toast.makeText(this, "Не удалось инициализировать TextToSpeech.", Toast.LENGTH_LONG).show()
        }
    }

    private fun startTextToSpeech() {
        scope.launch {
            tts.stop()

            while (isSpeaking) {
                val newText = getNextText()
                newText?.let {
                    tts.speak(it, TextToSpeech.QUEUE_ADD, null, null)
                }
                delay(1000)
            }
        }
    }

    private fun stopTextToSpeech() {
        isSpeaking = false
        tts.stop()
    }

    private fun getNextText(): String? {
        val fullText = binding.detectedItemValue1.text.toString()
        if (fullText.isNotBlank() && fullText.trim().split(" ").size > textBuffer.toString().trim().split(" ").size) {
            val newText = fullText.trim().split(" ").drop(textBuffer.toString().trim().split(" ").size).joinToString(" ")
            textBuffer.append(" $newText")
            return newText.trim()
        }
        return null
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build()

            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

            setORTAnalyzer()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun switchCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) {
            CameraSelector.DEFAULT_BACK_CAMERA
        } else {
            CameraSelector.DEFAULT_FRONT_CAMERA
        }

        startCamera()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        isSpeaking = false
        tts.stop()
        tts.shutdown()
        backgroundExecutor.shutdown()
        ortSession?.close()
        ortEnv?.close()
        scope.cancel()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this, "Permissions not granted by the user.", Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private val lastDetectedItems = mutableListOf<String>() // Список для хранения последних добавленных элементов
    private val maxRepetitions = 1 // Максимальное количество повторений одного элемента

    private fun updateUI(result: Result) {
        if (result.detectedIndices.isEmpty() || result.detectedIndices[0] > labelData.size - 1) {
            Log.w(TAG, "Detected index out of bounds: ${result.detectedIndices.firstOrNull()}")
            return
        }

        runOnUiThread {
            val detectedText = labelData[result.detectedIndices[0]]

            // Проверяем, не превышает ли текущий элемент лимит повторений
            if (lastDetectedItems.count { it == detectedText } < maxRepetitions) {
                binding.detectedItemValue1.append(detectedText)
                lastDetectedItems.add(detectedText)

                // Удаляем старые элементы, чтобы список не рос бесконечно
                if (lastDetectedItems.size > maxRepetitions * 2) {
                    lastDetectedItems.removeAt(0)
                }

                if (isSpeaking) {
                    tts.speak(detectedText, TextToSpeech.QUEUE_ADD, null, null)
                }
            } else {
                Log.d(TAG, "Skipped duplicate: $detectedText")
            }
        }
    }

    private fun readLabels(): List<String> {
        return resources.openRawResource(R.raw.imagenet_classes).bufferedReader().readLines()
    }

    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.IO) {
        try {
            val modelBytes = resources.openRawResource(R.raw.mobilenetv2_tsm).readBytes()
            val sessionOptions = OrtSession.SessionOptions()
            // Configure session options if needed
            return@withContext ortEnv?.createSession(modelBytes, sessionOptions)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create OrtSession", e)
            return@withContext null
        }
    }

    private fun setORTAnalyzer() {
        scope.launch {
            imageAnalysis?.clearAnalyzer()
            ortSession = createOrtSession()
            if (ortSession != null) {
                imageAnalysis?.setAnalyzer(
                    backgroundExecutor,
                    ORTAnalyzer(ortSession!!, ::updateUI, ortEnv!!)
                )
            } else {
                Log.e(TAG, "Failed to create OrtSession")
                Toast.makeText(this@MainActivity, "Failed to load the model.", Toast.LENGTH_LONG).show()
            }
        }
    }

    companion object {
        const val TAG = "Jestuno"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}

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

internal class ORTAnalyzer(
    private val ortSession: OrtSession,
    private val callBack: (Result) -> Unit,
    private val ortEnv: OrtEnvironment
) : ImageAnalysis.Analyzer {

    private val frameInterval = 3
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
                val rotatedBitmap = imgBitmap

                val result = Result()

                val imgData = imgDataPool.acquire()
                preProcess(rotatedBitmap, imgData)
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

                            // Фильтрация по порогу вероятности
                            val probabilityThreshold = 0.06f // Порог вероятности
                            val kek = probabilities[topIndex]
                            if (kek < probabilityThreshold) {
                                Log.d("ORTAnalyzer", "Skipped low probability result: $topIndex $kek")
                                return@launch
                            }

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

// Extension function to convert ImageProxy to Bitmap
private fun ImageProxy.toBitmapUsingBuffer(): Bitmap {
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