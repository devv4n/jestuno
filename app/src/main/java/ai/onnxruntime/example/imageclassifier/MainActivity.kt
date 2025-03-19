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
import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.view.MenuItem
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.Window
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.BaseAdapter
import android.widget.EditText
import android.widget.GridView
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.switchmaterial.SwitchMaterial
import java.io.IOException

// Top-level extension function for rotating a Bitmap
fun Bitmap.rotate(degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}

private lateinit var labelData: List<String>

fun initializeLabels(context: Context) {
    labelData = context.resources.openRawResource(R.raw.imagenet_classes).bufferedReader().readLines()
}

private var devModeThreshold = 0.067f
private var showAllLetters = false

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private lateinit var binding: ActivityMainBinding

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)


    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA


    private var hintToast: Toast? = null
    private var pressStartTime = 0L
    private val devModeHandler = Handler(Looper.getMainLooper())

    private lateinit var tts: TextToSpeech
    private val textBuffer: StringBuilder = StringBuilder()
    private var isSpeaking: Boolean = false
    private var isRecognitionActive = false


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

        binding.button2.setOnClickListener {
            switchCamera()
        }

        binding.speakButton1.setOnClickListener {
            isSpeaking=!isSpeaking
            if (isSpeaking) {
                startTextToSpeech()

                binding.speakButton1.setImageResource(R.drawable.mute__1_)
                Toast.makeText(this, "Озвучивание включено", Toast.LENGTH_SHORT).show()
            } else {
                stopTextToSpeech()

                binding.speakButton1.setImageResource(R.drawable.volume__1_)
                Toast.makeText(this, "Озвучивание выключено", Toast.LENGTH_SHORT).show()
            }
        }

        binding.startButton.setOnLongClickListener {
            binding.detectedItemValue1.text = ""
            Toast.makeText(this, "Очищено", Toast.LENGTH_SHORT).show()
            true
        }

        binding.startButton.setOnClickListener {
            if (isRecognitionActive) {
                stopRecognition() // Остановка распознавания
                binding.startButton.setImageResource(android.R.drawable.ic_media_play)
            } else {
                startRecognition() // Запуск распознавания
                binding.startButton.setImageResource(android.R.drawable.ic_media_pause)

                Toast.makeText(this, "Для очистки удерживайте", Toast.LENGTH_SHORT).show()
            }
        }

        initializeLabels(this)

        binding.helpButton.setOnClickListener {
            val intent = Intent(this, GalleryActivity::class.java)
            startActivity(intent)
        }

        binding.helpButton.setOnTouchListener { v, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    // Начало долгого нажатия
                    pressStartTime = System.currentTimeMillis()
                    startDevModeCountdown()
                    true // Возвращаем true, чтобы обработать событие
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    // Завершение долгого нажатия
                    val pressDuration = System.currentTimeMillis() - pressStartTime
                    devModeHandler.removeCallbacksAndMessages(null)
                    hintToast?.cancel()

                    if (pressDuration < 5000) { // Если удержание меньше 5 секунд
                        // Это обычный клик, вызываем performClick()
                        v.performClick()
                    }
                    false // Возвращаем false, чтобы событие передалось в setOnClickListener
                }
                else -> false
            }
        }
    }

    private fun startRecognition() {
        isRecognitionActive = true
        setORTAnalyzer()
    }

    private fun stopRecognition() {
        isRecognitionActive = false
        imageAnalysis?.clearAnalyzer()
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
        // Сбрасываем буфер текста, чтобы начать чтение с начала
        textBuffer.clear()

        scope.launch {
            tts.stop() // Останавливаем текущее воспроизведение

            while (isSpeaking) {
                val newText = getNextText()
                newText?.let {
                    tts.speak(it, TextToSpeech.QUEUE_ADD, null, null)
                }
                delay(1000) // Задержка между воспроизведением частей текста
            }
        }
    }

    private fun getNextText(): String? {
        val fullText = binding.detectedItemValue1.text.toString()

        // Если текст на экране не пустой и буфер пуст, начинаем с начала
        if (fullText.isNotBlank() && textBuffer.isEmpty()) {
            textBuffer.append(fullText)
            return fullText.trim()
        }

        // Если текст на экране изменился, обновляем буфер
        if (fullText.isNotBlank() && fullText.trim() != textBuffer.toString().trim()) {
            textBuffer.clear()
            textBuffer.append(fullText)
            return fullText.trim()
        }

        return null
    }

    private fun stopTextToSpeech() {
        isSpeaking = false
        tts.stop()
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

                // Воспроизводим текст только если кнопка воспроизведения активна
                if (isSpeaking) {
                    tts.speak(detectedText, TextToSpeech.QUEUE_ADD, null, null)
                }
            } else {
                Log.d(TAG, "Skipped duplicate: $detectedText")
            }
        }
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
        if (!isRecognitionActive) return // Проверка флага

        scope.launch {
            imageAnalysis?.clearAnalyzer()
            ortSession = createOrtSession()
            if (ortSession != null) {
                imageAnalysis?.setAnalyzer(
                    backgroundExecutor,
                    ORTAnalyzer(this@MainActivity,ortSession!!, ::updateUI, ortEnv!!)
                )
            } else {
                Log.e(TAG, "Failed to create OrtSession")
            }
        }
    }


    companion object {
        const val TAG = "Jestuno"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private fun startDevModeCountdown() {
        devModeHandler.postDelayed(object : Runnable {
            private var secondsPassed = 0

            override fun run() {
                val timePassed = (System.currentTimeMillis() - pressStartTime).toInt()

                when {
                    timePassed >= 5000 -> {
                        showDevModeDialog()
                        hintToast?.cancel()
                    }
                    timePassed >= 2000 -> {
                        val remaining = (5000 - timePassed) / 1000
                        showHint("Dev Mode: $remaining seconds remaining")
                        devModeHandler.postDelayed(this, 1000)
                    }
                    else -> {
                        secondsPassed = timePassed / 1000
                        devModeHandler.postDelayed(this, 1000)
                    }
                }
            }
        }, 2000)
    }

    private fun showHint(message: String) {
        hintToast?.cancel()
        hintToast = Toast.makeText(this, message, Toast.LENGTH_SHORT).also { it.show() }
    }

    private fun showDevModeDialog() {
        val dialog = Dialog(this).apply {
            requestWindowFeature(Window.FEATURE_NO_TITLE)
            setContentView(R.layout.dev_mode_dialog)
            window?.setLayout(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.WRAP_CONTENT
            )
            setCanceledOnTouchOutside(true) // Закрывать при нажатии за пределами
        }

        // Находим корневой layout диалога
        val rootLayout = dialog.findViewById<View>(R.id.rootLayout)
        rootLayout.setOnClickListener {
            // Ничего не делаем, чтобы клики внутри диалога не закрывали его
        }

        val thresholdInput = dialog.findViewById<EditText>(R.id.thresholdInput)
        val showAllSwitch = dialog.findViewById<SwitchMaterial>(R.id.showAllSwitch)

        // Устанавливаем текущее значение порога
        thresholdInput.setText("%.1f".format(devModeThreshold * 100))

        // Обработчик ввода порога
        thresholdInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                val inputValue = thresholdInput.text.toString().toFloatOrNull()
                if (inputValue != null && inputValue in 0f..100f) {
                    devModeThreshold = inputValue / 100f
                    Toast.makeText(this, "Threshold set to $inputValue%", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Invalid value! Enter a number between 0 and 100.", Toast.LENGTH_SHORT).show()
                }
                true
            } else {
                false
            }
        }

        // Обработчик для переключателя
        showAllSwitch.isChecked = showAllLetters
        showAllSwitch.setOnCheckedChangeListener { _, isChecked ->
            showAllLetters = isChecked
        }

        // Обработчик закрытия диалога
        dialog.setOnCancelListener {
        }

        dialog.show()
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

class GalleryActivity : AppCompatActivity() {
    private lateinit var images: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gallery)

        // Показываем кнопку "Назад" в ActionBar
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        val assetManager = assets
        images = assetManager.list("bb")
            ?.sortedWith(compareBy { it.filter { char -> char.isDigit() }.toIntOrNull() ?: 0 })
            ?: emptyList()


        val gridView: GridView = findViewById(R.id.gridView)
        gridView.adapter = ImageAdapter(this, images)

        gridView.setOnItemClickListener { _, _, position, _ ->
            val intent = Intent(this, FullScreenImageActivity::class.java).apply {
                putExtra("images", images.toTypedArray())
                putExtra("position", position)
            }
            startActivity(intent)
        }

        val btnBack: FloatingActionButton = findViewById(R.id.btnBack)
        btnBack.setOnClickListener {
            finish()  // Закрыть GalleryActivity и вернуться назад
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {  // Нажатие на кнопку "Назад"
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}


class ImageAdapter(private val context: Context, private val images: List<String>) : BaseAdapter() {
    override fun getCount(): Int = images.size
    override fun getItem(position: Int): Any = images[position]
    override fun getItemId(position: Int): Long = position.toLong()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {
        val imageView = convertView as? ImageView ?: ImageView(context).apply {
            layoutParams = ViewGroup.LayoutParams(400, 400)
            scaleType = ImageView.ScaleType.CENTER_CROP
            setPadding(8, 8, 8, 8)
        }

        val assetManager = context.assets
        val inputStream = assetManager.open("bb/${images[position]}")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        imageView.setImageBitmap(bitmap)

        return imageView
    }
}


class FullScreenImageActivity : AppCompatActivity() {
    private lateinit var viewPager: ViewPager2
    private lateinit var images: Array<String>
    private var startPosition = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_full_screen)

        // Получаем данные из Intent
        images = intent.getStringArrayExtra("images") ?: emptyArray()
        startPosition = intent.getIntExtra("position", 0)

        // Настройка ViewPager
        viewPager = findViewById(R.id.viewPager)
        viewPager.adapter = ImagePagerAdapter(this, images)
        viewPager.setCurrentItem(startPosition, false)

        // Добавляем кнопку "Назад" в ActionBar
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            finish()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    private class ImagePagerAdapter(
        private val context: Context,
        private val images: Array<String>
    ) : RecyclerView.Adapter<ImagePagerAdapter.ViewHolder>() {

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
            val imageView = ImageView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                scaleType = ImageView.ScaleType.FIT_CENTER
            }
            return ViewHolder(imageView)
        }

        override fun onBindViewHolder(holder: ViewHolder, position: Int) {
            try {
                val assetManager = context.assets
                val inputStream = assetManager.open("bb/${images[position]}")
                val bitmap = BitmapFactory.decodeStream(inputStream)
                holder.imageView.setImageBitmap(bitmap)
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }

        override fun getItemCount(): Int = images.size

        inner class ViewHolder(val imageView: ImageView) : RecyclerView.ViewHolder(imageView)
    }
}




