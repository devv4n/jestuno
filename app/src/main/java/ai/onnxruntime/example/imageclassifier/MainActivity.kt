package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainBinding
import android.Manifest
import android.app.Dialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// Top-level extension function for rotating a Bitmap
fun Bitmap.rotate(degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}

lateinit var labelData: List<String>

fun initializeLabels(context: Context) {
    labelData = context.resources.openRawResource(R.raw.imagenet_classes).bufferedReader().readLines()
}

var devModeThreshold = 0.055f
var showAllLetters = false

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

