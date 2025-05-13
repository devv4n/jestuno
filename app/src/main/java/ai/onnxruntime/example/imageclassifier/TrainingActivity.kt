package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.os.*
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import ai.onnxruntime.example.imageclassifier.databinding.ActivityTrainingBinding
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.*
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.Manifest
import android.content.Intent


class TrainingActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private lateinit var binding: ActivityTrainingBinding
    private lateinit var cameraExecutor: ExecutorService
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private lateinit var tts: TextToSpeech
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

    // Тренировочные переменные
    private var wordsList = mutableListOf<String>()
    private var currentWord = ""
    private var currentLetterIndex = 0
    private var correctGestures = 0
    private var bestScore = 0
    private var isTrainingActive = false
    private var isSpeaking: Boolean = false
    private var startTime: Long = 0
    private val handler = Handler(Looper.getMainLooper())
    private val timerRunnable = object : Runnable {
        override fun run() {
            updateTimer()
            handler.postDelayed(this, 1000)
        }
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initializeLabels(this)
        binding = ActivityTrainingBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Инициализация

        loadWordsFromFile()
        ortEnv = OrtEnvironment.getEnvironment()
        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        loadBestScore()

        // Настройка UI
        setupUI()

        // Проверка разрешений и запуск камеры
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun setupUI() {
        binding.score.text = "Счет: 0"
        binding.bestScore.text = "Лучший: $bestScore"
        binding.word.text = ""
        binding.timer.text = "00:00"

        binding.startButton.setOnClickListener {
            if (isTrainingActive) stopTraining() else startTraining()
        }

        binding.speakButton1.setOnClickListener {
            isSpeaking=!isSpeaking
            if (isSpeaking) {
                speakCurrentLetter()

                binding.speakButton1.setImageResource(R.drawable.mute__1_)
                Toast.makeText(this, "Озвучивание включено", Toast.LENGTH_SHORT).show()
            } else {
                binding.speakButton1.setImageResource(R.drawable.volume__1_)
                Toast.makeText(this, "Озвучивание выключено", Toast.LENGTH_SHORT).show()
            }
        }

        binding.helpButton.setOnClickListener {
            val intent = Intent(this, GalleryActivity::class.java)
            startActivity(intent)
        }
        binding.button2.setOnClickListener { switchCamera() }
    }

    private fun loadWordsFromFile() {
        try {
            wordsList = assets.open("training.txt").bufferedReader().readLines()
                .map { it.trim().uppercase() }
                .filter { it.isNotEmpty() }
                .toMutableList()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading words", e)
            wordsList = mutableListOf("мяч", "ваня", "кот")
        }
    }

    private fun startTraining() {
        if (wordsList.isEmpty()) {
            Toast.makeText(this, "Нет слов для тренировки", Toast.LENGTH_SHORT).show()
            return
        }

        currentWord = wordsList.random()
        currentLetterIndex = 0
        correctGestures = 0
        startTime = SystemClock.elapsedRealtime()

        displayWordWithProgress()
        isTrainingActive = true
        binding.startButton.setImageResource(android.R.drawable.ic_media_pause)
        handler.post(timerRunnable)

        // Активация распознавания
        setORTAnalyzer()
    }

    private fun stopTraining() {
        isTrainingActive = false
        binding.startButton.setImageResource(android.R.drawable.ic_media_play)
        handler.removeCallbacks(timerRunnable)
        imageAnalysis?.clearAnalyzer()

        if (correctGestures > bestScore) {
            bestScore = correctGestures
            binding.bestScore.text = "Лучший: $bestScore"
            saveBestScore()
        }
    }

    private fun displayWordWithProgress() {
        val wordBuilder = StringBuilder()
        for (i in currentWord.indices) {
            val color = when {
                i < currentLetterIndex -> "#00FF00" // Зеленый - пройденные
                i == currentLetterIndex -> "#FF0000" // Красный - текущая
                else -> "#6C6C6C"       // Серый - будущие
            }
            wordBuilder.append("<font color='$color'>${currentWord[i]}</font>")
            if (i < currentWord.length - 1) wordBuilder.append(" ")
        }
        binding.word.text = android.text.Html.fromHtml(wordBuilder.toString(), 0)
    }

    private fun updateTimer() {
        val elapsed = (SystemClock.elapsedRealtime() - startTime) / 1000
        binding.timer.text = String.format("%02d:%02d", elapsed / 60, elapsed % 60)
    }

    private fun speakCurrentLetter() {
        if (currentLetterIndex < currentWord.length) {
            tts.speak(currentWord[currentLetterIndex].toString(), TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    // Логика распознавания из MainActivity
    private fun setORTAnalyzer() {
        CoroutineScope(Dispatchers.Main).launch {
            imageAnalysis?.clearAnalyzer()
            ortSession = createOrtSession()
            if (ortSession != null) {
                imageAnalysis?.setAnalyzer(
                    cameraExecutor,
                    ORTAnalyzer(this@TrainingActivity, ortSession!!, ::handleRecognitionResult, ortEnv!!)
                )
            }
        }
    }

    private val prefsName = "TrainingPrefs"
    private val bestScoreKey = "best_score"

    private fun loadBestScore() {
        val prefs = getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        bestScore = prefs.getInt(bestScoreKey, 0)
        binding.bestScore.text = "Лучший: $bestScore"
    }

    private fun saveBestScore() {
        val prefs = getSharedPreferences(prefsName, Context.MODE_PRIVATE)
        prefs.edit().putInt(bestScoreKey, bestScore).apply()
    }


    private fun handleRecognitionResult(result: Result) {
        Log.d(TAG, "Recognition result: ${result.detectedIndices}")

        if (result.detectedIndices.isEmpty()) {
            Log.w(TAG, "Empty detection result")
            return
        }

        if (result.detectedIndices[0] >= labelData.size) {
            Log.w(TAG, "Index out of bounds: ${result.detectedIndices[0]}")
            return
        }

        val detectedText = labelData[result.detectedIndices[0]]
        Log.d(TAG, "Detected: $detectedText")

        runOnUiThread {
            if (!isTrainingActive) {
                Log.d(TAG, "Training not active, ignoring detection")
                return@runOnUiThread
            }

            // Добавьте логирование текущего состояния
            Log.d(TAG, "Current word: $currentWord, index: $currentLetterIndex")

            if (currentLetterIndex < currentWord.length) {
                val currentLetter = currentWord[currentLetterIndex].toString()
                val isMatch = detectedText.equals(currentLetter, ignoreCase = true)

                Log.d(TAG, "Comparing: $detectedText with $currentLetter = $isMatch")

                if (isMatch) {
                    // Логика обработки правильного жеста
                    correctGestures++
                    currentLetterIndex++

                    // Обновление UI
                    binding.score.text = "Счет: $correctGestures"
                    displayWordWithProgress()

                    if (currentLetterIndex >= currentWord.length) {
                        Toast.makeText(this, "Слово завершено!", Toast.LENGTH_SHORT).show()
                        stopTraining()
                    } else if(isSpeaking){
                        speakCurrentLetter()
                    }
                }
            }
        }
    }

    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.IO) {
        try {
            resources.openRawResource(R.raw.mobilenetv2_tsm).use {
                ortEnv?.createSession(it.readBytes(), OrtSession.SessionOptions()).also {
                    Log.d(TAG, "ORT Session created: ${it != null}")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create OrtSession", e)
            null
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun switchCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        startCamera()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }


    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) {
            startCamera()
        } else {
            Toast.makeText(this, "Требуются разрешения камеры", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale("ru", "RU")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        ortSession?.close()
        ortEnv?.close()
        tts.stop()
        tts.shutdown()
        handler.removeCallbacks(timerRunnable)
    }

    companion object {
        private const val TAG = "TrainingActivity"
        private const val REQUEST_CODE_PERMISSIONS = 101
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private lateinit var labelData: List<String>

        fun initializeLabels(context: Context) {
            labelData = context.resources.openRawResource(R.raw.imagenet_classes)
                .bufferedReader().readLines()
        }
    }
}