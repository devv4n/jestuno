package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.example.imageclassifier.databinding.ActivityMainMenuBinding
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainMenuActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainMenuBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainMenuBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnMainMode.setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }

        binding.btnStudyMode.setOnClickListener {
            startActivity(Intent(this, GalleryActivity::class.java))
        }

        binding.btnTrainingMode.setOnClickListener {
            startActivity(Intent(this, TrainingActivity::class.java))
        }
    }
}