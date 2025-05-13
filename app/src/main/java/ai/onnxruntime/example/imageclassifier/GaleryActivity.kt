package ai.onnxruntime.example.imageclassifier

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.GridView
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.RecyclerView
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.floatingactionbutton.FloatingActionButton
import java.io.IOException


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