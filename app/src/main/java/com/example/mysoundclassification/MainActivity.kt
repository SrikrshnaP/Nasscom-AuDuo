package com.example.myaudioclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class AudioClassifierActivity : AppCompatActivity() {
    private val TAG = "AudioClassifierActivity"
    private val modelPath = "skAudio/AudModel/audio_classifier_model.tflite"
    private val probabilityThreshold: Float = 0.3f
    private lateinit var textView: TextView

    private var classificationTimer: Timer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_audio_classifier)

        val RECORD_AUDIO_PERMISSION_CODE = 1337
        val permission = Manifest.permission.RECORD_AUDIO

        // Check and request audio recording permission
        if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(permission), RECORD_AUDIO_PERMISSION_CODE)
            // You can also show a message to the user explaining the need for the permission.
        }

        textView = findViewById<TextView>(R.id.output)
        val recorderSpecsTextView = findViewById<TextView>(R.id.textViewRecorderSpecs)

        // Load the audio classification model with error handling
        try {
            val audioClassifier = AudioClassifier.createFromFile(this, modelPath)

            // Create input tensor for audio data
            val inputTensor = audioClassifier.createInputTensorAudio()

            // Get audio format specifications
            val audioFormat = audioClassifier.requiredTensorAudioFormat
            val recorderSpecs = "Channels: ${audioFormat.channels}\n" +
                    "Sample Rate: ${audioFormat.sampleRate}"
            recorderSpecsTextView.text = recorderSpecs

            // Create and start the audio recorder with error handling
            val audioRecorder = audioClassifier.createAudioRecord()
            if (audioRecorder != null && audioRecorder.state == AudioRecord.STATE_INITIALIZED) {
                audioRecorder.startRecording()

                // Start classification timer
                classificationTimer = Timer()
                classificationTimer?.scheduleAtFixedRate(1, 500) {
                    // Load audio data and classify
                    val numberOfSamples = inputTensor.load(audioRecorder)
                    val classificationOutput = audioClassifier.classify(inputTensor)

                    // Filter out low probability classifications
                    val filteredOutput = classificationOutput[0].categories.filter {
                        it.score > probabilityThreshold
                    }

                    // Create a formatted string with the filtered results
                    val outputString =
                        filteredOutput.sortedByDescending { it.score }
                            .joinToString(separator = "\n") { "${it.label} -> ${it.score}" }

                    // Update the UI
                    if (outputString.isNotEmpty()) {
                        runOnUiThread {
                            textView.text = outputString
                        }
                    }
                }
            } else {
                Log.e(TAG, "Error initializing the audio recorder.")
            }
        } catch (e: Exception) {
            // Handle model loading error, e.g., show an error message.
            Log.e(TAG, "Error loading the audio classification model: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        // Cancel the classification timer to prevent memory leaks
        classificationTimer?.cancel()
    }
}
