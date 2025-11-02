package com.example.retinal_screening

import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.retinal.screening/model"
    private var module: Module? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initModel" -> {
                    try {
                        initModel()
                        result.success(true)
                    } catch (e: Exception) {
                        result.error("INIT_ERROR", e.message, null)
                    }
                }
                "runInference" -> {
                    try {
                        val imageData = call.argument<ByteArray>("imageData")
                        if (imageData == null) {
                            result.error("INVALID_ARGUMENT", "Image data is null", null)
                            return@setMethodCallHandler
                        }
                        val predictions = runInference(imageData)
                        result.success(mapOf("predictions" to predictions))
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun initModel() {
        if (module != null) return

        // Copy model from assets to cache directory
        val modelPath = "flutter_assets/assets/models/best_model_mobile.pth"
        val assetManager = assets
        val cacheFile = File(cacheDir, "best_model_mobile.pth")

        if (!cacheFile.exists()) {
            assetManager.open(modelPath).use { input ->
                FileOutputStream(cacheFile).use { output ->
                    input.copyTo(output)
                }
            }
        }

        // Load PyTorch model
        module = Module.load(cacheFile.absolutePath)
    }

    private fun runInference(imageData: ByteArray): FloatArray {
        if (module == null) {
            throw IllegalStateException("Model not initialized")
        }

        // Convert ByteArray to FloatArray
        val buffer = ByteBuffer.wrap(imageData).order(ByteOrder.nativeOrder())
        val floatArray = FloatArray(imageData.size / 4)
        buffer.asFloatBuffer().get(floatArray)

        // Create input tensor [1, 3, 224, 224]
        val inputTensor = Tensor.fromBlob(floatArray, longArrayOf(1, 3, 224, 224))

        // Run inference
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()

        // Get output as float array
        val scores = outputTensor.dataAsFloatArray

        return scores
    }

    override fun onDestroy() {
        module = null
        super.onDestroy()
    }
}

