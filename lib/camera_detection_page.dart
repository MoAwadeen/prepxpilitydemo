import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class CameraDetectionPage extends StatefulWidget {
  const CameraDetectionPage({super.key});

  @override
  State<CameraDetectionPage> createState() => _CameraDetectionPageState();
}

class _CameraDetectionPageState extends State<CameraDetectionPage>
    with WidgetsBindingObserver {
  CameraController? _controller;
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isProcessingFrame = false;
  bool _modelLoaded = false;
  String _status = 'Loading model...';
  List<dynamic> _results = [];

  // Will be read from model's input tensor
  // YOLO11n expects 640x640 input size
  int _inputHeight = 640;
  int _inputWidth = 640;
  int _inputChannels = 3;

  // Throttle inference: skip frames to reduce CPU load
  int _frameSkipCount = 0;
  static const int _frameSkipInterval = 10; // Process every 10th frame (increased for performance)
  DateTime _lastInferenceTime = DateTime.now();

  // #region agent log
  void _writeLog(String location, String message, Map<String, dynamic> data, String hypothesisId) {
    try {
      // Only log on debug builds to avoid file system issues
      if (kDebugMode) {
        debugPrint('[$location] $message: $data');
      }
    } catch (e) {
      // Silently fail if logging doesn't work
    }
  }
  // #endregion

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Show UI immediately, then initialize in background
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initialize();
    });
  }

  Future<void> _initialize() async {
    debugPrint('=== Starting initialization ===');
    
    // Load model with timeout
    try {
      await _loadModel().timeout(
        const Duration(seconds: 30),
        onTimeout: () {
          debugPrint('Model loading timed out after 30 seconds');
          throw TimeoutException('Model loading timed out');
        },
      );
      debugPrint('Model loaded successfully');
    } catch (e, stackTrace) {
      debugPrint('Model loading error: $e');
      debugPrint('Stack trace: $stackTrace');
      if (mounted) {
        setState(() {
          _status = 'Model error: ${e.toString().substring(0, e.toString().length > 100 ? 100 : e.toString().length)}';
        });
      }
      // Continue to start camera even if model fails
    }
    
    // Start camera with timeout
    try {
      debugPrint('Starting camera...');
      await _startCamera().timeout(
        const Duration(seconds: 10),
        onTimeout: () {
          debugPrint('Camera initialization timed out after 10 seconds');
          throw TimeoutException('Camera initialization timed out');
        },
      );
      debugPrint('Camera started successfully');
    } catch (e, stackTrace) {
      debugPrint('Camera initialization error: $e');
      debugPrint('Stack trace: $stackTrace');
      if (mounted) {
        setState(() {
          _status = 'Camera error: ${e.toString()}';
        });
      }
    }
    
    debugPrint('=== Initialization complete ===');
  }

  Future<void> _loadModel() async {
    try {
      debugPrint('Starting model loading...');
      if (mounted) {
        setState(() {
          _status = 'Loading labels...';
        });
      }
      
      _labels = await _loadLabels();
      debugPrint('Labels loaded: ${_labels.length} labels');
      
      if (mounted) {
        setState(() {
          _status = 'Loading TFLite model...';
        });
      }
      
      debugPrint('Loading model from: assets/models/best_float16.tflite');
      _interpreter = await Interpreter.fromAsset(
        'assets/models/best_float16.tflite',
        options: InterpreterOptions()..threads = 2,
      );
      debugPrint('Model loaded successfully');

      // Read input shape from model: typically [1, height, width, channels]
      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape;
      
      // #region agent log
      _writeLog('camera_detection_page.dart:83', 'Model input shape before resize', {'originalShape': inputShape.toString(), 'targetSize': 320}, 'A');
      // #endregion
      
      debugPrint('TFLite input tensor shape: $inputShape');
      debugPrint('TFLite input tensor type: ${inputTensor.type}');
      
      // Use the model's original input shape if it's valid, otherwise try common sizes
      if (inputShape.length == 4) {
        // Check if shape has dynamic dimensions (-1 or 0) or fixed dimensions
        final h = inputShape[1];
        final w = inputShape[2];
        final c = inputShape[3] > 0 ? inputShape[3] : 3;
        
        debugPrint('Model input dimensions: h=$h, w=$w, c=$c');
        
        // YOLO11n requires 640x640 - always resize to this size
        _inputHeight = 640;
        _inputWidth = 640;
        _inputChannels = 3;
        
        final targetShape = [1, _inputHeight, _inputWidth, _inputChannels];
        
        debugPrint('YOLO11n: Resizing input tensor to 640x640');
        debugPrint('Original model shape: $inputShape');
        
        // #region agent log
        _writeLog('camera_detection_page.dart:102', 'Resizing input tensor for YOLO11n', {
          'targetShape': targetShape.toString(),
          'originalShape': inputShape.toString(),
          'modelType': 'YOLO11n',
          'requiredSize': '640x640'
        }, 'A');
        // #endregion
        
        try {
          _interpreter!.resizeInputTensor(0, targetShape);
          _interpreter!.allocateTensors();
          
          // #region agent log
          final verifyShape = _interpreter!.getInputTensor(0).shape;
          debugPrint('Verified input tensor shape after resize: $verifyShape');
          _writeLog('camera_detection_page.dart:115', 'Input tensor shape after resize', {
            'verifiedShape': verifyShape.toString(),
            'expected': targetShape.toString(),
            'shapesMatch': verifyShape.toString() == targetShape.toString(),
            'success': true
          }, 'A');
          // #endregion
        } catch (e) {
          debugPrint('Tensor resize failed: $e');
          // #region agent log
          _writeLog('camera_detection_page.dart:125', 'Tensor resize failed', {
            'error': e.toString(),
            'errorType': e.runtimeType.toString(),
            'targetShape': targetShape.toString()
          }, 'A');
          // #endregion
          rethrow;
        }
      } else {
        // Fallback to 112x112
        _inputHeight = 112;
        _inputWidth = 112;
        _inputChannels = 3;
        // #region agent log
        _writeLog('camera_detection_page.dart:138', 'Unexpected input shape, using fallback', {
          'shapeLength': inputShape.length,
          'shape': inputShape.toString(),
          'fallbackSize': '112x112'
        }, 'A');
        // #endregion
      }

      final outputTensor = _interpreter!.getOutputTensor(0);
      debugPrint('TFLite output tensor shape: ${outputTensor.shape}');
      
      // Check for multiple output tensors (some YOLO models have separate outputs)
      // Try to access output tensors, catching errors if they don't exist
      try {
        final tensor0 = _interpreter!.getOutputTensor(0);
        debugPrint('Output tensor 0 shape: ${tensor0.shape}, type: ${tensor0.type}');
        
        // Try to get additional output tensors if they exist
        try {
          final tensor1 = _interpreter!.getOutputTensor(1);
          debugPrint('Output tensor 1 shape: ${tensor1.shape}, type: ${tensor1.type}');
        } catch (e) {
          debugPrint('Only one output tensor found');
        }
      } catch (e) {
        debugPrint('Error accessing output tensor: $e');
      }

      debugPrint('Model initialization complete');
      if (mounted) {
        setState(() {
          _modelLoaded = true;
          _status = 'Model loaded (${_inputWidth}x$_inputHeight). Starting camera...';
        });
      }
    } catch (e, stackTrace) {
      debugPrint('Model load error: $e');
      debugPrint('Stack trace: $stackTrace');
      if (mounted) {
        setState(() {
          _modelLoaded = false;
          _status = 'Model load failed: ${e.toString()}';
        });
      }
      rethrow; // Re-throw so _initialize can catch it
    }
  }

  Future<List<String>> _loadLabels() async {
    try {
      final raw = await rootBundle.loadString('assets/models/labels.txt');
      return raw
          .split('\n')
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty && !line.startsWith('#'))
          .toList();
    } catch (_) {
      return [];
    }
  }

  Future<void> _startCamera() async {
    try {
      await _controller?.dispose();

      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _status = 'No camera found on this device.';
        });
        return;
      }

      final selectedCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        selectedCamera,
        ResolutionPreset.medium, // Balanced quality and performance
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await controller.initialize();
      await controller.startImageStream(_onCameraImage);

      if (!mounted) {
        return;
      }

      setState(() {
        _controller = controller;
        _status =
            _modelLoaded ? 'Running detection...' : 'Camera ready (model missing)';
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _status = 'Camera error: $e';
      });
    }
  }

  void _onCameraImage(CameraImage image) async {
    if (!_modelLoaded || _interpreter == null || _isProcessingFrame) {
      return;
    }

    // Throttle: skip frames to reduce CPU usage
    _frameSkipCount++;
    if (_frameSkipCount < _frameSkipInterval) {
      return;
    }
    _frameSkipCount = 0;

    // Also throttle by time: at least 500ms between inferences (increased for performance)
    final now = DateTime.now();
    if (now.difference(_lastInferenceTime).inMilliseconds < 500) {
      return;
    }
    _lastInferenceTime = now;

    _isProcessingFrame = true;

    try {
      final results = await _runInference(image);

      if (!mounted) {
        return;
      }

      setState(() {
        _results = results;
        _status = results.isEmpty ? 'No confident results' : 'Running detection...';
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _status = 'Detection error: $e';
        });
      }
    } finally {
      _isProcessingFrame = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      controller.stopImageStream();
      controller.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _startCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera + TFLite'),
      ),
      body: Column(
        children: [
          Expanded(
            child: _controller?.value.isInitialized == true
                ? CameraPreview(_controller!)
                : Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 16),
                        Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Text(
                            _status,
                            textAlign: TextAlign.center,
                            style: const TextStyle(fontSize: 16),
                          ),
                        ),
                      ],
                    ),
                  ),
          ),
          Container(
            width: double.infinity,
            color: Colors.black87,
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Status: $_status',
                  style: const TextStyle(color: Colors.white, fontSize: 14),
                ),
                const SizedBox(height: 8),
                if (_results.isEmpty)
                  const Text(
                    'No results yet',
                    style: TextStyle(color: Colors.white70),
                  )
                else
                  ..._results.map(
                    (result) => Text(
                      _formatResult(result),
                      style: const TextStyle(color: Colors.white),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatResult(dynamic result) {
    if (result is Map) {
      final label = result['label'] ?? 'Unknown';
      final confidence = result['confidence'];
      if (confidence is num) {
        return '$label — ${(confidence * 100).toStringAsFixed(1)}%';
      }
      return '$label';
    }
    return result.toString();
  }

  Future<List<Map<String, dynamic>>> _runInference(CameraImage image) async {
    // #region agent log
    _writeLog('camera_detection_page.dart:262', 'Starting inference', {'cameraWidth': image.width, 'cameraHeight': image.height, 'targetSize': '${_inputWidth}x$_inputHeight'}, 'A');
    // #endregion
    
    final rgbBytes = _convertYUV420ToRGB(image);
    final inputFlat = _resizeAndNormalize(rgbBytes, image.width, image.height);

    // #region agent log
    _writeLog('camera_detection_page.dart:266', 'Input prepared', {'inputFlatLength': inputFlat.length, 'expectedLength': _inputHeight * _inputWidth * _inputChannels}, 'A');
    // #endregion

    // Reshape to [1, height, width, channels] based on model's input tensor
    final input = inputFlat.reshape([1, _inputHeight, _inputWidth, _inputChannels]);

    // Verify input tensor shape matches what we're providing
    final actualInputTensor = _interpreter!.getInputTensor(0);
    final actualInputShape = actualInputTensor.shape;
    
    // #region agent log
    _writeLog('camera_detection_page.dart:355', 'Input tensor verification', {
      'actualInputShape': actualInputShape.toString(),
      'providedInputShape': '[1, $_inputHeight, $_inputWidth, $_inputChannels]',
      'inputFlatLength': inputFlat.length,
      'reshapedInputElements': _inputHeight * _inputWidth * _inputChannels,
      'shapesMatch': actualInputShape.toString() == '[1, $_inputHeight, $_inputWidth, $_inputChannels]'
    }, 'A');
    // #endregion

    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final outputLen = outputShape.reduce((a, b) => a * b);
    final output = List.filled(outputLen, 0.0).reshape(outputShape);

    // #region agent log
    _writeLog('camera_detection_page.dart:366', 'Running inference', {
      'inputShape': '[1, $_inputHeight, $_inputWidth, $_inputChannels]',
      'actualInputTensorShape': actualInputShape.toString(),
      'outputShape': outputShape.toString(),
      'outputLength': outputLen
    }, 'A');
    // #endregion

    try {
      _interpreter!.run(input, output);
      
      // #region agent log
      _writeLog('camera_detection_page.dart:280', 'Inference completed', {'outputLength': outputLen, 'success': true}, 'A');
      // #endregion
    } catch (e) {
      // #region agent log
      _writeLog('camera_detection_page.dart:284', 'Inference failed', {'error': e.toString(), 'inputShape': '[1, $_inputHeight, $_inputWidth, $_inputChannels]'}, 'A');
      // #endregion
      rethrow;
    }

    // YOLO11n output format: Can be [1, 5, 8400] or [1, 7, 8400]
    // Detect actual output shape at runtime
    // Layout in memory (row-major): [all x's][all y's][all w's][all h's][all conf's][optional: all class1's][optional: all class2's]
    
    // Flatten output for easier access
    final outputBuffer = _flattenOutput(output);
    
    final results = <Map<String, dynamic>>[];
    
    // Determine output format: [1, values, detections] or [1, detections, values]
    int numDetections;
    int numValues;
    bool isTransposed = false;
    
    if (outputShape.length == 3) {
      // Check which dimension is larger (likely the number of detections)
      if (outputShape[1] > outputShape[2]) {
        // Format: [1, detections, values] - transposed
        numDetections = outputShape[1];
        numValues = outputShape[2];
        isTransposed = true;
        debugPrint('Detected transposed format: [1, $numDetections, $numValues]');
      } else {
        // Format: [1, values, detections] - standard
        numDetections = outputShape[2];
        numValues = outputShape[1];
        debugPrint('Detected standard format: [1, $numValues, $numDetections]');
      }
    } else {
      // Fallback
      numDetections = outputShape.length > 2 ? outputShape[2] : 8400;
      numValues = outputShape.length > 1 ? outputShape[1] : 5;
    }
    
    // Log actual output shape for debugging
    debugPrint('Actual output shape: $outputShape, buffer length: ${outputBuffer.length}');
    debugPrint('Number of detections: $numDetections, values per detection: $numValues');
    
    // Debug: Check some sample confidence values
    int highConfCount = 0;
    int mediumConfCount = 0;
    double maxConf = 0.0;
    double minConf = 1.0;
    
    // Parse YOLO detections from flattened buffer
    // Check all detections (not sampling) to find any detections
    for (int i = 0; i < numDetections; i++) {
      double x, y, w, h, finalConf;
      int classId = 0;
      String classLabel = 'Object';
      
      if (isTransposed) {
        // Transposed format: [1, detections, values]
        // Layout: [detection0: x,y,w,h,conf...], [detection1: x,y,w,h,conf...], ...
        final baseIndex = i * numValues;
        if (baseIndex + 4 >= outputBuffer.length) break;
        
        x = outputBuffer[baseIndex];
        y = outputBuffer[baseIndex + 1];
        w = outputBuffer[baseIndex + 2];
        h = outputBuffer[baseIndex + 3];
        
        if (numValues >= 5) {
          finalConf = outputBuffer[baseIndex + 4];
          if (numValues >= 8) {
            // Has class scores: [x, y, w, h, objectness, class_0, class_1, class_2]
            final objectness = outputBuffer[baseIndex + 4];
            final class0 = outputBuffer[baseIndex + 5];
            final class1 = outputBuffer[baseIndex + 6];
            final class2 = outputBuffer[baseIndex + 7];
            final maxClass = max(max(class0, class1), class2);
            finalConf = objectness * maxClass;
            
            if (class0 > class1 && class0 > class2) classId = 0;
            else if (class1 > class2) classId = 1;
            else classId = 2;
            classLabel = classId < _labels.length ? _labels[classId] : 'Class $classId';
          }
        } else {
          finalConf = 0.0;
        }
      } else {
        // Standard format: [1, values, detections]
        // Layout: [all x's][all y's][all w's][all h's][all conf's]...
        if (4 * numDetections + i >= outputBuffer.length) break;
        
        x = outputBuffer[i]; // x for detection i
        y = outputBuffer[numDetections + i]; // y for detection i
        w = outputBuffer[2 * numDetections + i]; // w for detection i
        h = outputBuffer[3 * numDetections + i]; // h for detection i
        
        if (numValues >= 7) {
          // 7-channel format for YOLO11 with 3 classes: [x, y, w, h, class_0, class_1, class_2]
          // Apply sigmoid to convert logits to probabilities if needed
          if (6 * numDetections + i < outputBuffer.length) {
            double class0 = outputBuffer[4 * numDetections + i]; // class 0 (Racket)
            double class1 = outputBuffer[5 * numDetections + i]; // class 1 (Tennis-Ball)
            double class2 = outputBuffer[6 * numDetections + i]; // class 2 (person)
            
            // Apply sigmoid if values are outside [0,1] range (they're logits)
            // Sigmoid: 1 / (1 + exp(-x))
            if (class0 < 0 || class0 > 1 || class1 < 0 || class1 > 1 || class2 < 0 || class2 > 1) {
              class0 = 1.0 / (1.0 + exp(-class0));
              class1 = 1.0 / (1.0 + exp(-class1));
              class2 = 1.0 / (1.0 + exp(-class2));
            }
            
            // Calculate final confidence: max of all class probabilities
            final maxClassScore = max(max(class0, class1), class2);
            finalConf = maxClassScore; // Use max class score as confidence
            
            // Determine class label (0-indexed)
            if (class0 > class1 && class0 > class2) {
              classId = 0;
            } else if (class1 > class2) {
              classId = 1;
            } else {
              classId = 2;
            }
            classLabel = classId < _labels.length ? _labels[classId] : 'Class $classId';
          } else {
            // Fallback if indices are out of bounds
            finalConf = outputBuffer[4 * numDetections + i]; // Use 5th value as confidence
          }
        } else {
          // 5-channel format: [x, y, w, h, confidence]
          finalConf = outputBuffer[4 * numDetections + i]; // confidence directly
          classLabel = 'Object'; // Generic label since class info not available in 5-channel format
        }
      }
      
      // Skip invalid coordinates
      if (x.isNaN || y.isNaN || w.isNaN || h.isNaN || 
          x.isInfinite || y.isInfinite || w.isInfinite || h.isInfinite) {
        continue;
      }
      
      // Track confidence statistics
      if (!finalConf.isNaN && !finalConf.isInfinite) {
        if (finalConf > maxConf) maxConf = finalConf;
        if (finalConf < minConf) minConf = finalConf;
        if (finalConf > 0.5) highConfCount++;
        if (finalConf > 0.25) mediumConfCount++;
      }
      
      // Lower threshold significantly for testing - YOLO11 outputs may need sigmoid/softmax
      // Also check that width and height are reasonable (not too small or too large)
      // Try very low threshold first to see if we get any detections
      if (finalConf > 0.001 && w > 0.01 && h > 0.01 && w < 1.0 && h < 1.0) {
        results.add({
          'x': x,
          'y': y,
          'width': w,
          'height': h,
          'confidence': finalConf,
          'label': classLabel,
          'classId': classId,
        });
      }
      
      // Limit results to prevent too many detections
      if (results.length >= 20) {
        break;
      }
    }
    
    // Log confidence statistics
    debugPrint('Confidence stats: max=$maxConf, min=$minConf, high=$highConfCount, medium=$mediumConfCount');
    debugPrint('Found ${results.length} detections above threshold');
    
    // Sort by confidence and return top 3 detections (reduced for performance)
    results.sort((a, b) => (b['confidence'] as double)
        .compareTo(a['confidence'] as double));
    
    return results.take(min(3, results.length)).toList();
  }

  Uint8List _convertYUV420ToRGB(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
    final rgbBuffer = Uint8List(width * height * 3);

    int bufferIndex = 0;
    for (int y = 0; y < height; y++) {
      final uvRow = uvRowStride * (y >> 1);
      final yRow = image.planes[0].bytesPerRow * y;
      for (int x = 0; x < width; x++) {
        final uvIndex = uvRow + (x >> 1) * uvPixelStride;
        final yValue = image.planes[0].bytes[yRow + x];
        final uValue = image.planes[1].bytes[uvIndex];
        final vValue = image.planes[2].bytes[uvIndex];

        final r =
            (yValue + (1.370705 * (vValue - 128))).clamp(0, 255).toInt();
        final g = (yValue -
                (0.337633 * (uValue - 128)) -
                (0.698001 * (vValue - 128)))
            .clamp(0, 255)
            .toInt();
        final b =
            (yValue + (1.732446 * (uValue - 128))).clamp(0, 255).toInt();

        rgbBuffer[bufferIndex++] = r;
        rgbBuffer[bufferIndex++] = g;
        rgbBuffer[bufferIndex++] = b;
      }
    }

    return rgbBuffer;
  }

  Float32List _resizeAndNormalize(
      Uint8List rgbBytes, int srcWidth, int srcHeight) {
    final output = Float32List(_inputHeight * _inputWidth * _inputChannels);
    final xScale = srcWidth / _inputWidth;
    final yScale = srcHeight / _inputHeight;

    int outIndex = 0;
    for (int y = 0; y < _inputHeight; y++) {
      final sy = min((y * yScale).floor(), srcHeight - 1);
      for (int x = 0; x < _inputWidth; x++) {
        final sx = min((x * xScale).floor(), srcWidth - 1);
        final srcIndex = (sy * srcWidth + sx) * 3;
        output[outIndex++] = rgbBytes[srcIndex] / 255.0;
        output[outIndex++] = rgbBytes[srcIndex + 1] / 255.0;
        output[outIndex++] = rgbBytes[srcIndex + 2] / 255.0;
      }
    }

    return output;
  }

  List<double> _flattenOutput(dynamic output) {
    final result = <double>[];
    _flattenRecursive(output, result);
    return result;
  }

  void _flattenRecursive(dynamic data, List<double> result) {
    if (data is List) {
      for (final item in data) {
        _flattenRecursive(item, result);
      }
    } else if (data is num) {
      result.add(data.toDouble());
    }
  }
}


