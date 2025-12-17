import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as image_lib;
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

  static const int _inputSize = 224;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  Future<void> _initialize() async {
    await _loadModel();
    await _startCamera();
  }

  Future<void> _loadModel() async {
    try {
      _labels = await _loadLabels();
      _interpreter = await Interpreter.fromAsset(
        'assets/models/model.tflite',
        options: InterpreterOptions()..threads = 2,
      );

      setState(() {
        _modelLoaded = true;
        _status = 'Model loaded. Starting camera...';
      });
    } catch (e) {
      setState(() {
        _modelLoaded = false;
        _status = 'Model load failed: $e';
      });
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
        ResolutionPreset.medium,
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
                : Center(child: Text(_status)),
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
    final rgbImage = _convertYUV420ToImage(image);
    final resized =
        image_lib.copyResize(rgbImage, width: _inputSize, height: _inputSize);

    final input = Float32List(_inputSize * _inputSize * 3);
    int idx = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[idx++] = (image_lib.getRed(pixel) / 255.0);
        input[idx++] = (image_lib.getGreen(pixel) / 255.0);
        input[idx++] = (image_lib.getBlue(pixel) / 255.0);
      }
    }

    final inputTensor = input.reshape([1, _inputSize, _inputSize, 3]);

    final outputShape = _interpreter!.getOutputTensor(0).shape;
    final outputLen = outputShape.reduce((a, b) => a * b);
    final outputBuffer = Float32List(outputLen);

    _interpreter!.run(inputTensor, outputBuffer);

    final results = <Map<String, dynamic>>[];
    final maxResults = min(3, outputLen);
    for (int i = 0; i < outputLen; i++) {
      final confidence = outputBuffer[i];
      results.add({
        'index': i,
        'label': i < _labels.length ? _labels[i] : 'Label $i',
        'confidence': confidence,
      });
    }
    results.sort((a, b) => (b['confidence'] as double)
        .compareTo(a['confidence'] as double));
    return results.take(maxResults).toList();
  }

  image_lib.Image _convertYUV420ToImage(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    final imageBuffer = image_lib.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final uvRow = uvRowStride * (y >> 1);
      for (int x = 0; x < width; x++) {
        final uvIndex = uvRow + (x >> 1) * uvPixelStride;
        final yValue = image.planes[0].bytes[y * image.planes[0].bytesPerRow + x];
        final uValue = image.planes[1].bytes[uvIndex];
        final vValue = image.planes[2].bytes[uvIndex];

        final r = (yValue + (1.370705 * (vValue - 128))).clamp(0, 255).toInt();
        final g = (yValue -
                (0.337633 * (uValue - 128)) -
                (0.698001 * (vValue - 128)))
            .clamp(0, 255)
            .toInt();
        final b =
            (yValue + (1.732446 * (uValue - 128))).clamp(0, 255).toInt();

        imageBuffer.setPixelRgb(x, y, r, g, b);
      }
    }

    return imageBuffer;
  }
}

