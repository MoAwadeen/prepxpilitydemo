import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_v2/tflite_v2.dart';

class CameraDetectionPage extends StatefulWidget {
  const CameraDetectionPage({super.key});

  @override
  State<CameraDetectionPage> createState() => _CameraDetectionPageState();
}

class _CameraDetectionPageState extends State<CameraDetectionPage>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isProcessingFrame = false;
  bool _modelLoaded = false;
  String _status = 'Loading model...';
  List<dynamic> _results = [];

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
      final data = await rootBundle.load('assets/models/model.tflite');
      if (data.lengthInBytes < 1024) {
        setState(() {
          _status =
              'Placeholder model detected. Replace assets/models/model.tflite with your TFLite file.';
          _modelLoaded = false;
        });
        return;
      }

      await Tflite.close();
      await Tflite.loadModel(
        model: 'assets/models/model.tflite',
        labels: 'assets/models/labels.txt',
        numThreads: 2,
        isAsset: true,
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
    if (!_modelLoaded || _isProcessingFrame) {
      return;
    }

    _isProcessingFrame = true;

    try {
      final orientation = _controller?.description.sensorOrientation ?? 0;
      final recognitions = await Tflite.runModelOnFrame(
        bytesList: image.planes.map((plane) => plane.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        numResults: 3,
        threshold: 0.5,
        rotation: orientation,
        asynch: true,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _results = recognitions ?? [];
        _status = 'Running detection...';
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
    Tflite.close();
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
      final confidence = result['confidence'] ?? result['confidenceInClass'];
      if (confidence is num) {
        return '$label — ${(confidence * 100).toStringAsFixed(1)}%';
      }
      return '$label';
    }
    return result.toString();
  }
}

