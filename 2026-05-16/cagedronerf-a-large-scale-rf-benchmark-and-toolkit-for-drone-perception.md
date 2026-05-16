---
title: "CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2601.03302"
---

## Executive Summary
CageDroneRF (CDRF) provides the first comprehensive benchmark and toolkit for RF-based drone perception, addressing critical gaps in dataset diversity and realism. It spans 39 classes across 23 drone models with dual-environment collection (controlled Faraday cage and outdoor) and features a raw-signal augmentation pipeline that maintains label consistency through end-to-end transformations. For engineers building counter-drone systems, this benchmark enables training models that generalise beyond clean environments to handle real-world interference, low SNR, and frequency shifts.

## Why This Matters for Practitioners
If you're developing RF-based counter-drone systems for deployment at airports or critical infrastructure, models trained on today's narrow benchmarks will fail under operational conditions. CDRF directly addresses this by providing a benchmark that systematically varies SNR (from -20 dB to 30 dB), injects realistic interference (Bluetooth, Wi-Fi), and applies frequency shifts while preserving label consistency. This means you can now train and evaluate models under realistic conditions before deployment, reducing field failures. The automatic bounding-box recomputation for frequency shifts eliminates manual re-annotation when modifying signals, saving weeks of engineering effort per model iteration cycle. For edge deployments, CDRF's 20 MHz sampling rate (chosen for feasibility on resource-constrained hardware) ensures your models can operate within tight computational budgets.

## Problem Statement
Current RF drone detection research suffers from a "clean-room illusion," where models are trained and evaluated in artificial environments that don't reflect operational realities. Imagine training a self-driving car on a perfectly empty highway with no weather, traffic, or complex intersections, it would catastrophically fail when faced with rain, obstacles, or multi-lane roads. Similarly, RF models trained on narrow, clean datasets (like DroneRF's three drone models in controlled environments) achieve near-perfect scores on simplified benchmarks but degrade severely when encountering Bluetooth/Wi-Fi interference, spectrum crowding, or low SNR in real deployments.

## Proposed Approach
CDRF combines a large-scale dataset with a raw-signal augmentation pipeline that enables realistic stress testing of RF perception models. The system spans two data collection environments (controlled Faraday cage and open campus) and provides a complete processing chain from raw I/Q signals through spectrogram generation to detection annotations. Crucially, the augmentation pipeline operates at the I/Q level before time-frequency conversion, allowing programmatic control of SNR, interference, and frequency shifts while automatically recomputing bounding boxes to maintain label consistency.

```python
def augment_iq_signal(iq_data, snr, interference=None, frequency_shift=0):
    # Apply SNR control with controlled noise injection
    noisy_iq = add_noise(iq_data, target_snr=snr)
    
    # Inject interfering signals if specified
    if interference:
        noisy_iq = mix_interference(noisy_iq, interference)
    
    # Apply frequency shift, preserving signal characteristics
    shifted_iq = apply_frequency_shift(noisy_iq, frequency_shift)
    
    # Recompute spectrogram with consistent parameters
    spectrogram = stft(shifted_iq, fft_size=512, hop_length=10)
    
    # Recompute bounding boxes with wrap-around handling
    adjusted_boxes = adjust_bounding_boxes(
        original_boxes, 
        frequency_shift,
        spectrogram.shape[1]
    )
    
    return spectrogram, adjusted_boxes
```

## Key Technical Contributions
CDRF's core innovation lies in its end-to-end traceability from raw signals to detection annotations, eliminating manual re-annotation when modifying signals. The key technical contributions include:

1. **I/Q-level augmentation with label consistency**: Unlike prior datasets that distribute pre-rendered spectrograms with fixed annotations (requiring manual re-annotation after signal modifications), CDRF applies transformations to complex baseband I/Q data, then recomputes spectrograms and bounding boxes with correct wrap-around behaviour on the frequency axis. For frequency shifts, the pipeline precisely calculates the new frequency position: 500 Hz / (20 MHz / 512) = 12.8 pixels, then adjusts bounding boxes by this amount with exact wrap-around handling.

2. **SNR-structured datasets**: CDRF enables programmatic creation of SNR-stratified splits (including noise-only backgrounds) across a wide range (-20 dB to 30 dB), allowing systematic testing of model robustness under varying signal quality. This addresses the critical gap where existing datasets provide limited SNR diversity or apply noise post-hoc rather than capturing it in situ.

3. **Edge-device feasible sampling**: CDRF adopts a 20 MHz sampling rate deliberately chosen for edge-device compatibility, as opposed to RFUAV's 100 MHz rate that imposes impractical computational demands (100 MS/s) for real-time deployment where signal capture, spectrogram generation, and inference must execute within tight computational budgets.

4. **Interoperable tooling**: CDRF provides dataset-agnostic utilities that operate on existing public benchmarks, with tools for dataset creation, metadata generation, cleaning, and evaluation. The YOLO toolkit includes raw-IQ augmentation and automatic bounding-box recomputation, while the data module loads .dat files via memory mapping and slices long recordings into time windows.

## Experimental Results
CDRF enables standardized benchmarking for classification, open-set recognition, and object detection across challenging conditions. While the abstract doesn't provide specific performance numbers, it notes that models trained on clean datasets (like DroneRF) achieve near-perfect scores on easy benchmarks but degrade severely under interference and low SNR. CDRF's dual-environment collection (Faraday cage and outdoor) provides the first benchmark spanning 23 drone models with diverse environmental conditions, addressing the consistent realism gap identified in prior work.

## Related Work
CDRF builds upon existing RF drone datasets like DroneRF (3 drone models), DroneDetect V2 (7 models), and RFUAV (37 UAV types), but addresses critical limitations. Unlike these benchmarks that distribute pre-rendered spectrograms with fixed annotations (requiring manual re-annotation for signal modifications), CDRF maintains a complete, parameterized processing chain from raw I/Q through to detection annotation. This end-to-end traceability enables unlimited programmatic generation of correctly labelled training samples from any raw recording under any processing parameters, a capability absent from all prior RF drone datasets.

## Limitations
The authors acknowledge that CDRF is based on captures from a single university campus and a controlled RF-cage facility, limiting geographic diversity. While the dataset spans 23 drone models, it doesn't cover all commercially available drones, particularly military-grade systems. The benchmark focuses on consumer and professional drones, so models may not generalise to military UAVs with different RF signatures. The abstract doesn't specify whether the paper includes comparisons to military-grade systems.

## Appendix: Worked Example
Consider a 20 ms segment of raw I/Q data (400 samples at 20 MHz sampling rate) from a DJI Mavic Pro at 2.4 GHz. The spectrogram uses 512-point FFT, 20 ms window length, and 10 ms hop size, resulting in a 128x128 spectrogram image. A bounding box originally covering pixels [20-30, 40-60] (representing frequency range 20-30) needs adjustment for a 500 Hz frequency shift.

1. Calculate pixel shift: 500 Hz / (20 MHz / 512) = 12.8 pixels
2. Adjust bounding box coordinates: [20+12.8, 30+12.8] → [32.8, 42.8]
3. Handle wrap-around: Since the spectrogram spans 128 pixels (0-127), 42.8 stays within bounds
4. Rounded annotation: [33-43, 40-60]

This process ensures bounding boxes remain accurate even after signal transformations, eliminating the need for manual re-annotation. The raw-signal augmentation pipeline automatically recomputes these coordinates for any signal modification, enabling efficient model iteration (see Appendix for implementation details).

## References

- **Code:** https://github.com/DroneGoHome/U-RAPTOR-PUB.
- Mohammad Rostami, Atik Faysal, Hongtao Xia, Hadi Kasasbeh, Ziang Gao, Huaxia Wang, "CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2601.03302

Tags: #security-and-privacy #drone-detection #rf-communication #dataset-augmentation #edge-computing
