---
title: "Offshore oil and gas platform dynamics in the North Sea, Gulf of Mexico, and Persian Gulf: Exploiting the Sentinel-1 archive"
venue: "Gulf of Mexico"
paper_url: "https://arxiv.org/abs/2603.19801"
---

## Executive Summary
This paper presents a scalable system for automated detection and monitoring of offshore oil and gas platforms using freely available Sentinel-1 satellite data and deep learning. It creates a consistent quarterly time series of platform locations across three major production regions (North Sea, Gulf of Mexico, Persian Gulf) from 2017-2025, tracking lifecycle information and spatial attributes. Engineers should care because it demonstrates a practical approach to using publicly available Earth observation data for consistent, long-term monitoring of maritime infrastructure, which could inform energy infrastructure planning and environmental management systems.

## Why This Matters for Practitioners
For teams building environmental monitoring or infrastructure management systems, this work provides a template for using freely available satellite data for scalable monitoring without needing expensive proprietary solutions. If you're implementing a system that needs to track infrastructure changes over time across large geographical areas, this paper shows how to leverage open datasets (like Sentinel-1) with deep learning to create a consistent time series at scale. Specifically, the approach of using SAR (Synthetic Aperture Radar) data instead of optical imagery avoids cloud coverage issues that plague traditional monitoring approaches. The quarterly time series approach means you can now design systems that update infrastructure inventories on a regular cadence without needing to develop custom data acquisition pipelines for each region. The dataset they've created (OPD v1.0.0) is publicly available, so you can immediately start using it to enrich your own monitoring systems rather than starting from scratch.

## Problem Statement
Today's monitoring of offshore infrastructure is like trying to track a moving target with a broken map - regional reporting systems are incomplete, inconsistent, and often rely on data that's not publicly available due to security concerns. Just as a city planner would struggle to manage traffic if they only had partial, outdated maps of street layouts, maritime authorities struggle to manage infrastructure when they lack consistent, long-term data on platform locations. The problem isn't just about missing data; it's about the fragmented nature of that data, with different regions having different monitoring approaches that don't work together.

## Proposed Approach
The authors developed a modular workflow that uses Sentinel-1 satellite data and a deep learning model to detect offshore platforms consistently across large geographical areas. The system works in several stages: data acquisition and preprocessing (creating median composites from Sentinel-1 data), inference with a YOLOv10 model, postprocessing to clean detections, spatiotemporal consolidation to group detections into individual platforms, and spatial enrichment to add attributes like water depth and distance to coast. The key innovation is using SAR data (which works regardless of weather conditions) combined with a deep learning model trained to generalise across different marine environments.

```python
def detect_offshore_platforms(sentinel_1_data):
    # Preprocess: create median composites for each quarter
    median_composites = preprocess_sentinel_1_data(sentinel_1_data)
    
    # Inference: run YOLOv10 model on the composites
    detections = yolo_v10_inference(median_composites, confidence_threshold=0.4)
    
    # Postprocessing: remove false positives (low backscatter) and duplicates
    clean_detections = postprocess_detections(detections)
    
    # Spatiotemporal consolidation: group detections into individual platforms
    individual_platforms = consolidate_platforms(clean_detections)
    
    # Spatial enrichment: add attributes like water depth, distance to coast
    enriched_platforms = enrich_platforms(individual_platforms)
    
    return enriched_platforms
```

## Key Technical Contributions
The paper makes several key technical contributions that enable consistent monitoring across diverse offshore environments.

The authors developed a transferable deep learning model that generalises across different marine regions without requiring region-specific training data. This is achieved by training the model on data from the South China Sea, Caspian Sea, Gulf of Guinea, and Brazilian coast while testing on entirely independent regions (North Sea, Persian Gulf, Gulf of Mexico), ensuring generalisation to unseen environments. The model achieved a precision of 0.91, recall of 0.89, and F1 score of 0.90 for the combined platform class at a confidence threshold of 0.5 and intersection-over-union (IoU) of 0.3.

They created a robust spatiotemporal consolidation approach that handles the dynamic nature of platform operations. Rather than treating each quarter's detection as independent, they merged overlapping detections across quarters using an IoU threshold of 0.1 to form unique platform clusters that represent individual physical structures over time. This approach accommodates the fact that platforms operate continuously over long periods rather than through repeated short-term installation and decommissioning cycles.

They developed a scalable processing pipeline that handles the massive volume of Sentinel-1 data. By creating a 1.8° grid, generating median composites per quarter, and processing in tiles, they managed the computational complexity of analysing nearly 4,103 quarterly median composite tiles across the 33 quarters from 2017-2025.

## Experimental Results
The system achieved a macro-weighted F1 score of 0.884 across all three regions, outperforming other freely available datasets (Paolo et al.: 0.823; OOGPs v1.0: 0.868; OpenStreetMap: 0.430). Regionally, OPD v1.0.0 attained the highest F1 scores in the Persian Gulf (0.925) and Gulf of Mexico (0.868), while in the North Sea (0.860), it remained above Paolo et al. (0.860) and OpenStreetMap (0.430), though OOGPs v1.0 reached a higher F1 score (0.920) in that region. The authors evaluated performance on a ground truth test dataset from Spanier et al. 2026, using standard accuracy metrics (precision, recall, F1 score) with a confidence threshold of 0.4.

## Related Work
The paper positions itself within the growing field of using satellite data for monitoring infrastructure. Previous work on automated detection of offshore targets has often relied on rule-based methods (constant false alarm rate detectors or threshold-based methods with geometric filters), which are limited in their ability to handle the complexity and variability of offshore structures. Optical and multispectral approaches were used in some studies but are limited by atmospheric disturbances like clouds and haze. The authors build on this by demonstrating the effectiveness of SAR (Synthetic Aperture Radar) data, which operates independently of weather conditions and provides consistent backscatter signatures for infrastructure detection. They extend the work of Hoeser et al. (2022) who developed DeepOWT for offshore wind energy infrastructure, applying similar techniques to oil and gas platforms while adding a comprehensive time series and spatial enrichment component.

## Limitations
The authors acknowledge that their approach relies on Sentinel-1 data, which has regional variations in acquisition density (higher density in Europe, lower in America), potentially affecting detection consistency in less covered regions. While they achieved good results across the three regions, the model was trained on different regions (South China Sea, Caspian Sea, Gulf of Guinea, and Brazilian coast) and tested on the three study areas, but the transferability to other regions with very different infrastructure layouts or environmental conditions might be limited. The paper doesn't explicitly address how the system would handle regions with extremely high platform density or areas with complex coastal environments that might cause false positives.

## Appendix: Worked Example
Let's walk through a specific example of how the system would process data for a single platform in the Persian Gulf:

Start with Sentinel-1 data for the Persian Gulf region covering 2023Q1-2023Q4 (four quarters of data). The system processes each quarter independently, creating median composites. The YOLOv10 model detects a platform in all four quarters with confidence scores of 0.78, 0.82, 0.85, and 0.75. The system then removes detections with confidence below 0.4 and those with backscatter values below 150 (≈ -16.5 dB), which are all above the threshold.

For spatiotemporal consolidation, the system identifies that the four detections are spatially overlapping (IoU > 0.1) and merges them into a single platform cluster. The platform is first detected in 2023Q1 and last detected in 2023Q4, giving it a lifetime of four quarters (one year). The system then adds spatial attributes: the minimum distance to the coastline is calculated as 45 km (using a global coastline dataset), the water depth at the platform location is 38 meters (from GEBCO global bathymetry), and the bounding box area (indicating platform size) is 2,300 square meters (representing the backscatter signature of the platform).

This process is repeated for all platforms across all quarters, creating a time series that shows the dynamics of platform installations and decommissioning over time.

## References

- Robin Spanier, Thorsten Hoeser, John Truckenbrodt, Felix Bachofer, Claudia Kuenzer, "Offshore oil and gas platform dynamics in the North Sea, Gulf of Mexico, and Persian Gulf: Exploiting the Sentinel-1 archive", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19801

Tags: #environmental-science #computer-vision #deep-learning #object-detection #time-series
