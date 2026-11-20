# PathWise — Computer Vision Concepts and Technical Reference

**Project:** PathWise\
**Purpose:** Reference document for understanding the computer-vision and video-processing technologies used in PathWise.

**Learning strategy:** Build the application first. Study the concepts and implementation after the development milestones are complete.

---

# 1. Project Overview

PathWise is an Android application for pedestrian accessibility and footpath-condition mapping.

Its computer-vision system will analyse photographs and videos of pedestrian infrastructure to identify:

- Cracked and damaged footpaths.
- Broken or uneven pavement.
- Parked vehicles obstructing footpaths.
- Poles, bins, barriers and other physical obstacles.
- Other accessibility-related infrastructure issues as the project evolves.

The application will allow users to review AI-generated observations, correct predictions and save reports with geographical information.

The initial implementation will focus on two separate computer-vision tasks:

1. Semantic segmentation for footpath damage.
2. Object detection for pedestrian obstacles.

Video processing will connect these capabilities to real-world smartphone recordings.

# 2. Image Processing Fundamentals

## 2.1 Digital Images and Pixels

An image is represented as a matrix of numerical pixel values.

An RGB image contains three channels:

- Red
- Green
- Blue

Each pixel typically contains three values between 0 and 255 when stored as an 8-bit image.

**Why PathWise uses this:**

Images captured by the Android camera will be decoded, resized, converted and normalised before being passed to a neural network.

## 2.2 Image Dimensions and Resolution

An image has a width, height and number of channels.

For example:

- 1920 × 1080: high-resolution image.
- 256 × 256: smaller model input.

**Why PathWise uses this:**

Smaller images reduce CPU computation and memory requirements. However, excessive resizing can remove small cracks and other important details.

Resolution will therefore be evaluated experimentally.

## 2.3 Image Normalisation

Normalisation converts pixel values into a numerical range suitable for model training.

A common transformation is:

\[\
x' = \\frac{x}{255}\
\]

This converts values from 0–255 into 0–1.

**Why PathWise uses this:**

It provides a consistent input representation and helps numerical stability during neural-network training.

## 2.4 Image Interpolation

When images are resized, new pixel values must be calculated.

Common methods:

- Nearest neighbour.
- Bilinear interpolation.
- Bicubic interpolation.

**Why PathWise uses this:**

Photographs can use bilinear or bicubic interpolation. Segmentation masks should generally use nearest-neighbour interpolation because class IDs must remain discrete.

# 3. Semantic Segmentation

## 3.1 What Is Semantic Segmentation?

Semantic segmentation assigns a class to every pixel in an image.

For PathWise, the initial task is binary segmentation:

| **Class ID** | **Meaning** |
| --- | --- |
| 0 | Background / non-damage |
| 1 | Footpath damage |

The output is a mask showing which pixels the model believes belong to damaged pavement.

**Why PathWise uses this:**

Cracks and broken pavement have irregular shapes. Bounding boxes cannot accurately represent their boundaries.

## 3.2 Binary Segmentation Masks

A ground-truth mask contains the correct pixel labels supplied by the dataset.

For a binary mask:

- 0 represents background.
- 1 represents damage.

The original photograph and its corresponding mask form a training pair.

**Why PathWise uses this:**

The model learns by comparing its predicted mask with the ground-truth mask.

## 3.3 Semantic vs Instance Segmentation

Semantic segmentation labels pixels by class.

Instance segmentation additionally distinguishes individual objects belonging to the same class.

Example:

- Semantic: all damaged pavement pixels are labelled as damage.
- Instance: three separate damaged pavement regions receive separate instance identities.

**PathWise decision:** Begin with semantic segmentation. Instance segmentation is not required for the first damage experiment.

# 4. Convolutional Neural Networks

## 4.1 Convolutional Neural Networks (CNNs)

CNNs are neural networks designed to process spatial data such as images.

They learn visual features through convolutional filters.

Early layers may learn:

- Edges.
- Lines.
- Corners.
- Basic textures.

Deeper layers may learn more complex patterns and structures.

**Why PathWise uses this:**

Footpath damage has visual patterns involving cracks, edges, texture irregularities and broken surfaces.

## 4.2 Convolution

A convolution applies a small filter, also called a kernel, across an image or feature map.

The operation extracts local visual information.

**Why PathWise uses this:**

Convolutional layers allow the segmentation model to learn visual features automatically instead of relying entirely on manually designed image-processing rules.

## 4.3 Pooling

Pooling reduces the spatial dimensions of feature maps.

Max pooling is a common example.

**Why PathWise uses this:**

It reduces computation and allows the network to learn increasingly abstract visual features.

However, spatial information is important for segmentation, so the decoder must reconstruct fine details.

# 5. U-Net Architecture

## 5.1 What Is U-Net?

U-Net is an encoder-decoder neural-network architecture designed for image segmentation.

Its major components are:

1. Encoder.
2. Bottleneck.
3. Decoder.
4. Skip connections.

## 5.2 Encoder

The encoder extracts visual features while progressively reducing spatial resolution.

**Why PathWise uses it:**

It learns increasingly complex patterns associated with damaged pavement.

## 5.3 Bottleneck

The bottleneck is the deepest representation in the network.

It contains compressed, higher-level visual information.

**Why PathWise uses it:**

It allows the network to combine broader visual context with lower-level features.

## 5.4 Decoder

The decoder progressively increases spatial resolution and generates the segmentation output.

**Why PathWise uses it:**

The model must return a pixel-level prediction, not merely classify the entire image.

## 5.5 Skip Connections

Skip connections transfer feature information from encoder layers to corresponding decoder layers.

**Why PathWise uses them:**

Fine spatial details can disappear during downsampling. Skip connections help recover these details, which is particularly important for thin cracks.

## 5.6 Why Start with a Small U-Net?

A small U-Net provides:

- A relatively simple architecture.
- A manageable number of parameters.
- An understandable implementation.
- A useful segmentation baseline.
- CPU-friendly experimentation compared with larger models.

The initial architecture will use fewer feature channels than a conventional large U-Net.

# 6. Training a Segmentation Model

## 6.1 Training Dataset

The training dataset contains images and their corresponding ground-truth masks.

The model learns from these examples by adjusting its parameters.

## 6.2 Forward Pass

The image is passed through the neural network.

The model produces a tensor of predicted damage scores, called logits.

## 6.3 Logits and Sigmoid

A logit is an unrestricted numerical output.

For binary segmentation, sigmoid transforms a logit into a value between 0 and 1:

\[\
\\sigma(x)=\\frac{1}{1+e^{-x}}\
\]

These values are commonly interpreted as damage scores.

A threshold converts the scores into binary predictions.

For example, a threshold of 0.5 means:

- Score greater than or equal to 0.5: damage.
- Score below 0.5: background.

**Why PathWise uses this:**

It converts neural-network outputs into an interpretable damage mask.

## 6.4 Loss Functions

A loss function measures how different a model prediction is from the ground truth.

The initial experiment will use Binary Cross Entropy and Dice Loss.

### Binary Cross Entropy (BCE)

Measures the difference between predicted pixel scores and actual binary labels.

### Dice Loss

Encourages overlap between the predicted and actual damage regions.

A common formulation is:

\[\
L\_{Dice}=1-\\frac{2\|P\\cap G\|+\\epsilon}{\|P\|+\|G\|+\\epsilon}\
\]

Where:

- P = predicted damage mask.
- G = ground-truth damage mask.
- ε = small smoothing constant.

### Combined Loss

The initial experiment proposes:

\[\
L=L\_{BCE}+L\_{Dice}\
\]

**Why PathWise uses both:**

BCE provides pixel-wise supervision, while Dice Loss encourages overlap. This combination is useful when damaged pixels occupy a relatively small portion of an image.

## 6.5 Backpropagation

Backpropagation calculates gradients indicating how model parameters should change to reduce the loss.

## 6.6 Optimisation

An optimiser updates the model parameters using the calculated gradients.

The initial experiment uses Adam.

## 6.7 Learning Rate

The learning rate determines the size of parameter updates.

A learning rate that is too high may cause unstable training. One that is too low may make training unnecessarily slow.

## 6.8 Epochs and Batches

An epoch represents one complete pass through the training dataset.

A batch is a subset of training examples processed together.

**Why PathWise uses these:**

Batch size and epoch count control training behaviour and computational requirements.

The initial CPU experiment uses a small batch size and a limited number of epochs.

# 7. Data Preparation and Augmentation

## 7.1 Image-Mask Pairing

Every photograph must correspond to its correct segmentation mask.

An incorrect pairing teaches the model incorrect relationships.

## 7.2 Data Augmentation

Augmentation creates modified training examples using transformations such as:

- Horizontal flipping.
- Small rotations.
- Cropping.
- Brightness changes.
- Contrast changes.

**Why PathWise uses this:**

The model should learn from variations in camera orientation, lighting and pavement appearance.

Geometric transformations must be applied consistently to both the image and its mask.

## 7.3 Train, Validation and Test Sets

The dataset is divided into three groups.

**Training:** Used to update model parameters.

**Validation:** Used to evaluate training progress and select model configurations.

**Testing:** Used for final evaluation after model selection.

The official dataset splits should be preserved.

## 7.4 Overfitting

Overfitting occurs when a model performs well on training examples but poorly on unseen examples.

**Why PathWise monitors it:**

The dataset is relatively small. A model may memorise training images instead of learning patterns that generalise to new footpaths.

# 8. Model Evaluation

Pixel accuracy alone is insufficient for segmentation.

PathWise will use the following metrics.

## 8.1 True Positive, False Positive, False Negative and True Negative

- TP: Damage correctly predicted as damage.
- FP: Background incorrectly predicted as damage.
- FN: Actual damage missed by the model.
- TN: Background correctly predicted as background.

## 8.2 Intersection over Union (IoU)

\[\
IoU=\\frac{TP}{TP+FP+FN}\
\]

Measures the overlap between predicted and ground-truth damage regions.

## 8.3 Dice Coefficient

\[\
Dice=\\frac{2TP}{2TP+FP+FN}\
\]

Measures segmentation overlap.

## 8.4 Precision

\[\
Precision=\\frac{TP}{TP+FP}\
\]

Measures how many predicted damage pixels were actually damage.

## 8.5 Recall

\[\
Recall=\\frac{TP}{TP+FN}\
\]

Measures how much actual damage the model successfully identifies.

## 8.6 Threshold Selection

A threshold converts continuous pixel scores into binary predictions.

The initial threshold may be 0.5, but different thresholds can affect precision and recall.

Threshold tuning must use validation data, not the test set.

## 8.7 Error Analysis

Inspect examples where the model:

- Detects actual damage correctly.
- Misses thin cracks.
- Mistakes shadows for damage.
- Mistakes pavement textures for cracks.
- Fails under different lighting.
- Predicts incorrect damage boundaries.

**Why PathWise uses this:**

The objective is to understand the model's limitations, not merely report one numerical score.

# 9. Video Processing with OpenCV

This section will become important when PathWise moves from photographs to recorded footpath videos.

## 9.1 Video as a Sequence of Frames

A video consists of individual images called frames.

Each frame can be processed independently by the segmentation or detection model.

## 9.2 Frame Rate (FPS)

FPS represents the number of frames displayed or captured per second.

For example, a 30 FPS video contains 30 frames per second.

## 9.3 Frame Extraction

OpenCV can decode videos and extract selected frames.

**Why PathWise uses this:**

Running inference on every frame is unnecessary for an initial prototype and can be computationally expensive.

## 9.4 Frame Sampling

Frame sampling selects frames at controlled intervals.

For example, a 30 FPS video could initially be sampled at one frame per second.

**Why PathWise uses this:**

It reduces computational requirements and avoids processing many almost-identical images.

## 9.5 Timestamps

Each extracted frame should retain its original video timestamp.

**Why PathWise uses this:**

Timestamps help associate predictions with the correct point in the recorded sequence.

## 9.6 Duplicate Detection Suppression

A single obstruction may appear in multiple consecutive frames.

The system should avoid generating a separate report for every appearance.

Possible future techniques:

- Intersection over Union between bounding boxes.
- Object tracking.
- Detection association across frames.
- Spatial proximity and timestamp-based grouping.

This will initially be a post-processing task rather than part of the segmentation model itself.

## 9.7 Temporal Consistency

A model may predict slightly different masks in neighbouring frames.

Temporal consistency techniques can help reduce unstable predictions.

This is a future improvement, not a requirement for the first segmentation baseline.

# 10. Object Detection — Second CV Workstream

PathWise will eventually detect discrete obstacles.

Examples:

- Motorcycles.
- Cars.
- Poles.
- Bins.
- Barriers.
- Vendor carts.

Object detection produces:

- Class label.
- Bounding box.
- Confidence score.

The planned model family is a compact YOLO-style detector, subject to dataset and model-licence verification.

Important concepts to study later:

- Bounding boxes.
- IoU.
- Confidence thresholds.
- Non-Maximum Suppression (NMS).
- Precision and recall.
- Mean Average Precision (mAP).
- Object detection versus segmentation.

**Important:** Detecting a motorcycle does not automatically mean it obstructs a footpath. Determining obstruction requires spatial context and human review.

# 11. Computer-Vision Integration

## 11.1 PyTorch

PyTorch will be used for model development, training and evaluation.

## 11.2 OpenCV

OpenCV will handle image and video decoding, frame extraction and preprocessing.

## 11.3 FastAPI

FastAPI will expose the CV pipeline through HTTP endpoints.

Android will submit selected images or videos and receive structured predictions.

## 11.4 Prediction Output

The inference service should return structured information, such as:

- Model version.
- Input dimensions.
- Predicted mask or mask reference.
- Confidence or pixel-score information.
- Processing duration.
- Relevant frame timestamp.

This creates a stable interface between the CV engine and Android.

# 12. Hardware and Performance

The development environment is:

- Windows laptop.
- 16 GB RAM.
- No NVIDIA GPU.
- Redmi Note 11 Android test device.

Important concepts:

- CPU inference.
- Memory consumption.
- Batch size.
- Image resolution.
- Model parameter count.
- Inference latency.
- Model optimisation.

The first implementation should use small CPU-feasible experiments.

Possible future optimisations:

- ONNX Runtime.
- Quantisation.
- TensorFlow Lite or LiteRT.
- On-device inference.

These are not required for the initial baseline.

# 13. Dataset Domain Shift

The public footpath dataset contains images from Pune, while our own collection will be around VIT Vellore.

A model trained on one environment may perform differently in another.

This is called domain shift.

Potential causes:

- Different paving materials.
- Different lighting.
- Different cameras.
- Different camera heights.
- Different weather.
- Different types of footpath damage.

Our local Vellore dataset will eventually help evaluate and improve generalisation.

# 14. Future Accessibility Reasoning

Segmentation identifies visual damage, but it does not establish whether a pedestrian route is accessible.

Future capabilities may require:

- Footpath-region segmentation.
- Obstacle detection.
- Curb-ramp recognition.
- Walkable-width estimation.
- Spatial relationships between obstacles and walking regions.
- Human verification.

PathWise must distinguish visual predictions from verified accessibility assessments.

# 15. Learning Roadmap After Development

Recommended learning order:

1. Image processing fundamentals.
2. Binary masks and semantic segmentation.
3. CNNs and convolution.
4. U-Net architecture.
5. PyTorch tensors and training.
6. BCE and Dice Loss.
7. IoU, Dice, precision and recall.
8. Dataset augmentation and overfitting.
9. Object detection and YOLO.
10. OpenCV video processing.
11. Frame sampling and timestamps.
12. Object tracking and temporal consistency.
13. Model deployment and inference optimisation.

---

# Final Note

PathWise is intended to be a practical learning project and a demonstrable portfolio application.

The first objective is to build a functioning end-to-end system. The second objective is to understand how its components work.

Do not treat every future capability described here as an implemented feature. Distinguish:

- Implemented functionality.
- Planned functionality.
- Experimental functionality.
- Future research possibilities.

Keep this document updated as the architecture and experiments evolve.

**The ultimate learning goal:** Be able to explain, reproduce, evaluate and improve every important computer-vision component in PathWise without depending on an AI coding agent to explain the project for you.
