## 1 Introduction

Lidar-based 3D object detection enhances how machines perceive and navigate their environment -enabling accurate tracking, motion forecasting, and planning. Lidar data can be represented in various forms such as unordered points, 3D voxel grids, bird's-eye view projections, and rangeview representations. Each representation differs in terms of its sparsity and how it encodes spatial relationships between points. Point-based representations preserve all information but compromise on the efficient computation of spatial relationships between points. Voxel-based and bird's-eye view representations suffer from information loss and sparsity, yet they maintain efficient neighborhood computation. The range-view representation preserves the data losslessly and densely in the "native" view of the sensor, but 2D neighborhoods in such an encoding can span enormous 3D distances and objects exhibit scale variance because of the perspective viewpoint.

The field of range-view-based 3D object detection is relatively less explored than alternative representations. Currently, the research community focuses on bird's-eye view or voxel-based methods. This is partially due to the performance gap between these models and range-view-based models.

## 2 Related Work

Point-based Methods. Point-based methods aim to use the full point cloud without projecting to a lower dimensional space [8,9,10]. PointNet [8] and Deep Sets [11] introduced permutation invariant networks which enabled direct point cloud processing, which eliminates reliance on the relative ordering of points. PointNet++ [9] further improved on previous point-based methods, but still remained restricted to the structured domain of indoor object detection. While PointRCNN [10] extends methods to the urban autonomous driving domain, point-based methods do not scale well with the number of points and size of scenes, which makes them unsuitable for real-time, low-latency safety critical applications.

Grid-based Projections. Grid-based projection methods first discretize the world into either 2D Bird's Eye View [12,13,14,15] or 3D Voxel [16,17,18,19] grid cells, subsequently placing 3D  lidar points into their corresponding 2D or 3D cell. These methods often result in collisions, where point density is high and multiple lidar points are assigned to the same grid cell. While some methods resolve these collisions by simply selecting the nearest point [12], others use a max-pooled feature [14] or a learned feature fusion [19,13].

Range-based Projections. A range view representation is a spherical projection that maps a point cloud onto a 2D image plane, with the result often referred to as a range image. Representing 3D points as a range image yields a few notable advantages: (a) memory-efficient representation, i.e., the image can be constructed in a way in which few pixels are "empty", (b) implicit encoding of occupancy or "free-space" in the world, (c) compute-efficient -image processing can be performed with dense 2D convolution, (d) scaling to longer ranges is invariant of the discretization resolution. A range image can be viewed as a dense, spherical indexing of the 3D world. Notably, spherical projections are O(1) in space complexity as a function of range at a fixed azimuth and inclination resolution. Due to these advantages, range view has been adapted in several works for object detection [1,20,21,5,3]. Subsequent works have explored the range view for joint object detection and forecasting [22,23].

Multi-View Projections. To leverage the best of both projections, recent work has explored multiview methods [19,13,15,24]. These methods extract features from both the bird's-eye view and the range view and fuse them to generate object proposals. While Zhou et al. [19] proposes to fuse multi-view features in the point space, Laddha et al. [13] suggests fusing these features by projecting range view features to the bird's-eye view. Fadadu et al. [15] investigates the fusing of RGB camera features with multi-view lidar features. In this work we explore how competitive range-view representations can be without these additional views.

## 3 Range-view D Object Detection

We begin by describing the range view and its features, then we outline the task of 3D object detection and its specific formulation in the range view. Next, we describe a common architecture for a range-view-based 3D object detection model.

Inputs: Range View Features. The range view is a dense grid of features captured by a lidar sensor. In Fig. 1, we illustrate the range view, its 3D representation along with an explicit geometric correspondence of a building between views and the location of the physical lidar sensor which captures the visualized 3D data. Fig. 2 shows the range-view features used for our Waymo Open experiments. We show a range image, the object confidences from a network, and their corresponding 3D cuboids shown in the bird's-eye view for a scene with multiple parked vehicles. For each visible point in the range image, our range-view 3D object detection model learns (1) which category an object belongs to (2) the offset from the visible point to the center of the object, its 3D size, and its orientation. In the above example, we show one particular point (Point A) from two different perspectivesthe range view and the bird's-eye view. Blue boxes indicate the ground truth cuboids, green boxes indicate true positives, and red boxes indicate false positives. Importantly, each object can have many thousands of proposals -however, most will be removed through non-maximum suppression.

3D Object Detection. Given range view features shown in Fig. 2, we seek to map over 50,000 3D points to a much smaller set of objects in 3D and describe their location, size, and orientation. Given the difficulty of this problem, approaches usually "anchor" their predictions on a set of 3D "query" points (3D points which are pixels in range view features). For each 3D point stored as features in a range image, we predict a 3D cuboid which describes a 3D object. Fig. 3 illustrates the range view input (top), the smaller set of object points (middle), and the regressed object cuboids prior to non-maximum suppression (bottom). Importantly, each object may contain thousands of salient points which are transformed into 3D object proposals.

### 4.1 Datasets

Argoverse 2. The dataset contains 1,000 sequences of synchronized, multi-modal data. The dataset contains 750 training sequences, 150 validation sequences, and 150 testing sequences. In our experiments, we use the top lidar sensor to construct an 1800 × 32 range image. The official Argoverse 2 3D object detection evaluation contains 26 categories evaluated at a 150 m range with the following metrics: average precision (AP), average translation (ATE), scaling (ASE), and orientation (AOE) errors, and a composite detection score (CDS). AP is a VOC-style computation with a true positive defined at 3D Euclidean distance averaged over 0.5 m, 1.0 m, 2.0 m, and 4.0 m. We outline additional information in supplemental material and refer readers to Wilson et al. [6] for further details.

Waymo Open. The Waymo Open dataset [7] contains three evaluation categories, Vehicle, Pedestrian, and Cyclist, evaluated at a maximum range of approximately 80 m. We use the training split (798 logs) and the validation split (202 logs) in our experiments. The dataset contains one medium-range and four near-range lidar sensors. The medium range lidar sensor is distributed as a single, dense range image. We utilize the 2650 × 64 range image from the medium-range lidar for all experiments. We evaluate our Waymo experiments using 3D Average Precision (AP). Following RangeDet [5], we report Level-1 (L1) results. Additional details can be found in supplemental material and the original paper [7].

### 4.2 Experiments

In this section, we report our experimental results. Full details on our baseline model can be found in the supplemental material. Table 2: Input Feature Dimensionality: Waymo Open. Level-1 Mean Average Precision across two input feature dimensionalities d on the Waymo Open validation set. Using a larger input feature dimensionality leads to a notable improvement across categories. Further scaling was limited by available GPU memory.

Input Feature Dimensionality. We find that input feature dimensionality plays a large role in classification and localization performance. We explore scaling feature dimensionality of the high resolution pathway in both the backbone, the classification, and regression heads.

In Table 1, we find that performance on Argoverse 2 consistently improves when doubling the backbone feature dimensionality. Additionally, the error metrics continue to decrease despite increasingly challenging true positives being detected. We report a similar trend in Table 2 on Waymo Open. We suspect that the performance improvements are largely due to learning difficult variances found in the range-view (e.g. scale variance and large changes in depth). We choose an input feature dimensionality of 256 and 128 for our state-of-the-art comparison on Argoverse 2 and Waymo Open, respectively, to balance performance and runtime.

3D Input Encoding. Close proximity of pixels in a range image does not guarantee that they are close in Euclidean distance. Previous literature has explored incorporating explicit 3D information to better retain geometric information into the input encodings. We re-implement two of these methods: the Meta-Kernel from RangeDet [5] and the Range-aware Kernel (RAK) from RangePerception [2]. We find that the Meta-Kernel outperforms our baseline by 2% mAP and 1.4% CDS. Additionally, the average translation, scale, and orientation errors are reduced. Unexpectedly, we are unable to reproduce the performance improvement from the Range Aware Kernel. On the Waymo Open dataset, we find that the Meta-Kernel yields a 4.17% and 5.8% improvement over the Vehicle and Pedestrian categories. Consistent with our results on Argoverse 2, the Range Aware Kernel fails to reach our baseline performance. Full results are in the supplemental material. We will adopt the Meta-Kernel for our state-of-the-art comparison.

Dynamic 3D Classification Supervision. We compare two different strategies across the Argoverse 2 and Waymo datasets. In Fig. 5, we illustrate the difference between the two different methods, Dynamic IoU BEV and our proposed Dynamic 3D Centerness. On Argoverse 2, our Dynamic 3D centerness outperforms IoU BEV by by 2.7% mAP and 1.9% CDS. We speculate that this performance improvement occurs because Argoverse 2 contains many small objects e.g. bollards, construction cones, and construction barrels, which receive low classification scores due to translation error under IoU-based metrics. Dynamic 3D Centerness also incurs less translation, scale, and orientation errors than competing rankings. The optimal ranking strategy remains less evident for the Waymo dataset. The official Waymo evaluation uses vehicle, pedestrian, and cyclist as their object evaluation categories, which are larger on average than many of the smaller categories in Argoverse 2. We find that Dynamic IoU BEV and Dynamic 3D Centerness perform nearly identically at 59.9% AP; however, for smaller objects such as pedestrian, Dynamic 3D Centerness outperforms IoU BEV by 0.95. Full tables are in the supplemental material. Our experiments suggest that IoU prediction is unnecessary for strong performance on either dataset. We adopt Dynamic 3D Centerness for our state-of-the-art comparison since it performs well on both datasets.

Range-based Sampling. We compare our baseline architecture (single resolution prediction head with no sub-sampling) against the Range-conditioned Pyramid (RCP) [5] and our Range Subsampling (RSS) approach. In Table 3, we surprisingly find that RCP performs worse than our baseline model in overall performance; however, it yields a modest improvement in runtime by reducing the total number of object proposals processed via NMS. By sampling object proposals as a post-processing step, our method, RSS, outperforms both RCP and the baseline with no additional parameters or network complexity, and comparable runtime. Similarly, we examine the impact of range-based sampling across the Waymo Open dataset in Table 4. We find that the Range-conditioned pyramid yields marginal performance against our baseline despite having 2.8x the number of parameters in the network heads. We speculate that feature-pyramid-network (FPN) approaches are not as effective in the range-view since objects cannot be normalized in the manner proposed by the original FPN [27]. We will adopt RSS in our state-of-the-art comparison.

Method Head Params. (M) mAP ↑ ATE ↓ ASE ↓ AOE ↓ CDS ↑ FPS ↑ Baseline 1.2 16.1 0.78 0.47 1.04 12.1 11.6 RCP [5] 3.4 16.6 0.74 0.47 1.03 12.5 23.20 RSS (ours) 1.2 16.9 0.77 0.46 1.04 12.8 24.87

Table 3: Subsampling by Range: Argoverse 2. Argoverse 2 evaluation metrics on the validation set. We compare our baseline against two different subsampling strategies. The range-conditioned pyramid modifies the network architecture with each multi-resolution head responsible for a range-interval. In contrast, our RSS approach is a parameter free approach and only changes the subsampling procedure before NMS. Comparison against State-of-the-Art. Combining a scaled input feature dimensionality with Dynamic 3D Centerness and Range-based sampling yields a model which is competitive with existing voxel-based methods on the Argoverse 2 dataset, and state-of-the-art amongst Range-view models on Waymo Open. In Table 5, we report our mAP over the 26 categories in Argoverse 2. We outperform VoxelNext [28], the strongest voxel-based model, by 0.9% mAP. In Table 6, we show our L1 AP against a variety of different models on Waymo Open. Our method outperforms all existing range-view models while also being multi-class. Mean R. Vehicle Pedestrian Bollard C. Barrel C. Cone S. Sign Bicycle L. Vehicle B. Truck W. Device Sign Bus V. Trailer Truck Motorcycle T. Cab Distribution (%) -56.92 17.95 6.8 3.62 2.63 1.99 1.42 1.25 1.09 1.06 0.91 0.83 0.69 0.54 0.47 0.44 mAP ↑ CenterPoint [29] 22.0 67.6 46.5 40.1 32.2 29.5 -24.5 3.9 37.4 -6.3 38.9 22.4 22.6 33.4 -FSD [30] 28.2 68.1 59.0 41.8 42.6 41.2 -38.6 5.9 38.5 -11.9 40.9 26.9 14.8 49.0 -VoxelNext [28] 30.7 72.7 63.2 53.9 64.9 44.9 -40.6 6.8 40.1 -14.9 38.8 20.9 19.9 42.4 Ours 34.4 76.5 69.1 50.0 72.9 51.3 39.7 41.4 6.7 36.2 23.1 20.0 48.4 24.7 24.2 51.3 21.9 Table 6: Comparison against State-of-the-Art: Waymo Open. We compare our range-view model against different state-of-the-art methods on the Waymo validation set. Our model outperforms all other range-viewbased methods. To the best of our knowledge, we are the only open-source, multi-class range-view method. Not all methods report Cyclist performance and we're unable to compare FPS fairly since we do not have access to their code.

## 5 Conclusion

In this paper, we examine a diverse set of considerations when designing a range-view 3D object detection model. Surprisingly, we find that not all contributions from past literature yield meaningful performance improvements. We propose a straightforward dynamic 3D centerness technique which performs well across datasets, and a simple sub-sampling technique to improve range-view model runtime. These techniques allow us to establish the first range-view method on Argoverse 2, which is competitive with voxel-based methods, and a new state-of-the-art amongst range-view models on Waymo Open. Our results demonstrate that simple methods are at least as effective than recentlyproposed techniques, and that range-view models are a promising avenue for future research.

## 6 Supplementary Material

Our supplementary materials covers the following: background on 3D object detection in the range view, additional quantitative results, qualitiative results, dataset details, and implementation details for our models.

## 7 Range View Representation

The range view representation, also known as a range image, is a 2D grid containing the spherical coordinates of an observed point with respect to the lidar laser's original reference frame. We define a range image as:

r ≜ {(φ ij , θ ij , r ij ) : 1 ≤ i ≤ H; 1 ≤ j ≤ W },(2)

where (φ ij , θ ij , r ij ) are the inclination, azimuth, and range, and H, W are the height and width of the image. Importantly, the cells of a range image are not limited to containing only spherical coordinates. They may also contain auxillary sensor information such as a lidar's intensity.

### 7.1 3D Object Detection

Given a range image r, we construct a set of 3D object proposals which are ranked by a confidence score. Each proposal consists of a proposed location, size, orientation, and category. Let D represent are predictions from a network.

D ≜ d i ∈ R 8 K i=1 , where K ⊂ N,(3)

d i ≜ x ego i , y ego i , z ego i , l i , w i , h i , θ i , c i(4)

where x ego i , y ego i , z ego i are the coordinates of the object in the ego-vehicle reference frame, l i , w i , h i are the length, width, and height of the object, θ i is the counter-clockwise rotation about the vertical axis, and c i is the object likelihood. Similarly, we define the ground truth cuboids as:

G ≜ g i ∈ R 8 M i=1 , where M ⊂ N,(5)

g i ≜ x ego i , y ego i , z ego i , l i , w i , h i , θ i , q i ,(6)

where q i is a continuous value computed dynamically during training. For example, q i may be set to Dynamic 3D Centerness or IoU BEV . The detected objects, D are decoded as the same parameterization as G.

D ≜ d k ∈ R 8 : c 1 ≥ • • • ≥ c k K k=1 , where K ⊂ N,(7)

d k ≜ x ego k , y ego k , z ego k , l k , w k , h k , θ k .(8)

We seek to predict a continuous representation of the ground truth targets as:

D ≜ d k ∈ R 8 : c 1 ≥ • • • ≥ c k K k=1 , where K ⊂ N,(9)

g k ≜ x ego k , y ego k , z ego k , l k , w k , h k , θ k , c k ,(10)

where x ego k , y ego k , z ego k are the coordinates of the object in the ego-vehicle reference frame, l k , w k , h k are the length, width, and height of the object, θ k is the counter-clockwise rotation about the vertical axis, and c k is the object category likelihood.

3D Anchor Points in the Range View. To predict objects, we bias our predictions by the location of observed 3D points which are features of the projected pixels in a range image. For all the 3D points contained in a range image, we produce a detection d k .

Regression Targets. Following previous literature, we do not directly predict the object proposal representation in Section 7.1. Instead, we define the regression targets as the following:

T (P, G) = {t i (p i , g i ) ∈ R 8 } K i=1 , where K ∈ N, (11

) t i (p i , g i ) = {∆x i , ∆y i , ∆z i , log l i , log w i , log h i , sin θ i , cos θ i } ,(12)

where P and G are the sets of points in the range image and the ground truth cuboids in the 3D scene, ∆x i , ∆y i , ∆z i are the offsets from the point to the associated ground truth cuboid in the point-azimuth reference frame, log l i , log w i , log h i are the logarithmic length, width, and height of the object, respectively, and sin θ i , cos θ i are continuous representations of the object's heading θ i .

Classification Loss. Once all of the candidate foreground points have been ranked and assigned, each point needs to incur loss proportional to its regression quality. We use Varifocal loss [26] with a sigmoid-logit activation for our classification loss:

VFL(c i , q i ) = q i (-q i log(c i ) + (1 -q i ) log(1 -c i )) if q i > 0 -αc γ i log(1 -c i ) otherwise,(13)

where c i is classification likelihood and q i is 3D classification targets (e.g., Dynamic IoU BEV or Dynamic 3D Centerness). Our final classification loss for an entire 3D scene is:

L c = 1 M N j=1 |P j G | i=1 VFL(c j i , q j i ),(14)

where M is the total number of foreground points, N is the total number of objects in a scene, P j G is the set of 3D points which fall inside the j th ground truth cuboid, c j i is the likelihood from the network classification head, and q j i is the 3D classification target.

Regression Loss. We use an ℓ 1 regression loss to predict the regression residuals. The regression loss for an entire 3D scene is:

L r = 1 N N j=1 1 |P j G | |P j G | i=1 L1Loss(r j i , t j i ),(15)

where N is the total number of objects in a scene, P j G is the set of 3D points which fall inside the j th ground truth cuboid, r j i is the predicted cuboid parameters from the network, and t j i are the target residuals to be predicted. Total Loss. Our final loss is written as:

L = L c + L r(16)

### 7.2 Argoverse 2

Additional details on the evaluation metrics used in the Argoverse 2.

• Average Precision (AP): VOC-style computation with a true positive defined at 3D Euclidean distance averaged over 0.5 m, 1.0 m, 2.0 m, and 4.0 m.

• Average Translation Error (ATE): 3D Euclidean distance for true positives at 2 m.

• Average Scale Error (ASE): Pose-aligned 3D IoU for true positives at 2 m.

• Average Orientation Error (AOE): Smallest yaw angle between the ground truth and prediction for true positives at 2 m.

• Composite Detection Score (CDS): Weighted average between AP and the normalized true positive scores:

CDS = AP • x∈X 1 -x, where x ∈ {ATE unit , ASE unit , AOE unit } .(17)

We refer readers to Wilson et al. [6] for further details.

### 7.3 Waymo Open

Additional details on the evaluation metrics used in the Waymo Open are listed below.

1. 3D Mean Average Precision (mAP): VOC-style computation with a true positive defined by 3D IoU. The gravity-aligned-axis is fixed.

(a) Level 1 (L1): All ground truth cuboids with at least five lidar points within them. (b) Level 2 (L2): All ground cuboids with at least 1 point and additionally incorporates heading into its true positive criteria.

Following RangeDet [5], we report L1 results.

8 Range-view 3D Object Detection Baseline Model. Our baseline models are all multi-class and utilize the Deep Layer Aggregation (DLA) [32] architecture with an input feature dimensionality of 64. In our Argoverse 2 experiments, we incorporate five input features: x, y, z, range, and intensity, while for our Waymo experiments, we include six input features: x, y, z, range, intensity, and elongation. These inputs are then transformed to the backbone feature dimensionality of 64 using a single basic block. For post-processing, we use weighted non-maximum suppression (WNMS). All models are trained and evaluated using mixed-precision with BrainFloat16 [33]. Both models use a OneCycle scheduler with AdamW using a learning rate of 0.03 across four A40 gpus. All models in the ablations are trained for 5 epochs on a uniformly sub-sampled fifth of the training set.

State-of-the-art Comparison Model. We leverage the best performing and most general methods from our experiments for our state-of-the-art comparison for both the Argoverse 2 and Waymo Open dataset models. The Argoverse 2 and Waymo Open models use an input feature dimensionality of 256 and 128, respectively. Both models uses the Meta-Kernel and a 3D input encoding, Dynamic 3D Centerness for their classification supervision, and we use our proposed Range-Subsampling with range partitions of [0 -30 m), [30 m, 50 m), [50 m, ∞) with subsampling rates of 8, 2, 1, respectively. For both datasets, models are trained for 20 epochs. Table 8: Classification Supervision: Waymo Open. Evaluation metrics and errors using two different classification supervision methods on the Waymo Open validation set. Our results suggest that Dynamic 3D Centerness is a competitive alternative to IoUBEV, while being simpler.

### 8.1 Qualitative Results

We include qualitative results for both Argoverse 2 and Waymo Open shown in Figs. 6 and 7.

