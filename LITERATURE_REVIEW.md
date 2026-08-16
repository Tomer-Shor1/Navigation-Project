# Literature Review — Visual Navigation for Low‑Flying Drones (GNSS‑Denied, 20–200 m)

**Scope.** This review surveys the algorithms and open‑source tools most widely
used today for *visual navigation of low‑flying drones* (roughly 20–200 m AGL)
when GNSS is unavailable, with an emphasis on **"papers‑with‑code"** as the
assignment requests. It closes by mapping *this project* onto that landscape and
naming the published method it most resembles.

> Note on citations: entries below give the canonical author/venue/year and the
> public code repository where one exists. Verify links against the original
> sources before formal submission.

---

## 1. The problem and a taxonomy

"Visual navigation without GNSS" is not one problem but a family, and it helps to
separate them along three axes:

- **Relative vs. absolute.** *Relative* methods (visual odometry / SLAM) track
  how the drone moves from a starting point but drift over time; *absolute*
  methods recover a global coordinate by matching the live view to a
  georeferenced reference (prior imagery, orthophoto, or GIS/satellite map).
- **Geometric feature matching vs. image retrieval.** *Feature matching* finds
  pixel‑level correspondences between two images and solves for geometry;
  *retrieval / place recognition* asks "which reference image (tile) is this?"
  using a global descriptor, usually as a coarse first stage.
- **Classical (hand‑crafted) vs. learned.** Classical detectors/descriptors
  (SIFT, ORB) vs. deep detectors/matchers (SuperPoint, SuperGlue, LoFTR).

The problem this assignment targets — *given a georeferenced reference of an area,
localize a new GNSS‑less frame and report the ground coordinate under the video
center* — is **absolute, feature‑matching‑based localization**, optionally with a
retrieval front‑end and a motion/inertial filter for real‑time robustness. The
five families below cover the tools in current use.

---

## 2. Family A — Classical local features + geometric verification *(most generic; our approach)*

The workhorse of image matching for two decades: detect repeatable keypoints,
describe them, match descriptors, then filter with a robust geometric model.

- **SIFT** — Lowe, *Distinctive Image Features from Scale‑Invariant Keypoints*,
  IJCV 2004. The reference detector/descriptor; scale/rotation invariant,
  accurate, slower. Introduced the **ratio test** still used everywhere.
- **SURF** — Bay et al., *SURF: Speeded‑Up Robust Features*, ECCV 2006. Faster
  SIFT approximation.
- **ORB** — Rublee, Rabaud, Konolige, Bradski, *ORB: an efficient alternative to
  SIFT or SURF*, ICCV 2011. FAST corners + oriented BRIEF binary descriptor;
  extremely fast, patent‑free, matched by cheap Hamming distance. The default in
  OpenCV and in most lightweight/real‑time UAV pipelines. **(Used in this
  project.)**
- **AKAZE / KAZE** — Alcantarilla et al., ECCV 2012 / BMVC 2013. Nonlinear
  scale‑space features; often more robust than ORB at moderate extra cost.
- **RANSAC** — Fischler & Bolles, CACM 1981, and successors (PROSAC, MAGSAC++,
  Barath 2020). The standard robust estimator used to fit a homography/essential
  matrix and reject outlier matches. Available directly in OpenCV
  (`findHomography`, `findEssentialMat`).

**Strengths for 20–200 m drones:** light, fast, CPU‑only, no training data,
easy to build — ideal for onboard/real‑time and for same‑season, similar‑viewpoint
matching (e.g. drone‑to‑drone from the same or a recent flight). **Weaknesses:**
degrade under large viewpoint change (oblique drone vs. nadir satellite),
seasonal/illumination change, and low‑texture terrain (water, fields). Tooling:
**OpenCV** (`cv2`) provides all of the above out of the box.

*Representative UAV use:* Luo et al., *Fast Automatic Registration of UAV Images
via Bidirectional Matching*, Sensors 2023 — ORB + Hamming + a RANSAC‑family
filter, plus a forward/backward (mutual) consistency check.

---

## 3. Family B — Learned local features and matchers *(state of the art for hard matching)*

Deep networks now dominate benchmarks when the two images differ strongly in
viewpoint, illumination, or modality — exactly the regime where ORB fails.

- **SuperPoint** — DeTone, Malisiewicz, Rabinovich, CVPRW 2018. Self‑supervised
  joint keypoint + descriptor. Code: `magicleap/SuperPointPretrainedNetwork`.
- **SuperGlue** — Sarlin, DeTone, Malisiewicz, Rabinovich, CVPR 2020. A graph
  neural network that matches two sets of features using attention + optimal
  transport; a major robustness jump. Code:
  `magicleap/SuperGluePretrainedNetwork`.
- **LoFTR** — Sun, Shen, Wang, Bao, Zhou, CVPR 2021. *Detector‑free* transformer
  matching that produces dense correspondences even in low‑texture regions. Code:
  `zju3dv/LoFTR`.
- **LightGlue** — Lindenberger, Sarlin, Pollefeys, ICCV 2023. An efficient,
  adaptive re‑design of SuperGlue for real‑time use. Code: `cvg/LightGlue`.
- **DISK / R2D2 / ALIKED** — learned dense/keypoint descriptors frequently paired
  with the matchers above.

**Relevance:** these are the recommended drop‑in when moving from same‑flight
references to **cross‑view / satellite / GIS reference imagery** (the final
project). They cost a GPU and model weights, which is why lightweight pipelines
still start with ORB.

---

## 4. Family C — Absolute localization against pre‑existing maps / orthophotos *(the exact problem)*

Methods whose goal is identical to this assignment: replace GNSS by registering a
live UAV frame to a *pre‑existing georeferenced* map.

- **Goforth & Lucey**, *GPS‑Denied UAV Localization using Pre‑existing Satellite
  Imagery*, ICRA 2019. Fine‑tunes a CNN (VGG‑style) to register UAV frames to
  satellite tiles, precisely because hand‑crafted features degrade across the
  UAV↔satellite gap. **Same top‑level problem as this project; they use learned
  features where we use ORB.** Code: `hmgoforth/gps-denied-uav-localization`.
- **Kinnari, Verdoja & Kyrki**, *GNSS‑denied geolocalization of UAVs by visual
  matching of onboard camera images with orthophotos*, ICAR 2021 (+ journal
  extension). **Orthorectifies** the UAV image under a local‑planarity assumption
  before matching — directly addressing the oblique‑viewpoint distortion that our
  single‑GSD model only approximates.
- **Bianchi & Barfoot**, *UAV Localization Using Autoencoded Satellite Images*,
  IEEE RA‑L 2021. Learns a compact embedding of satellite imagery for fast
  matching/retrieval.
- **Survey:** Couturier & Akhloufi, *A review on absolute visual localization for
  UAV*, Robotics and Autonomous Systems 2021 — the best single entry point to
  this sub‑field and its taxonomy (map‑matching, feature‑based, learning‑based).

**Takeaway:** the field's trajectory is classical‑features → learned‑features/
orthorectification to survive the viewpoint and appearance gap between drone and
map. This project sits at the classical end and documents that gap as future work.

---

## 5. Family D — Cross‑view geo‑localization & Visual Place Recognition (VPR)

The *retrieval* view of the problem: given a query image and a database of
georeferenced images/tiles, find the matching tile (coarse position), often as a
first stage before geometric refinement. Central to "which part of the map am I
over?".

- **Benchmark:** Zheng, Wei & Yang, *University‑1652: A Multi‑view Multi‑source
  Benchmark for Drone‑based Geo‑localization*, ACM MM 2020 — the standard
  drone/satellite/ground retrieval benchmark. Code:
  `layumi/University1652-Baseline`.
- **NetVLAD** — Arandjelović et al., CVPR 2016. The seminal learned global
  descriptor for place recognition; **Patch‑NetVLAD** (Hausler et al., CVPR 2021)
  adds local re‑ranking.
- **CosPlace** (Berton et al., CVPR 2022), **MixVPR** (Ali‑bey et al., WACV 2023),
  **EigenPlaces** (Berton et al., ICCV 2023) — current strong, efficient VPR
  descriptors.
- **Cross‑view transformers:** TransGeo (Zhu et al., CVPR 2022) and Sample4Geo
  (Deuser et al., ICCV 2023) target the drone↔satellite retrieval gap directly.

**Relevance:** a VPR front‑end is the natural way to *initialize* localization
("global re‑localization") when there is no start position, complementing the
geometric matching used for fine position. In this project, the spatial motion
gate plays a lightweight, non‑learned version of the same "narrow the candidates"
role.

---

## 6. Family E — Visual Odometry / Visual‑Inertial Odometry / SLAM (relative motion + fusion)

Relative‑motion estimators that track pose frame‑to‑frame and (in SLAM) build a
map with loop closure. They drift without an absolute reference, so in GNSS‑denied
navigation they are typically *fused* with map‑matching (Family C/D).

- **ORB‑SLAM2/3** — Mur‑Artal & Tardós, T‑RO 2017; Campos et al., T‑RO 2021.
  Feature‑based visual(‑inertial) SLAM; the most cited open SLAM system. Code:
  `UZ-SLAMLab/ORB_SLAM3`.
- **VINS‑Mono / VINS‑Fusion** — Qin, Li & Shen, T‑RO 2018. Robust monocular
  visual‑inertial odometry widely used on drones. Code:
  `HKUST-Aerial-Robotics/VINS-Mono`.
- **OpenVINS** — Geneva et al., ICRA 2020. A filter‑based (MSCKF) VIO framework.
- **DSO / SVO** — direct and semi‑direct alternatives to feature‑based VO.

**Relevance to us:** our **motion model + dead reckoning** is a deliberately
minimal stand‑in for VIO — it *predicts* between absolute fixes and *coasts*
through gaps. Crucially, we measured that true visual odometry did **not** work on
our 1 Hz oblique footage (near‑zero correlation with GPS displacement) and that
constant‑velocity coasting diverges without an IMU — which is exactly why
production systems use a real VIO/IMU here, and a strong argument for adding one.
The **loosely‑coupled fusion** of VO/VIO with absolute map matching (predict with
VIO, correct with map fixes) is the standard architecture our loop imitates.

---

## 7. Where this project sits (and its closest published analog)

Mapping our system onto the taxonomy:

| Component in this project | Family | Method used |
|---|---|---|
| Feature detection/description | A (classical) | **ORB** (FAST + BRIEF), OpenCV |
| Matching + outlier rejection | A | BF/Hamming + **Lowe ratio test** + **RANSAC homography** |
| Confidence / acceptance | A | **geometric inlier count/ratio** |
| Reference = prior georeferenced imagery | C (absolute) | same‑flight georeferenced frames |
| Candidate narrowing ("which tile") | D (retrieval, lightweight) | **motion‑model spatial gate** (not learned) |
| Between/through fixes | E (VO/VIO) | **constant‑velocity motion model + capped dead reckoning** |
| Pixel → world | — | GSD (slant range from barometric altitude + FOV) + heading + UTM |

**Closest published method:** **Goforth & Lucey (ICRA 2019)** — same top‑level
goal (replace GNSS by matching a live UAV frame to pre‑existing georeferenced
imagery) and the same core pipeline shape (feature registration → geometric
transform → geocoordinate). The essential difference is the feature front‑end:
**they learn CNN features to survive the UAV↔satellite gap; we use classical ORB
because our reference is same‑flight, similar‑viewpoint imagery.** **Kinnari et
al. (2021)** is the second‑closest and addresses our largest unmodeled error
(oblique viewpoint) via orthorectification. Architecturally, our
predict‑gate‑match‑correct loop with an innovation gate is a lightweight instance
of the **loosely‑coupled VIO + map‑matching** design from Family E.

In short: this project is a **classical‑feature, absolute visual‑localization
system with a loosely‑coupled motion filter** — the generic, well‑established
baseline that the learned methods in Families B–D improve upon.

---

## 8. Open challenges and directions relevant to this project

Drawn directly from the gaps the families above expose:

1. **Cross‑view robustness** (Family B/C): replace ORB with SuperPoint+SuperGlue,
   LoFTR, or LightGlue to match oblique drone frames against nadir Google‑Earth /
   GIS tiles — the single biggest change needed for the final project.
2. **Orthorectification** (Kinnari 2021): warp the oblique frame to nadir before
   matching, instead of correcting only the average scale with one GSD.
3. **Global re‑localization** (Family D): a VPR front‑end (NetVLAD/CosPlace/MixVPR)
   to initialize without a known start position.
4. **True inertial fusion** (Family E): add IMU/VIO so dead reckoning between fixes
   is metric and drift‑bounded, rather than the capped constant‑velocity proxy we
   were forced into by the absence of IMU data.
5. **Robust estimators**: MAGSAC++ over vanilla RANSAC; mutual/bidirectional match
   consistency (Luo 2023) for cleaner correspondences.

---

## 9. References (selected, papers‑with‑code emphasized)

**Classical features & geometry**
- Lowe. *Distinctive Image Features from Scale‑Invariant Keypoints.* IJCV 2004.
- Bay, Tuytelaars, Van Gool. *SURF.* ECCV 2006.
- Rublee, Rabaud, Konolige, Bradski. *ORB.* ICCV 2011.
- Alcantarilla, Bartoli, Davison. *KAZE Features.* ECCV 2012. (AKAZE, BMVC 2013.)
- Fischler, Bolles. *Random Sample Consensus.* CACM 1981. (MAGSAC++, Barath et al., CVPR 2020.)
- Luo et al. *Fast Automatic Registration of UAV Images via Bidirectional Matching.* Sensors 2023.

**Learned features & matchers**
- DeTone, Malisiewicz, Rabinovich. *SuperPoint.* CVPRW 2018. — magicleap/SuperPointPretrainedNetwork
- Sarlin et al. *SuperGlue.* CVPR 2020. — magicleap/SuperGluePretrainedNetwork
- Sun et al. *LoFTR.* CVPR 2021. — zju3dv/LoFTR
- Lindenberger, Sarlin, Pollefeys. *LightGlue.* ICCV 2023. — cvg/LightGlue

**Absolute localization vs. maps/orthophotos**
- Goforth, Lucey. *GPS‑Denied UAV Localization using Pre‑existing Satellite Imagery.* ICRA 2019. — hmgoforth/gps-denied-uav-localization
- Kinnari, Verdoja, Kyrki. *GNSS‑denied geolocalization of UAVs by visual matching with orthophotos.* ICAR 2021.
- Bianchi, Barfoot. *UAV Localization Using Autoencoded Satellite Images.* IEEE RA‑L 2021.
- Couturier, Akhloufi. *A review on absolute visual localization for UAV.* Robotics and Autonomous Systems 2021.

**Cross‑view geo‑localization & VPR**
- Zheng, Wei, Yang. *University‑1652.* ACM MM 2020. — layumi/University1652-Baseline
- Arandjelović et al. *NetVLAD.* CVPR 2016. (Patch‑NetVLAD, Hausler et al., CVPR 2021.)
- Berton, Masone, Caputo. *CosPlace.* CVPR 2022. (EigenPlaces, ICCV 2023.)
- Ali‑bey, Chaib‑draa, Giguère. *MixVPR.* WACV 2023.
- Zhu et al. *TransGeo.* CVPR 2022. Deuser et al. *Sample4Geo.* ICCV 2023.

**VO / VIO / SLAM**
- Mur‑Artal, Tardós. *ORB‑SLAM2.* IEEE T‑RO 2017. Campos et al. *ORB‑SLAM3.* T‑RO 2021. — UZ-SLAMLab/ORB_SLAM3
- Qin, Li, Shen. *VINS‑Mono.* IEEE T‑RO 2018. — HKUST-Aerial-Robotics/VINS-Mono
- Geneva et al. *OpenVINS.* ICRA 2020.
