---
marp: true
theme: default
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
style: |
  section {
    font-size: 28px;
  }
  h1 {
    color: #D32F2F;
  }
  h2 {
    color: #C62828;
  }
  .limitation {
    background: #FFEBEE;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
  }
  .solution {
    background: #E8F5E9;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
  }
---

# RESEARCH LIMITATIONS
## Retinal AI Screening System

**Critical Analysis & Future Improvements**

---

# Overview

## Categories of Limitations

1. 🏗️ **Model Architecture Limitations**
2. 📊 **Data-Related Limitations**
3. 🔧 **System Architecture Limitations**
4. 📱 **Deployment Limitations**
5. 🔬 **Methodology Limitations**

---

# Model Architecture Limitations

---

## Limitation 1: Fixed Input Size

**Issue:**
- Model requires exactly 224×224 pixel images
- Original high-resolution fundus images are downscaled
- Loss of fine-grained details during resizing

**Impact:**
- May miss subtle pathological features
- Reduced ability to detect early-stage diseases

---

## Solution 1: Multi-Scale Processing

**Practical Approaches:**

✅ **Implement multi-resolution models**
- Accept various input sizes (224, 384, 512)
- Use adaptive pooling layers

✅ **Patch-based analysis**
- Process high-res images in overlapping patches
- Aggregate predictions from multiple patches

✅ **Progressive refinement**
- Initial screening at 224×224
- Detailed analysis at higher resolution for suspicious cases

---

## Limitation 2: Single Model Architecture

**Issue:**
- Relies solely on GraphCLIP architecture
- No ensemble of diverse models
- Susceptible to model-specific biases

**Impact:**
- May underperform on certain disease types
- Limited robustness to edge cases

---

## Solution 2: Ensemble Approach

**Practical Approaches:**

✅ **Multi-model ensemble**
- Combine GraphCLIP + ResNet + EfficientNet
- Voting or weighted averaging of predictions

✅ **Specialized sub-models**
- Train separate models for disease categories
- Route images to appropriate specialist models

✅ **Model versioning**
- Maintain multiple model versions
- A/B testing for continuous improvement

---

## Limitation 3: Computational Constraints

**Issue:**
- TFLite quantization reduces precision
- Float32 → Int8/Float16 conversion
- Trade-off between speed and accuracy

**Impact:**
- Potential accuracy drop (2-5%)
- May affect borderline cases

---

## Solution 3: Optimization Strategies

**Practical Approaches:**

✅ **Selective quantization**
- Keep critical layers in Float32
- Quantize only less sensitive layers

✅ **Hardware acceleration**
- Leverage GPU/NPU on mobile devices
- Use TFLite GPU delegate

✅ **Adaptive inference**
- Full precision for uncertain cases
- Quantized models for clear cases

---

# Data-Related Limitations

---

## Limitation 4: Limited Training Dataset

**Issue:**
- Model trained on specific datasets
- May not represent global population diversity
- Potential geographical/demographic biases

**Impact:**
- Reduced performance on underrepresented populations
- Generalization issues across different imaging equipment

---

## Solution 4: Data Diversity

**Practical Approaches:**

✅ **Expand data collection**
- Partner with hospitals worldwide
- Include diverse demographics (age, ethnicity, geography)

✅ **Transfer learning**
- Fine-tune on local clinical data
- Domain adaptation techniques

✅ **Synthetic data augmentation**
- Use GANs to generate diverse samples
- Advanced augmentation (color, contrast, artifacts)

---

## Limitation 5: Class Imbalance

**Issue:**
- Some diseases are rare (few training examples)
- Common conditions dominate the dataset
- Model biased toward frequent diseases

**Impact:**
- Poor detection of rare conditions
- False negatives for uncommon diseases

---

## Solution 5: Balanced Training

**Practical Approaches:**

✅ **Resampling techniques**
- Oversample minority classes
- Undersample majority classes
- SMOTE for synthetic minority examples

✅ **Loss function adjustment**
- Focal loss for hard examples
- Class-weighted loss functions

✅ **Two-stage detection**
- General screening → Specialized rare disease detection

---

## Limitation 6: Image Quality Dependency

**Issue:**
- Model sensitive to image quality
- Blur, poor lighting, artifacts affect performance
- No quality assessment before inference

**Impact:**
- Unreliable predictions on low-quality images
- User frustration with failed analyses

---

## Solution 6: Quality Control

**Practical Approaches:**

✅ **Pre-inference quality check**
- Blur detection algorithms
- Contrast and brightness validation
- Reject poor-quality images with feedback

✅ **Image enhancement pipeline**
- Automatic contrast adjustment
- Denoising algorithms
- Artifact removal

✅ **Quality-aware predictions**
- Confidence scores adjusted by image quality
- Explicit quality indicators in results

---

# System Architecture Limitations

---

## Limitation 7: Mobile Resource Constraints

**Issue:**
- Limited memory (150-200 MB peak usage)
- Battery consumption during inference
- Thermal throttling on repeated use

**Impact:**
- May crash on low-end devices
- Poor user experience on budget phones

---

## Solution 7: Resource Optimization

**Practical Approaches:**

✅ **Adaptive resource management**
- Detect device capabilities
- Adjust model complexity accordingly
- Fallback to cloud inference for low-end devices

✅ **Memory optimization**
- Lazy loading of resources
- Aggressive garbage collection
- Model pruning for lighter variants

✅ **Battery-aware inference**
- Batch processing when charging
- Low-power mode options

---

## Limitation 8: No Cloud Connectivity

**Issue:**
- Fully offline system (privacy-first)
- No model updates without app reinstall
- No remote monitoring or telemetry

**Impact:**
- Can't deploy bug fixes quickly
- No continuous improvement from usage data

---

## Solution 8: Hybrid Architecture

**Practical Approaches:**

✅ **Optional cloud sync**
- User-controlled data sharing
- Anonymous usage statistics
- OTA model updates

✅ **Federated learning**
- Train on-device without data sharing
- Aggregate improvements across users
- Privacy-preserving ML

✅ **Staged rollout**
- Beta testing channel
- Gradual model deployment
- Rollback capability

---

## Limitation 9: Single Platform Support

**Issue:**
- Android-only deployment
- No iOS support
- No web interface

**Impact:**
- Excludes iOS users (significant market)
- Limited accessibility

---

## Solution 9: Cross-Platform Deployment

**Practical Approaches:**

✅ **iOS development**
- Port to CoreML for iOS
- Maintain feature parity

✅ **Web application**
- WebAssembly + TensorFlow.js
- Progressive Web App (PWA)

✅ **Desktop clients**
- Electron-based desktop app
- Higher resolution analysis on PC

---

## Limitation 10: Lack of Real-Time Processing

**Issue:**
- Inference takes 350-700ms
- No video stream analysis
- Single image at a time

**Impact:**
- Can't analyze live fundus imaging
- Slower workflow in clinical settings

---

## Solution 10: Performance Enhancement

**Practical Approaches:**

✅ **Model optimization**
- Neural architecture search (NAS)
- Pruning and quantization
- Knowledge distillation

✅ **Pipeline optimization**
- Parallel preprocessing
- GPU acceleration
- Batch inference

✅ **Video processing**
- Frame selection algorithms
- Real-time lightweight models
- Progressive refinement

---

# Deployment Limitations

---

## Limitation 11: Large APK Size

**Issue:**
- 77.9 MB APK size
- Large download barrier
- Storage concerns on budget devices

**Impact:**
- User reluctance to install
- Update friction

---

## Solution 11: Size Reduction

**Practical Approaches:**

✅ **Dynamic feature delivery**
- Download model on first use
- Google Play Asset Delivery

✅ **Model compression**
- Further quantization (Int8)
- Weight sharing
- Smaller architecture variants

✅ **Multiple APK variants**
- Different models for different devices
- App bundles for targeted delivery

---

## Limitation 12: No Offline Updates

**Issue:**
- Model embedded in APK
- Requires full app update for model changes
- No incremental improvements

**Impact:**
- Slow deployment of improvements
- All-or-nothing updates

---

## Solution 12: Dynamic Model Loading

**Practical Approaches:**

✅ **Separate model downloads**
- Models as separate assets
- In-app model updates
- Version management

✅ **A/B testing framework**
- Multiple model versions
- User-specific routing
- Performance comparison

✅ **Incremental updates**
- Delta updates for models
- Background downloading

---

## Limitation 13: Limited Error Recovery

**Issue:**
- Basic error handling
- Generic error messages
- No automatic retry mechanisms

**Impact:**
- Poor user experience on failures
- Difficult troubleshooting

---

## Solution 13: Robust Error Handling

**Practical Approaches:**

✅ **Detailed error categorization**
- Specific error codes
- Actionable user messages
- Automatic recovery attempts

✅ **Fallback mechanisms**
- Multiple inference paths
- Graceful degradation
- Offline queue for later processing

✅ **Error reporting**
- Anonymous crash reports
- Debug logs for support
- User feedback integration

---

# Methodology Limitations

---

## Limitation 14: No Clinical Validation

**Issue:**
- Not tested in real clinical settings
- No FDA/CE approval
- Not validated by ophthalmologists

**Impact:**
- Cannot be used for diagnosis
- Legal and ethical concerns

---

## Solution 14: Clinical Trials

**Practical Approaches:**

✅ **Clinical validation study**
- Partner with hospitals
- Ophthalmologist review of predictions
- Compare against gold standard diagnosis

✅ **Regulatory approval process**
- FDA 510(k) pathway
- CE marking in Europe
- Document validation results

✅ **Clinical decision support**
- Position as screening tool, not diagnostic
- Explicit disclaimers
- Integration with clinical workflow

---

## Limitation 15: No Explainability

**Issue:**
- Black-box predictions
- No visualization of decision areas
- Cannot explain why a disease was detected

**Impact:**
- Lack of clinical trust
- Difficult to verify predictions
- No educational value

---

## Solution 15: Interpretability Features

**Practical Approaches:**

✅ **Grad-CAM visualization**
- Highlight regions influencing prediction
- Heat maps overlay on fundus images
- Show attention areas

✅ **Feature importance**
- Explain which features matter
- Link to clinical markers

✅ **Confidence breakdown**
- Show uncertainty sources
- Per-class confidence intervals

---

## Limitation 16: No Longitudinal Tracking

**Issue:**
- Single-image analysis only
- No disease progression monitoring
- No patient history integration

**Impact:**
- Miss temporal patterns
- Cannot track treatment effectiveness

---

## Solution 16: Temporal Analysis

**Practical Approaches:**

✅ **Image history database**
- Store previous analyses (with consent)
- Compare current vs. previous scans
- Progression alerts

✅ **Trend analysis**
- Graph disease markers over time
- Predict future progression

✅ **EHR integration**
- Link with patient medical records
- Context-aware predictions

---

## Limitation 17: Binary Decision Focus

**Issue:**
- Present/absent classification
- No severity staging for most conditions
- Limited grading information

**Impact:**
- Insufficient detail for treatment planning
- Cannot prioritize urgent cases

---

## Solution 17: Fine-Grained Classification

**Practical Approaches:**

✅ **Multi-stage models**
- Stage 1: Disease presence
- Stage 2: Severity grading
- Stage 3: Sub-type classification

✅ **Continuous scoring**
- Severity scores (0-10)
- Risk stratification

✅ **Clinical grading**
- ETDRS scale for DR
- AREDS for AMD
- Standard clinical scales

---

## Limitation 18: Single Eye Analysis

**Issue:**
- Analyzes one image at a time
- No comparison between left/right eyes
- Misses bilateral patterns

**Impact:**
- Incomplete assessment
- May miss systemic conditions

---

## Solution 18: Bilateral Analysis

**Practical Approaches:**

✅ **Dual-eye comparison**
- Accept two images (OD + OS)
- Cross-eye analysis
- Detect asymmetries

✅ **Systemic condition detection**
- Identify bilateral patterns
- Flag systemic diseases (diabetes, hypertension)

✅ **Comprehensive reports**
- Combined assessment
- Comparison metrics

---

# Additional Limitations

---

## Limitation 19: Language Barrier

**Issue:**
- English-only interface
- Disease names in English
- Limited accessibility globally

**Impact:**
- Excludes non-English speakers
- Reduced adoption in non-English regions

---

## Solution 19: Internationalization

**Practical Approaches:**

✅ **Multi-language support**
- i18n framework implementation
- Translate UI and disease names
- Local medical terminology

✅ **Voice interface**
- Speech-to-text for queries
- Text-to-speech for results
- Accessibility features

✅ **Cultural adaptation**
- Region-specific recommendations
- Local clinical guidelines

---

## Limitation 20: No Integration with Diagnostic Devices

**Issue:**
- Manual image upload required
- No direct connection to fundus cameras
- Workflow disruption

**Impact:**
- Extra steps in clinical workflow
- Potential data loss or corruption

---

## Solution 20: Device Integration

**Practical Approaches:**

✅ **DICOM support**
- Direct import from medical imaging systems
- Standard medical image format

✅ **API for devices**
- SDK for fundus camera manufacturers
- Automatic image transfer

✅ **Workflow automation**
- Batch processing
- Integration with PACS systems

---

# Summary of Key Limitations

---

## Critical Limitations

| Category | Count | Priority |
|----------|-------|----------|
| **Model Architecture** | 3 | 🔴 High |
| **Data Quality** | 3 | 🔴 High |
| **System Design** | 4 | 🟡 Medium |
| **Deployment** | 3 | 🟡 Medium |
| **Methodology** | 5 | 🔴 High |
| **Integration** | 2 | 🟢 Low |

**Total Limitations Identified:** 20

---

## Prioritization Framework

### 🔴 High Priority (Address First)
- Clinical validation
- Model explainability
- Data diversity
- Class imbalance

### 🟡 Medium Priority (Next Phase)
- Multi-platform support
- Model updates mechanism
- Resource optimization
- Quality control

### 🟢 Low Priority (Future Work)
- Device integration
- Internationalization
- Video processing
- Bilateral analysis

---

# Implementation Roadmap

---

## Phase 1: Foundation (Months 1-3)

**Focus:** Critical limitations

✅ Clinical validation study
✅ Expand training dataset
✅ Implement explainability (Grad-CAM)
✅ Quality control pipeline
✅ Address class imbalance

**Expected Impact:** 
- Increased accuracy by 5-10%
- Clinical trust improvement
- Regulatory pathway initiation

---

## Phase 2: Enhancement (Months 4-6)

**Focus:** User experience

✅ iOS app development
✅ Multi-scale processing
✅ Error handling improvements
✅ Dynamic model updates
✅ Multi-language support

**Expected Impact:**
- Broader user base
- Better reliability
- Improved UX metrics

---

## Phase 3: Advanced Features (Months 7-12)

**Focus:** Clinical integration

✅ EHR integration
✅ Longitudinal tracking
✅ Device connectivity (DICOM)
✅ Ensemble models
✅ Real-time processing

**Expected Impact:**
- Clinical workflow integration
- Professional adoption
- Enterprise readiness

---

# Risk Mitigation

---

## Technical Risks

**Risk 1:** Performance degradation with improvements
**Mitigation:** A/B testing, staged rollout

**Risk 2:** Regulatory approval delays
**Mitigation:** Parallel development, positioning as screening tool

**Risk 3:** Data privacy concerns
**Mitigation:** Privacy-by-design, transparent policies

**Risk 4:** Computational cost increases
**Mitigation:** Cloud-based heavy processing option

---

# Research Gaps

---

## Areas Requiring Further Investigation

🔬 **Transfer learning effectiveness**
- Cross-population generalization
- Domain adaptation strategies

🔬 **Optimal model compression**
- Accuracy vs. size trade-offs
- Hardware-specific optimization

🔬 **Uncertainty quantification**
- Bayesian approaches
- Ensemble diversity metrics

🔬 **Clinical utility**
- Cost-benefit analysis
- Impact on patient outcomes

---

# Ethical Considerations

---

## Limitations in Ethics

**Issue 1:** Potential for misdiagnosis
**Address:** Clear disclaimers, professional oversight requirement

**Issue 2:** Access inequality
**Address:** Free/low-cost options, offline capability

**Issue 3:** Algorithmic bias
**Address:** Regular fairness audits, diverse datasets

**Issue 4:** Data privacy
**Address:** On-device processing, minimal data collection

---

# Comparative Analysis

---

## Our System vs. Alternatives

| Aspect | Our System | Cloud-Based | Traditional |
|--------|------------|-------------|-------------|
| **Privacy** | ✅ High | ❌ Low | ✅ High |
| **Speed** | ✅ Fast | ⚠️ Variable | ❌ Slow |
| **Cost** | ✅ Free | ❌ Per-use | ❌ Expensive |
| **Accuracy** | ⚠️ 85-92% | ✅ 90-95% | ✅ 95%+ |
| **Accessibility** | ✅ High | ⚠️ Medium | ❌ Low |
| **Updates** | ❌ Manual | ✅ Automatic | N/A |

---

# Lessons Learned

---

## Key Insights

💡 **Mobile ML is feasible** but requires careful optimization

💡 **Privacy and performance** are often trade-offs

💡 **Clinical validation** is essential for medical AI

💡 **User experience** matters as much as accuracy

💡 **Continuous improvement** requires infrastructure planning

💡 **Documentation** helps identify and address limitations

---

# Future Research Directions

---

## Promising Avenues

🔮 **Multi-modal fusion**
- Combine fundus + OCT + patient data
- Holistic assessment

🔮 **Federated learning at scale**
- Privacy-preserving collaborative improvement
- Global model training

🔮 **Causal inference**
- Move beyond correlation
- Understand disease mechanisms

🔮 **Active learning**
- Smart data collection
- Efficient labeling strategies

---

# Acknowledgment of Limitations

---

## Transparent Research

✅ **Openly documented** all known limitations

✅ **Practical solutions proposed** for each issue

✅ **Prioritization framework** established

✅ **Roadmap created** for systematic improvements

✅ **Continuous monitoring** commitment

**Research integrity through honest limitation assessment**

---

# Conclusion

---

## Summary

📋 **20 limitations identified** across 5 categories

💡 **Practical solutions proposed** for each

🗺️ **Clear roadmap** for addressing issues

🎯 **Prioritized action plan** for implementation

🔬 **Foundation for future research** established

---

## Final Thoughts

**Every limitation is an opportunity for improvement**

The identified limitations don't diminish the system's value—
they provide a **clear path forward** for making it better.

**Transparency about limitations = Research integrity**

---

# Questions & Discussion

## Open Dialogue

**Which limitations concern you most?**

**What solutions seem most feasible?**

**What did we miss?**

---

# References & Further Reading

## Recommended Resources

📚 **Mobile ML Optimization**
- TensorFlow Lite Best Practices
- Model Compression Techniques

📚 **Clinical AI Validation**
- FDA Guidelines for Medical AI
- Clinical Trial Design for ML

📚 **Bias and Fairness**
- Algorithmic Fairness in Healthcare
- Dataset Diversity Standards

---

# Thank You!

## Retinal AI Screening System
### Limitations & Future Work

**Honest Assessment + Clear Path Forward = Better Research**

---

# Appendix: Detailed Metrics

---

## Current Performance Baseline

**Accuracy Metrics:**
- Overall: 87.3%
- Top-5: 96.1%
- Sensitivity: 84.2%
- Specificity: 91.5%

**Performance Metrics:**
- Inference: 350ms avg
- Memory: 180MB peak
- Battery: 2.1% per analysis

**Baseline for measuring improvements**

---

## Target Improvements

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| **Accuracy** | 87% | 92% | 6 months |
| **Speed** | 350ms | 200ms | 3 months |
| **APK Size** | 78MB | 50MB | 6 months |
| **Languages** | 1 | 5 | 6 months |
| **Platforms** | 1 | 3 | 12 months |

---

## Success Criteria

**Technical Success:**
✅ 5% accuracy improvement
✅ 30% speed improvement
✅ Clinical validation completion

**User Success:**
✅ 10,000+ active users
✅ 4.5+ star rating
✅ 90%+ satisfaction score

**Clinical Success:**
✅ Published validation study
✅ Hospital partnerships
✅ Regulatory approval pathway

---

# End of Presentation

**Contact Information**
**Project Repository:** https://github.com/mpairwe7/MLOPS_V1
