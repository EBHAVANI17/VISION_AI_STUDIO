---
title: Vision AI Studio
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Segmentation, captioning, translation and speech in one app
---

# Vision AI Studio

Upload an image and get, in one pass:

- **Semantic segmentation** with a color-mapped overlay, ranked class breakdown and mask
- **A natural-language caption** from BLIP
- **Translation** into Hindi, Tamil, Telugu, Kannada, Malayalam, French, Spanish or Chinese
- **Spoken audio** of the caption

Everything runs on CPU inside this Space. Images are processed in memory and are not stored.

Model weights are baked into the image, so the app starts serving immediately. The
translation model (NLLB-200, ~2.4 GB) downloads on the first non-English request unless
the image was built with `PREFETCH_TRANSLATOR=nllb`.

Source, architecture notes and evaluation scripts:
[github.com/EBHAVANI17/VISION_AI_STUDIO](https://github.com/EBHAVANI17/VISION_AI_STUDIO)
