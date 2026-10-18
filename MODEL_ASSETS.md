# Локальные ресурсы, не опубликованные в GitHub

Восстановите нужные файлы в прежних относительных путях:

- `outputs/forte_qwen_plus30/classifier30.joblib`
- `outputs/forte_qwen30/classifier30.joblib`
- `outputs/forte_qwen30/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`
- `outputs/forte_qwen30/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- `outputs/forte_transfer30/classifier30.joblib`
- `outputs/forte_transfer30/event_head30.joblib`
- `outputs/forte_transfer30/finetuned_tail30.pt`
- `outputs/forte_improved/all_classifier.joblib`
- `outputs/forte_improved/dinov2_vitb14_pretrain.pth`
- `outputs/forte_improved/existing_classifier.joblib`
- `outputs/forte_improved/finetuned_matched_tail.pt`
- `outputs/forte_improved/finetuned_stable_tail.pt`
- `outputs/forte_improved/finetuned_tail.pt`
- `outputs/forte_improved/window_classifier.joblib`
- `outputs/forte_model/r3d18_kinetics400.pth`
- `outputs/forte_model/resnet18_imagenet.pth`

Также требуются метаданные активного эксперимента (selection_locked.json, split.json, test_metrics.json, feature_spec.json), encoder_provenance.json и локальный vendor DINOv2; llama.cpp в outputs/forte_qwen/runtime; аннотации videos.json, events.json и taxonomy.json в outputs/forte_dataset/annotations. Демо требует локального X-026.mov. Эти ресурсы отсутствуют в публичной копии намеренно. Не публикуйте приватные видео и результаты.
