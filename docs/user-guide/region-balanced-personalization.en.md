# Region-balanced personalization

This optional Anima workflow combines a single primary rectangle per image, a spatial-loss schedule that returns to
whole-image training, SID-style captions, existing DreamBooth-style regularization data, and an APT-inspired
overfitting controller. See the [Chinese guide](region-balanced-personalization.md) for the complete walkthrough and
the [ROCm example](../../examples/rocm/anima-yuemeng-region-balance.example.yaml) for all parameters.

Use `JoyCaption (local Ollama)` or `SID subject disentanglement JSON (Ollama JoyCaption)` on the Tag page. Both target
`http://localhost:11434/v1` and `llama-joycaption-beta-one-hf-llava`; change the model field if `ollama list` reports a
different local tag. Use a class such as `1girl` separately from the unique trigger token.

Draw one rectangle under Preprocess → Primary region. Enable `region_balance_enabled` and optionally `apt_enabled` in
the Loss group. Region emphasis is held through 45% of training, annealed with a cosine from 45% to 55%, and is exactly
off thereafter. APT adds a frozen-base reference forward per step and is therefore substantially slower.

This is an Anima rectified-flow engineering adaptation, not a bit-for-bit reproduction of the SDXL APT method. V1 does
not implement intermediate-representation stabilization or cross-attention alignment, and it is incompatible with
NaViT, Leap, and (for APT) InfoNoise.
