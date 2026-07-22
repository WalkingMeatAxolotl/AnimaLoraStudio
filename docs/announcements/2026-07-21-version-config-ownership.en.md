---
date: 2026-07-21
tag: migration
title: Version configs no longer follow global settings
pin: true
version: "0.21.0"
---
As of v0.21.0, a saved training version config is never rewritten behind your back — not even when you hit "start training". **Most people don't have to do anything**: if you have never changed a model in global settings and never used the trigger word field in the training config, you can skip this.

**If you change the base model / text encoder / VAE in global settings**

Existing versions used to follow along; they no longer do. Open the version's training config — when a path differs from the current global value, a "restore default" link appears next to the field label, and one click realigns it. You can also use "choose model" next to the field to pick from what you have downloaded, or type the path yourself.

**If you relied on the trigger word field to reach your sample images**

That field is retired, and sample prompts no longer get the trigger word prepended. Write the trigger word at the start of the sample prompt instead. The trigger word on the Tagging page (prepended to every caption) is unaffected.
