"""Projects 域 endpoints（PR-6.5 起从 server.py 抽出，71 routes 分 5 子文件）。

按子域切：
    crud.py        commit 1  16 routes  projects + versions CRUD + activate +
                                        advance/skip-phase + lora_ckpts / state_ckpts
    exports.py     commit 2  6 routes   train.zip / bundle.zip / export-bundle /
                                        import-bundle (path/upload) / import-train
    ingestion.py   commit 3  ~13 routes download estimate/start/status, upload,
                                        preprocess start/status/files/duplicates-removed/
                                        crop workspace+start/files reset+restore/thumb
    curation.py    commit 4  ~10 routes files delete/list/thumb, curation
                                        get/copy/remove/folder, duplicates scan/apply
    training.py    commit 5  ~26 routes tag, captions, snapshots, reg, reg_ai,
                                        version_config, queue training, version_thumb,
                                        jobs latest

_shared.py — 跨子文件共用 helper：_project_payload / _publish_*_state /
_version_dir_or_404 / _project_and_version_or_404 / _version_train_dir_or_404 /
_reg_dir。
"""
