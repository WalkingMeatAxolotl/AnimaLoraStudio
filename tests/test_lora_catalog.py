"""统一 LoRA catalog service/API。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services import lora_catalog
from studio.services.projects import projects, versions


@pytest.fixture
def catalog_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")

    models = tmp_path / "models"
    external = tmp_path / "comfy" / "models" / "loras"
    secrets.update({
        "models": {"root": str(models)},
        "generate": {"lora_catalog_dirs": [str(external)]},
    })
    lora_catalog.clear_cache()
    yield {
        "db": dbfile,
        "models": models,
        "default": models / "loras",
        "external": external,
    }
    lora_catalog.clear_cache()


@pytest.fixture
def client(catalog_env: dict) -> TestClient:  # noqa: ARG001
    return TestClient(server.app)


def _write(path: Path, data: bytes = b"x", *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _create_project_checkpoint(
    catalog_env: dict, name: str = "hero_final.safetensors"
) -> tuple[dict, dict, Path]:
    with db.connection_for(catalog_env["db"]) as conn:
        project = projects.create_project(conn, title="Hero Project")
        version = versions.create_version(
            conn, project_id=project["id"], label="trained-v1"
        )
    output = versions.version_dir(
        project["id"], project["slug"], version["label"]
    ) / "output"
    return project, version, _write(output / name, b"project")


def test_generate_config_has_external_lora_catalog_dirs() -> None:
    cfg = secrets.GenerateConfig()
    assert cfg.lora_catalog_dirs == []
    cfg = secrets.GenerateConfig(lora_catalog_dirs=["D:/ComfyUI/models/loras"])
    assert cfg.lora_catalog_dirs == ["D:/ComfyUI/models/loras"]


def test_catalog_aggregates_project_default_and_recursive_external_sources(
    client: TestClient, catalog_env: dict
) -> None:
    project, version, project_path = _create_project_checkpoint(catalog_env)
    default_path = _write(
        catalog_env["default"] / "styles" / "watercolor.safetensors",
        b"studio",
    )
    external_path = _write(
        catalog_env["external"] / "characters" / "alice.SAFETENSORS",
        b"external",
    )
    _write(catalog_env["external"] / "characters" / "ignored.txt")

    response = client.get("/api/lora-catalog?refresh=true&limit=50")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 3
    by_name = {item["name"]: item for item in body["items"]}

    project_item = by_name[project_path.name]
    assert project_item["source_type"] == "project"
    assert project_item["project_id"] == project["id"]
    assert project_item["version_id"] == version["id"]
    assert project_item["project_title"] == "Hero Project"
    assert project_item["version_label"] == "trained-v1"
    assert project_item["kind"] == "final"
    assert project_item["size"] == len(b"project")

    default_item = by_name[default_path.name]
    assert default_item["source_type"] == "studio_models"
    assert default_item["relative_path"] == "styles/watercolor.safetensors"
    assert default_item["kind"] == "other"

    external_item = by_name[external_path.name]
    assert external_item["source_type"] == "external"
    assert external_item["relative_path"] == "characters/alice.SAFETENSORS"

    sources = {source["source_type"] for source in body["sources"]}
    assert sources == {"project", "studio_models", "external"}
    assert all(source["error"] is None for source in body["sources"])


def test_catalog_deduplicates_normalized_full_paths_but_not_basenames(
    client: TestClient, catalog_env: dict
) -> None:
    shared = _write(catalog_env["default"] / "same.safetensors")
    other = _write(catalog_env["external"] / "same.safetensors")
    cfg = secrets.load()
    cfg.generate.lora_catalog_dirs = [
        str(catalog_env["default"]),
        str(catalog_env["external"]),
        str(catalog_env["external"]),
    ]
    secrets.save(cfg)

    body = client.get("/api/lora-catalog?refresh=true&limit=50").json()
    same_items = [item for item in body["items"] if item["name"] == "same.safetensors"]
    assert {Path(item["path"]) for item in same_items} == {
        shared.resolve(),
        other.resolve(),
    }
    # 默认来源优先，因此重复配置为外部来源的 shared 不会再出现。
    assert next(
        item for item in same_items if Path(item["path"]) == shared.resolve()
    )["source_type"] == "studio_models"
    assert len([s for s in body["sources"] if s["source_type"] == "external"]) == 2


def test_inaccessible_external_source_is_isolated(
    client: TestClient, catalog_env: dict
) -> None:
    available = _write(catalog_env["default"] / "available.safetensors")
    missing = catalog_env["external"] / "missing"
    cfg = secrets.load()
    cfg.generate.lora_catalog_dirs = [str(missing)]
    secrets.save(cfg)

    response = client.get("/api/lora-catalog?refresh=true")
    assert response.status_code == 200
    body = response.json()
    assert [item["path"] for item in body["items"]] == [str(available.resolve())]
    failed = next(s for s in body["sources"] if s["source_type"] == "external")
    assert failed["item_count"] == 0
    assert "not accessible" in failed["error"]
    default = next(s for s in body["sources"] if s["source_type"] == "studio_models")
    assert default["error"] is None


def test_search_sort_source_filter_and_cursor_pagination(
    client: TestClient, catalog_env: dict
) -> None:
    _write(catalog_env["default"] / "nested" / "Alpha.safetensors", b"a", mtime=10)
    _write(catalog_env["default"] / "nested" / "beta.safetensors", b"bb", mtime=20)
    _write(catalog_env["default"] / "other" / "gamma.safetensors", b"ccc", mtime=30)
    _write(catalog_env["external"] / "delta.safetensors", b"dddd", mtime=40)

    first = client.get(
        "/api/lora-catalog",
        params={
            "refresh": "true",
            "source": "studio_models",
            "sort": "name",
            "order": "desc",
            "limit": 2,
        },
    ).json()
    assert first["total"] == 3
    assert [item["name"] for item in first["items"]] == [
        "gamma.safetensors", "beta.safetensors"
    ]
    assert first["cursor"] == 0
    assert first["next_cursor"] == 2

    second = client.get(
        "/api/lora-catalog",
        params={
            "source": "studio_models",
            "sort": "name",
            "order": "desc",
            "limit": 2,
            "cursor": first["next_cursor"],
        },
    ).json()
    assert [item["name"] for item in second["items"]] == ["Alpha.safetensors"]
    assert second["next_cursor"] is None

    searched = client.get(
        "/api/lora-catalog", params={"q": "NESTED", "limit": 50}
    ).json()
    assert {item["name"] for item in searched["items"]} == {
        "Alpha.safetensors", "beta.safetensors"
    }

    by_size = client.get(
        "/api/lora-catalog",
        params={"sort": "size", "order": "desc", "limit": 50},
    ).json()
    assert [item["name"] for item in by_size["items"]] == [
        "delta.safetensors",
        "gamma.safetensors",
        "beta.safetensors",
        "Alpha.safetensors",
    ]


def test_refresh_bypasses_ttl_cache(client: TestClient, catalog_env: dict) -> None:
    _write(catalog_env["default"] / "first.safetensors")
    first = client.get("/api/lora-catalog?refresh=true&limit=50").json()
    assert first["cached"] is False
    assert first["total"] == 1

    _write(catalog_env["default"] / "second.safetensors")
    cached = client.get("/api/lora-catalog?limit=50").json()
    assert cached["cached"] is True
    assert cached["total"] == 1
    assert cached["generated_at"] == first["generated_at"]

    refreshed = client.get("/api/lora-catalog?refresh=true&limit=50").json()
    assert refreshed["cached"] is False
    assert refreshed["total"] == 2
    assert refreshed["generated_at"] >= first["generated_at"]


def test_archived_projects_are_opt_in(client: TestClient, catalog_env: dict) -> None:
    project, _, path = _create_project_checkpoint(catalog_env, "archived_step10.safetensors")
    with db.connection_for(catalog_env["db"]) as conn:
        projects.set_archived(conn, project["id"], True)

    hidden = client.get("/api/lora-catalog?refresh=true&limit=50").json()
    assert path.name not in {item["name"] for item in hidden["items"]}
    assert not [s for s in hidden["sources"] if s["source_id"] == f"project:{project['id']}"]

    shown = client.get(
        "/api/lora-catalog?refresh=true&include_archived=true&limit=50"
    ).json()
    item = next(item for item in shown["items"] if item["name"] == path.name)
    assert item["project_archived"] is True
    assert item["kind"] == "step"
