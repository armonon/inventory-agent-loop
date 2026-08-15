"""API tests — endpoints, status codes, and CSV persistence."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Point storage at a throwaway CSV so tests never touch the real one."""
    csv_file = tmp_path / "products.csv"
    monkeypatch.setenv("INVENTORY_CSV", str(csv_file))

    from api import app as app_module
    from api import storage

    importlib.reload(storage)
    importlib.reload(app_module)

    with TestClient(app_module.app) as c:  # triggers startup → seed
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_inventory_returns_seeded_catalogue(client: TestClient) -> None:
    r = client.get("/inventory")
    assert r.status_code == 200
    products = r.json()
    assert len(products) == 10
    assert {"id", "name", "quantity", "unit"} == set(products[0].keys())


def test_post_creates_product_with_new_id(client: TestClient) -> None:
    r = client.post(
        "/inventory", json={"name": "Almond milk", "quantity": 12, "unit": "liters"}
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Almond milk"
    assert body["quantity"] == 12
    assert body["unit"] == "liters"
    assert body["id"] == 11  # seeded catalogue ends at 10


def test_post_rejects_duplicate_name(client: TestClient) -> None:
    r = client.post(
        "/inventory", json={"name": "Oat milk", "quantity": 5, "unit": "liters"}
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


def test_post_rejects_negative_quantity(client: TestClient) -> None:
    r = client.post(
        "/inventory", json={"name": "Bad", "quantity": -1, "unit": "units"}
    )
    assert r.status_code == 422


def test_patch_positive_delta_is_incoming_stock(client: TestClient) -> None:
    before = client.get("/inventory/by-name", params={"name": "Oat milk"}).json()
    r = client.patch(f"/inventory/{before['id']}", json={"delta": 30})
    assert r.status_code == 200
    assert r.json()["quantity"] == before["quantity"] + 30


def test_patch_negative_delta_is_outgoing_stock(client: TestClient) -> None:
    before = client.get("/inventory/by-name", params={"name": "Arabica beans"}).json()
    r = client.patch(f"/inventory/{before['id']}", json={"delta": -12})
    assert r.status_code == 200
    assert r.json()["quantity"] == before["quantity"] - 12


def test_patch_unknown_product_returns_404(client: TestClient) -> None:
    r = client.patch("/inventory/9999", json={"delta": 1})
    assert r.status_code == 404
    assert "9999" in r.json()["detail"]


def test_patch_below_zero_returns_409_with_message(client: TestClient) -> None:
    product = client.get("/inventory/by-name", params={"name": "Napkins"}).json()  # 5 packs
    r = client.patch(f"/inventory/{product['id']}", json={"delta": -100})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "Napkins" in detail and "would leave" in detail


def test_alerts_default_threshold_is_10(client: TestClient) -> None:
    r = client.get("/inventory/alerts")
    assert r.status_code == 200
    body = r.json()
    assert body["threshold"] == 10
    # Seeded low stock: Oat milk (8), Vanilla syrup (6), Napkins (5)
    names = {p["name"] for p in body["products"]}
    assert names == {"Oat milk", "Vanilla syrup", "Napkins"}
    assert body["count"] == 3


def test_alerts_threshold_is_configurable(client: TestClient) -> None:
    r = client.get("/inventory/alerts", params={"threshold": 25})
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["products"]}
    assert "Robusta beans" in names  # 22, under 25 but over 10


def test_by_name_lookup_is_case_insensitive(client: TestClient) -> None:
    r = client.get("/inventory/by-name", params={"name": "oAt MiLk"})
    assert r.status_code == 200
    assert r.json()["name"] == "Oat milk"


def test_by_name_unknown_returns_404_with_helpful_detail(client: TestClient) -> None:
    r = client.get("/inventory/by-name", params={"name": "Unobtainium"})
    assert r.status_code == 404
    # The message must steer the LLM toward a recovery action.
    assert "list_inventory" in r.json()["detail"]


def test_by_name_handles_slash_in_product_name(client: TestClient) -> None:
    """Regression: 'Half/Half' used to hit FastAPI's routing 404 because
    the name was a path segment. It is a query param now."""
    client.post(
        "/inventory", json={"name": "Half/Half", "quantity": 9, "unit": "liters"}
    )
    r = client.get("/inventory/by-name", params={"name": "Half/Half"})
    assert r.status_code == 200
    assert r.json()["name"] == "Half/Half"


def test_by_name_requires_a_name(client: TestClient) -> None:
    r = client.get("/inventory/by-name")
    assert r.status_code == 422


def test_products_persist_to_csv_and_survive_reload(
    client: TestClient, tmp_path: Path
) -> None:
    """Rubric: products are persisted in products.csv and survive a restart."""
    client.post(
        "/inventory", json={"name": "Chai concentrate", "quantity": 7, "unit": "liters"}
    )

    # The file on disk must already contain it — not just memory.
    csv_file = tmp_path / "products.csv"
    with csv_file.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert any(row["name"] == "Chai concentrate" for row in rows)

    # Simulate a server restart: fresh module + fresh client, same file.
    from api import app as app_module
    from api import storage

    importlib.reload(storage)
    importlib.reload(app_module)
    with TestClient(app_module.app) as fresh:
        names = {p["name"] for p in fresh.get("/inventory").json()}
    assert "Chai concentrate" in names
