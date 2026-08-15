"""CSV-backed product storage.

Products live in `products.csv` at the repo root so the data survives
a server restart (rubric requirement). Every mutation rewrites the
file immediately — no in-memory-only state, no write-on-shutdown
that could be lost to a Ctrl+C.

Kept separate from `app.py` so the persistence rules can be unit
tested without spinning up FastAPI.
"""

from __future__ import annotations

import csv
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

# Repo root = parent of the `api/` package.
_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "products.csv"

FIELDNAMES = ("id", "name", "quantity", "unit")

# Serialises read-modify-write cycles so two concurrent PATCHes can't
# clobber each other's quantity update.
_lock = threading.Lock()


@dataclass
class Product:
    id: int
    name: str
    quantity: float
    unit: str

    def to_dict(self) -> dict:
        return asdict(self)


def csv_path() -> Path:
    """Resolved at call time so tests can point INVENTORY_CSV elsewhere."""
    override = os.environ.get("INVENTORY_CSV")
    return Path(override) if override else _DEFAULT_CSV


def _read_raw() -> list[Product]:
    path = csv_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    products: list[Product] = []
    for row in rows:
        # Skip malformed lines rather than crashing the whole API.
        try:
            products.append(
                Product(
                    id=int(row["id"]),
                    name=str(row["name"]).strip(),
                    quantity=float(row["quantity"]),
                    unit=str(row["unit"]).strip(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return products


def _write_raw(products: list[Product]) -> None:
    path = csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for product in products:
            writer.writerow(product.to_dict())


def list_products() -> list[Product]:
    with _lock:
        return _read_raw()


def get_product(product_id: int) -> Product | None:
    with _lock:
        for product in _read_raw():
            if product.id == product_id:
                return product
    return None


def find_by_name(name: str) -> Product | None:
    """Case-insensitive exact-name lookup.

    The agent needs this because Carla says "oat milk", not "id 4".
    Exposed through the API so the LLM never has to guess an id.
    """
    needle = name.strip().lower()
    with _lock:
        for product in _read_raw():
            if product.name.strip().lower() == needle:
                return product
    return None


def add_product(name: str, quantity: float, unit: str) -> Product:
    with _lock:
        products = _read_raw()
        next_id = max((p.id for p in products), default=0) + 1
        product = Product(
            id=next_id,
            name=name.strip(),
            quantity=quantity,
            unit=unit.strip(),
        )
        products.append(product)
        _write_raw(products)
        return product


def apply_delta(product_id: int, delta: float) -> Product:
    """Add `delta` to a product's quantity and persist.

    Raises KeyError when the product doesn't exist, ValueError when
    the change would drive stock negative. Both surface as distinct
    HTTP status codes in app.py.
    """
    with _lock:
        products = _read_raw()
        for product in products:
            if product.id == product_id:
                new_quantity = product.quantity + delta
                if new_quantity < 0:
                    raise ValueError(
                        f"Cannot apply delta {delta:g} to '{product.name}': "
                        f"stock is {product.quantity:g} {product.unit}, "
                        f"which would leave {new_quantity:g}."
                    )
                product.quantity = new_quantity
                _write_raw(products)
                return product
    raise KeyError(product_id)


def low_stock(threshold: float) -> list[Product]:
    with _lock:
        return [p for p in _read_raw() if p.quantity < threshold]


def seed_if_empty() -> None:
    """Seed a starter catalogue on first run so the agent has something
    to talk about. Never overwrites an existing file."""
    if csv_path().exists():
        return
    with _lock:
        _write_raw(
            [
                Product(1, "Arabica beans", 45, "kg"),
                Product(2, "Robusta beans", 22, "kg"),
                Product(3, "Oat milk", 8, "liters"),
                Product(4, "Whole milk", 30, "liters"),
                Product(5, "Paper cups 12oz", 400, "units"),
                Product(6, "Paper cups 16oz", 120, "units"),
                Product(7, "Vanilla syrup", 6, "liters"),
                Product(8, "Caramel syrup", 14, "liters"),
                Product(9, "Takeaway lids", 90, "units"),
                Product(10, "Napkins", 5, "packs"),
            ]
        )
