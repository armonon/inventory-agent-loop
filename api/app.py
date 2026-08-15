"""FastAPI inventory service for Carla's coffee-shop supply store.

Four endpoints, all backed by products.csv:

    GET   /inventory                  full product list
    POST  /inventory                  add a product
    PATCH /inventory/{product_id}     apply a stock delta (+ in, - out)
    GET   /inventory/alerts           products below a threshold

The agent in agent.py consumes these as its tool set. Nothing here
knows about the LLM — this is a plain REST API you can curl.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from api import storage


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Seed a starter catalogue on first boot; never overwrites existing data.
    storage.seed_if_empty()
    yield


app = FastAPI(
    title="Inventory API",
    description="Coffee-shop supply inventory, persisted to products.csv.",
    version="1.0.0",
    lifespan=lifespan,
)

DEFAULT_ALERT_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ProductOut(BaseModel):
    id: int
    name: str
    quantity: float
    unit: str


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Product name, e.g. 'Oat milk'")
    quantity: float = Field(..., ge=0, description="Starting stock, must be >= 0")
    unit: str = Field(
        ..., min_length=1, description="Unit of measure: units, kg, liters, packs..."
    )


class StockDelta(BaseModel):
    delta: float = Field(
        ...,
        description=(
            "Change to apply. Positive for incoming stock (a delivery), "
            "negative for outgoing (a sale)."
        ),
    )


class AlertsOut(BaseModel):
    threshold: float
    count: int
    products: list[ProductOut]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "inventory-api"}


@app.get("/inventory", response_model=list[ProductOut], tags=["inventory"])
def get_inventory() -> list[ProductOut]:
    """Return the full product list."""
    return [ProductOut(**p.to_dict()) for p in storage.list_products()]


@app.post(
    "/inventory",
    response_model=ProductOut,
    status_code=status.HTTP_201_CREATED,
    tags=["inventory"],
)
def create_product(payload: ProductCreate) -> ProductOut:
    """Add a new product to the catalogue."""
    existing = storage.find_by_name(payload.name)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"A product named '{existing.name}' already exists "
                f"(id {existing.id}, {existing.quantity:g} {existing.unit}). "
                "Use PATCH /inventory/{id} to change its stock."
            ),
        )
    product = storage.add_product(payload.name, payload.quantity, payload.unit)
    return ProductOut(**product.to_dict())


@app.patch(
    "/inventory/{product_id}", response_model=ProductOut, tags=["inventory"]
)
def update_stock(product_id: int, payload: StockDelta) -> ProductOut:
    """Apply a stock delta. Positive = delivery in, negative = sale out."""
    try:
        product = storage.apply_delta(product_id, payload.delta)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No product with id {product_id}.",
        ) from None
    except ValueError as exc:
        # Stock would go negative — a real data error, not a silent clamp.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from None
    return ProductOut(**product.to_dict())


@app.get("/inventory/alerts", response_model=AlertsOut, tags=["inventory"])
def get_alerts(
    threshold: float = Query(
        DEFAULT_ALERT_THRESHOLD,
        ge=0,
        description="Products with quantity strictly below this value are returned.",
    ),
) -> AlertsOut:
    """Return every product running low, i.e. quantity < threshold."""
    low = storage.low_stock(threshold)
    return AlertsOut(
        threshold=threshold,
        count=len(low),
        products=[ProductOut(**p.to_dict()) for p in low],
    )


@app.get("/inventory/by-name/{name}", response_model=ProductOut, tags=["inventory"])
def get_by_name(name: str) -> ProductOut:
    """Look a product up by name.

    Exists so the agent can resolve "oat milk" to an id without
    guessing — the LLM is given this as a tool.
    """
    product = storage.find_by_name(name)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No product named '{name}'. "
                "Call list_inventory to see the available product names."
            ),
        )
    return ProductOut(**product.to_dict())
