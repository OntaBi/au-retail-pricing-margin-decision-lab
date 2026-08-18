from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_SKUS = 1500

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

DEPARTMENTS = {
    "Technology": {
        "Computers": [
            "Laptops",
            "Monitors",
            "Keyboards & Mice",
            "Computer Accessories",
        ],
        "Printing": [
            "Printers",
            "Ink & Toner",
            "Printer Accessories",
        ],
        "Audio": [
            "Headphones",
            "Speakers",
            "Audio Accessories",
        ],
    },
    "Office Supplies": {
        "Stationery": [
            "Pens",
            "Notebooks",
            "Filing",
            "Desk Accessories",
        ],
        "Paper": [
            "Copy Paper",
            "Specialty Paper",
            "Labels",
        ],
        "Mail & Packaging": [
            "Mailing",
            "Packaging",
            "Tape",
        ],
    },
    "Furniture": {
        "Office Furniture": [
            "Desks",
            "Office Chairs",
            "Storage",
        ],
        "Home Office": [
            "Home Office Desks",
            "Home Office Chairs",
            "Ergonomic Accessories",
        ],
    },
}


BRANDS = [
    "Apex",
    "Nova",
    "Vertex",
    "Pulse",
    "Orbit",
    "Summit",
    "Core",
    "Element",
    "Vantage",
    "Metro",
    "Essentials",
    "ProLine",
]


SUPPLIERS = [
    "Southern Cross Distribution",
    "Pacific Retail Supply",
    "Metro Wholesale Group",
    "Australian Technology Partners",
    "National Office Supply",
    "Vertex Distribution",
    "East Coast Wholesale",
    "Central Merchandising Group",
]


PRICE_BANDS = [
    "Entry",
    "Value",
    "Core",
    "Premium",
]


CLASS_ECONOMICS = {
    # Technology - Computers
    "Laptops": ((399, 2499), (0.12, 0.25)),
    "Monitors": ((129, 899), (0.18, 0.32)),
    "Keyboards & Mice": ((15, 299), (0.25, 0.45)),
    "Computer Accessories": ((10, 249), (0.30, 0.55)),

    # Technology - Printing
    "Printers": ((79, 899), (0.15, 0.30)),
    "Ink & Toner": ((15, 249), (0.28, 0.48)),
    "Printer Accessories": ((8, 149), (0.30, 0.55)),

    # Technology - Audio
    "Headphones": ((15, 599), (0.25, 0.45)),
    "Speakers": ((25, 699), (0.25, 0.45)),
    "Audio Accessories": ((8, 149), (0.30, 0.55)),

    # Office Supplies - Stationery
    "Pens": ((2, 49), (0.35, 0.60)),
    "Notebooks": ((3, 39), (0.35, 0.58)),
    "Filing": ((3, 79), (0.32, 0.55)),
    "Desk Accessories": ((5, 129), (0.32, 0.55)),

    # Office Supplies - Paper
    "Copy Paper": ((6, 45), (0.18, 0.35)),
    "Specialty Paper": ((8, 69), (0.25, 0.45)),
    "Labels": ((5, 89), (0.30, 0.50)),

    # Office Supplies - Mail & Packaging
    "Mailing": ((2, 59), (0.30, 0.55)),
    "Packaging": ((3, 89), (0.28, 0.50)),
    "Tape": ((2, 39), (0.32, 0.55)),

    # Furniture - Office Furniture
    "Desks": ((99, 899), (0.28, 0.48)),
    "Office Chairs": ((79, 999), (0.30, 0.50)),
    "Storage": ((49, 699), (0.28, 0.48)),

    # Furniture - Home Office
    "Home Office Desks": ((79, 599), (0.28, 0.48)),
    "Home Office Chairs": ((69, 699), (0.30, 0.50)),
    "Ergonomic Accessories": ((15, 249), (0.32, 0.55)),
}


def build_hierarchy_lookup() -> list[tuple[str, str, str]]:
    hierarchy = []

    for department, categories in DEPARTMENTS.items():
        for category, classes in categories.items():
            for product_class in classes:
                hierarchy.append(
                    (
                        department,
                        category,
                        product_class,
                    )
                )

    return hierarchy


def generate_product_master(
    n_skus: int = N_SKUS,
    seed: int = SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hierarchy = build_hierarchy_lookup()

    hierarchy_index = rng.integers(
        0,
        len(hierarchy),
        size=n_skus,
    )

    hierarchy_values = [
        hierarchy[index]
        for index in hierarchy_index
    ]

    departments = [
        value[0]
        for value in hierarchy_values
    ]

    categories = [
        value[1]
        for value in hierarchy_values
    ]

    product_classes = [
        value[2]
        for value in hierarchy_values
    ]

    regular_sell_price = np.array(
        [
            rng.uniform(
                CLASS_ECONOMICS[product_class][0][0],
                CLASS_ECONOMICS[product_class][0][1],
            )
            for product_class in product_classes
        ]
    )

    target_margin_pct = np.array(
        [
            rng.uniform(
                CLASS_ECONOMICS[product_class][1][0],
                CLASS_ECONOMICS[product_class][1][1],
            )
            for product_class in product_classes
        ]
    )

    regular_sell_price = np.round(
        regular_sell_price,
        2,
    )

    price_band = np.empty(
        n_skus,
        dtype=object,
    )

    for product_class in set(product_classes):
        class_indices = np.where(
            np.array(product_classes) == product_class
        )[0]

        class_prices = regular_sell_price[class_indices]

        q20, q50, q85 = np.quantile(
            class_prices,
            [0.20, 0.50, 0.85],
        )

        for index in class_indices:
            price = regular_sell_price[index]

            if price <= q20:
                price_band[index] = "Entry"
            elif price <= q50:
                price_band[index] = "Value"
            elif price <= q85:
                price_band[index] = "Core"
            else:
                price_band[index] = "Premium"

    cost_price = regular_sell_price * (
        1 - target_margin_pct
    )

    cost_price = np.round(
        cost_price,
        2,
    )

    gross_margin_dollars = (
        regular_sell_price
        - cost_price
    )

    gross_margin_pct = (
        gross_margin_dollars
        / regular_sell_price
    )

    lifecycle_stage = rng.choice(
        [
            "New",
            "Core",
            "Mature",
            "Exit",
        ],
        size=n_skus,
        p=[
            0.08,
            0.62,
            0.23,
            0.07,
        ],
    )

    gst_applicable = rng.choice(
        [True, False],
        size=n_skus,
        p=[0.97, 0.03],
    )

    df = pd.DataFrame(
        {
            "sku_id": [
                f"SKU{i:06d}"
                for i in range(1, n_skus + 1)
            ],
            "department": departments,
            "category": categories,
            "product_class": product_classes,
            "brand": rng.choice(
                BRANDS,
                size=n_skus,
            ),
            "supplier": rng.choice(
                SUPPLIERS,
                size=n_skus,
            ),
            "price_band": price_band,
            "lifecycle_stage": lifecycle_stage,
            "gst_applicable": gst_applicable,
            "cost_price": cost_price,
            "regular_sell_price": regular_sell_price,
            "gross_margin_dollars": np.round(
                gross_margin_dollars,
                2,
            ),
            "gross_margin_pct": np.round(
                gross_margin_pct,
                4,
            ),
        }
    )

    return df


def save_dataset(
    df: pd.DataFrame,
    filename: str,
) -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / filename

    df.to_parquet(
        output_path,
        index=False,
    )

    return output_path


def main() -> None:
    product_master = generate_product_master()

    output_path = save_dataset(
        product_master,
        "product_master.parquet",
    )

    print()
    print("AU Retail Pricing & Margin Decision Lab")
    print("---------------------------------------")
    print(f"SKUs generated : {len(product_master):,}")
    print(f"Departments    : {product_master['department'].nunique():,}")
    print(f"Categories     : {product_master['category'].nunique():,}")
    print(f"Classes        : {product_master['product_class'].nunique():,}")
    print(f"Output         : {output_path}")
    print()

    print("Price summary")
    print(
        product_master[
            [
                "cost_price",
                "regular_sell_price",
                "gross_margin_pct",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()