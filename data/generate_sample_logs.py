"""
Generates reproducible synthetic microservice log data for LogSleuth AI.
Includes normal operations and a synthetic incident with payment connection pool exhaustion,
cascading timeouts, and order retries.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def generate_sample_logs(num_records: int = 260, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    base_time = datetime(2026, 10, 27, 10, 0, 0)

    services = [
        "authentication-service",
        "product-service",
        "order-service",
        "payment-service",
        "inventory-service"
    ]

    normal_messages = {
        "authentication-service": [
            ("INFO", "User login successful token issued for uid={}"),
            ("INFO", "JWT token refreshed for session id={}"),
            ("INFO", "Public key cached for auth verification"),
            ("INFO", "OAuth2 callback validated successfully"),
            ("INFO", "User logout completed session invalidated")
        ],
        "product-service": [
            ("INFO", "Product catalog cache hit for sku={}"),
            ("INFO", "Product details queried for category id={}"),
            ("INFO", "Fetched 20 active catalog items"),
            ("INFO", "Product pricing updated for flash sale sku={}"),
            ("INFO", "Inventory sync received for sku={}")
        ],
        "order-service": [
            ("INFO", "Incoming order checkout request received order_id={}"),
            ("INFO", "Order validation passed for customer={}"),
            ("INFO", "Order status transitioned to PENDING_PAYMENT"),
            ("INFO", "Cart details validated successfully"),
            ("INFO", "Order fulfillment dispatched to inventory")
        ],
        "payment-service": [
            ("INFO", "Payment intent created for amount=${}"),
            ("INFO", "Stripe gateway webhook verified"),
            ("INFO", "Payment authorization approved auth_code={}"),
            ("INFO", "Customer balance checked successfully"),
            ("INFO", "Payment batch reconciliation finished")
        ],
        "inventory-service": [
            ("INFO", "Stock level checked for sku={} available=45"),
            ("INFO", "Inventory reservation confirmed for order_id={}"),
            ("INFO", "Warehouse stock replenishing batch received"),
            ("INFO", "Item quantity updated in warehouse-01"),
            ("INFO", "Inventory audit completed without discrepancy")
        ]
    }

    records = []
    current_time = base_time

    # Part 1: Initial Normal Operating Phase (first 100 records)
    for i in range(100):
        current_time += timedelta(seconds=random.randint(1, 4))
        svc = random.choice(services)
        lvl, template = random.choice(normal_messages[svc])
        val = random.randint(1000, 9999)
        records.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "service": svc,
            "level": lvl,
            "message": template.format(val),
            "synthetic_incident": 0
        })

    # Part 2: Incident Onset Phase (records 101 to 170)
    # Payment-service starts running out of DB connections and throws timeouts
    incident_start_time = current_time
    for i in range(70):
        current_time += timedelta(seconds=random.randint(1, 3))
        r = random.random()

        if r < 0.45:
            # Payment-service degradation
            payment_incident_logs = [
                ("WARNING", "DB connection pool utilization reached 96% (active=48/50)"),
                ("ERROR", "Timeout waiting for available database connection in pool after 5000ms"),
                ("ERROR", "Connection refused to payment gateway host api.stripe.internal:443"),
                ("CRITICAL", "HikariPool-1 - Connection is not available, request timed out after 30004ms"),
                ("ERROR", "Payment authorization failed for order_id={}: Connection timed out")
            ]
            lvl, msg = random.choice(payment_incident_logs)
            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": "payment-service",
                "level": lvl,
                "message": msg.format(random.randint(1000, 9999)),
                "synthetic_incident": 1
            })
        elif r < 0.80:
            # Order-service cascading failures & retries
            order_incident_logs = [
                ("WARNING", "Call to downstream payment-service delayed response latency=4820ms"),
                ("WARNING", "Payment gateway call timed out; retrying attempt 1 of 3 for order_id={}"),
                ("WARNING", "Payment gateway call timed out; retrying attempt 2 of 3 with exponential backoff"),
                ("ERROR", "Payment gateway failed after 3 retry attempts; aborting checkout for order_id={}"),
                ("ERROR", "HTTP 504 Gateway Timeout received from downstream payment-service")
            ]
            lvl, msg = random.choice(order_incident_logs)
            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": "order-service",
                "level": lvl,
                "message": msg.format(random.randint(1000, 9999)),
                "synthetic_incident": 1
            })
        else:
            # Other services continuing routine operations
            svc = random.choice(["authentication-service", "product-service", "inventory-service"])
            lvl, template = random.choice(normal_messages[svc])
            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": svc,
                "level": lvl,
                "message": template.format(random.randint(1000, 9999)),
                "synthetic_incident": 0
            })

    # Part 3: Recovery Phase & Post-Incident Traffic (records 171 to 260)
    for i in range(90):
        current_time += timedelta(seconds=random.randint(1, 4))
        if i == 5:
            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": "payment-service",
                "level": "INFO",
                "message": "HikariPool connection pool refreshed. Active connections restored to nominal: 4/50",
                "synthetic_incident": 0
            })
        elif i == 8:
            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": "order-service",
                "level": "INFO",
                "message": "Payment service healthcheck passed. Resuming queued checkout processing.",
                "synthetic_incident": 0
            })
        else:
            svc = random.choice(services)
            lvl, template = random.choice(normal_messages[svc])
            records.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "service": svc,
                "level": lvl,
                "message": template.format(random.randint(1000, 9999)),
                "synthetic_incident": 0
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "sample_logs.csv"
    df = generate_sample_logs(num_records=260, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} sample log records to {out_path}")
