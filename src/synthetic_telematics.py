"""
Generate synthetic fleet telematics data, loosely shaped like Geotab's
LogRecord (GPS pings) and ExceptionEvent (harsh braking, speeding) data.

Run:  python src/synthetic_telematics.py   (works from any folder)
Out:  data/raw/gps_pings.csv, data/raw/trips.csv, data/raw/events.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

# This file lives in <project root>/src/, so the root is one folder up.
# Path(__file__) is this script's own location, so it works from any folder.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

RNG = np.random.default_rng(42)  # fixed seed -> reproducible data
PING_SECONDS = 10                # one GPS record every 10 s while driving
DEPOT = (33.7490, -84.3880)      # start point (lat, lon); change to taste
SPEED_LIMIT_KMH = 100
KM_PER_DEG_LAT = 111.0


def simulate_trip(vehicle_id: str, trip_id: str, start: pd.Timestamp):
    """Simulate one trip: accelerate, cruise with noise, stop in traffic, brake."""
    n_pings = int(RNG.integers(60, 360))          # 10 min to 1 hour
    cruise = RNG.uniform(40, 95)                  # driver's typical speed, km/h
    heading = RNG.uniform(0, 2 * np.pi)

    speed = np.zeros(n_pings)
    for i in range(1, n_pings):
        target = cruise + RNG.normal(0, 8)
        if RNG.random() < 0.03:                   # random stop (light, traffic)
            target = 0
        # speed can only change so fast between pings (+/- ~15 km/h)
        speed[i] = np.clip(speed[i - 1] + np.clip(target - speed[i - 1], -25, 15), 0, 140)
    speed[-3:] = np.linspace(speed[-4], 0, 3)     # slow down at the end

    # random-walk heading, then integrate speed into position
    heading += np.cumsum(RNG.normal(0, 0.08, n_pings))
    step_km = speed * PING_SECONDS / 3600
    lat = DEPOT[0] + np.cumsum(step_km * np.cos(heading)) / KM_PER_DEG_LAT
    lon = DEPOT[1] + np.cumsum(step_km * np.sin(heading)) / (
        KM_PER_DEG_LAT * np.cos(np.radians(DEPOT[0]))
    )

    times = start + pd.to_timedelta(np.arange(n_pings) * PING_SECONDS, unit="s")
    pings = pd.DataFrame(
        {
            "vehicle_id": vehicle_id,
            "trip_id": trip_id,
            "timestamp": times,
            "latitude": lat + RNG.normal(0, 2e-5, n_pings),   # GPS jitter
            "longitude": lon + RNG.normal(0, 2e-5, n_pings),
            "speed_kmh": speed.round(1),
        }
    )

    # derive events from the speed trace (this is the analytics-y part)
    accel = np.diff(speed, prepend=0) / PING_SECONDS      # km/h per second
    events = []
    for i in np.where(accel < -4.5)[0]:                    # harsh braking
        events.append((vehicle_id, trip_id, times[i], "harsh_braking"))
    for i in np.where(speed > SPEED_LIMIT_KMH)[0]:
        events.append((vehicle_id, trip_id, times[i], "speeding"))

    trip = {
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "start": times[0],
        "end": times[-1],
        "distance_km": round(step_km.sum(), 2),
        "idle_seconds": int((speed == 0).sum() * PING_SECONDS),
        "max_speed_kmh": speed.max(),
    }
    return pings, trip, events


def main(n_vehicles=5, n_days=14):
    all_pings, all_trips, all_events = [], [], []
    day0 = pd.Timestamp("2026-10-01")

    for v in range(n_vehicles):
        vehicle_id = f"V{v + 1:03d}"
        for d in range(n_days):
            if RNG.random() < 0.2:                         # day off
                continue
            for t in range(int(RNG.integers(1, 5))):       # 1-4 trips per day
                start = day0 + pd.Timedelta(days=d, hours=int(RNG.integers(6, 18)),
                                            minutes=int(RNG.integers(0, 60)))
                trip_id = f"{vehicle_id}-{d:02d}-{t}"
                pings, trip, events = simulate_trip(vehicle_id, trip_id, start)
                all_pings.append(pings)
                all_trips.append(trip)
                all_events.extend(events)

    out = RAW_DIR
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(all_pings).to_csv(out / "gps_pings.csv", index=False)
    pd.DataFrame(all_trips).to_csv(out / "trips.csv", index=False)
    pd.DataFrame(all_events, columns=["vehicle_id", "trip_id", "timestamp", "event"]
                 ).to_csv(out / "events.csv", index=False)
    print(f"{len(all_trips)} trips, {sum(map(len, all_pings))} pings, {len(all_events)} events")


if __name__ == "__main__":
    main()