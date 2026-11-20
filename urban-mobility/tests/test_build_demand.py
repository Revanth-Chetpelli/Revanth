import pandas as pd

from src.build_demand import build_hourly_demand


def test_build_hourly_demand_filters_month_and_counts_by_zone(tmp_path):
    source = tmp_path / "sample.parquet"
    output = tmp_path / "processed" / "demand.parquet"
    sample = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2024-01-01 00:05:00",
                "2024-01-01 00:40:00",
                "2024-01-01 00:20:00",
                "2023-12-31 23:59:00",
                "2024-02-01 00:00:00",
                "not-a-date",
                "2024-01-01 01:00:00",
            ],
            "PULocationID": [1, 1, 2, 1, 1, 1, 0],
        }
    )
    sample.to_parquet(source, index=False)

    report = build_hourly_demand(source, output, "2024-01")
    result = pd.read_parquet(output).sort_values(["hour", "zone_id"]).reset_index(drop=True)

    assert report == {
        "rows_read": 7,
        "rows_kept_in_month": 3,
        "rows_excluded": 4,
        "hour_zone_rows": 2,
        "zones": 2,
    }
    assert result["pickup_count"].tolist() == [2, 1]
    assert result["zone_id"].tolist() == [1, 2]
    assert str(result.loc[0, "hour"]) == "2024-01-01 00:00:00"
