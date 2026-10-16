import csv, os


class CsvSink:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", newline=""); self.w = None

    def write(self, row):
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row)); self.w.writeheader()
        self.w.writerow(row)

    def close(self): self.f.close()


class InfluxSink:
    """Writes measurement `drift_stream` (fields: err, rolling_err, alarm, true_drift, latency_ms) to InfluxDB 2.x."""
    def __init__(self, url="http://localhost:8086", token="drift-token", org="drift-org", bucket="drift", run="run1"):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS
        self.c = InfluxDBClient(url=url, token=token, org=org)
        self.w, self.bucket, self.org, self.run = self.c.write_api(write_options=SYNCHRONOUS), bucket, org, run

    def write(self, row):
        from influxdb_client import Point
        p = (Point("drift_stream").tag("run", self.run).field("err", row["err"])
             .field("rolling_err", float(row["rolling_err"])).field("alarm", row["alarm"])
             .field("true_drift", row["true_drift"]))
        if "latency_ms" in row:
            p = p.field("latency_ms", float(row["latency_ms"]))
        self.w.write(bucket=self.bucket, org=self.org, record=p)

    def close(self): self.c.close()
