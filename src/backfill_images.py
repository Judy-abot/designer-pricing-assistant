"""
Backfill primary image URLs onto an existing Etsy listings CSV.

WHY THIS EXISTS
----------------
etsy_collector.py's --fetch-images flag fetches image_url at collection
time, but Project 1's dataset (data/raw/listings.csv) was collected with
that flag off -- image_url is already a column in the CSV, it's just
empty for every row. Re-running the collector from scratch with
--fetch-images on would risk pulling a different sample of listings than
the exact one Project 1 was evaluated on. This script instead does a
targeted backfill: it takes the listing ids already in listings.csv,
as-is, and fetches only their images -- one extra API call per listing,
same endpoint etsy_collector.py already knows how to call -- leaving
every other column untouched.

WHERE TO RUN THIS
------------------
Same constraint as etsy_collector.py: needs real internet access to
https://api.etsy.com, which sandboxed dev environments typically block.
Run on your own machine. Uses the same ETSY_API_KEY / ETSY_SHARED_SECRET
environment variables as the collector -- if you still have those
exported from Project 1, you're already set up.

USAGE
-----
    python src/backfill_images.py --in data/raw/listings.csv \
        --out data/raw/listings_with_images.csv

Resumable: if this gets interrupted (network blip, daily API quota
cutoff), just re-run the exact same command. Any listing_id already
written to --out gets skipped, so you never waste a call re-fetching a
listing you already tried.
"""

import argparse
import csv
import os
import sys

from etsy_collector import CSV_FIELDS, EtsyClient


def backfill(in_path: str, out_path: str) -> None:
    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        sys.exit(1)

    api_key = os.environ.get("ETSY_API_KEY")
    shared_secret = os.environ.get("ETSY_SHARED_SECRET")
    client = EtsyClient(api_key, shared_secret)

    with open(in_path, newline="", encoding="utf-8") as f:
        input_rows = list(csv.DictReader(f))
    total = len(input_rows)
    print(f"Loaded {total} listings from {in_path}")

    # Resume support, same idea as etsy_collector.py: any listing_id
    # already written to --out -- whether the image fetch succeeded or
    # failed -- gets skipped on a re-run. A failed fetch (e.g. a listing
    # that's been deleted since Project 1 collected it) will just fail
    # again identically, so there's no point spending a retry on it.
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["listing_id"])
        print(f"Resuming: {len(done_ids)} listings already in {out_path}")

    write_header = not os.path.exists(out_path)
    fetched = 0
    already_had_image = 0
    failed = 0

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(input_rows, start=1):
            lid = row["listing_id"]
            if lid in done_ids:
                continue

            if row.get("image_url"):
                # Already has an image (e.g. this row came from a run
                # that used --fetch-images) -- no need to spend a call.
                already_had_image += 1
            else:
                try:
                    data = client.get_listing_images(int(lid))
                    results = data.get("results") or []
                    row["image_url"] = results[0].get("url_570xN") if results else ""
                    if row["image_url"]:
                        fetched += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  [warn] Couldn't fetch image for listing {lid} ({e})")
                    row["image_url"] = ""
                    failed += 1

            writer.writerow(row)
            # Flush after every row (not just at close) -- each request
            # is network-bound and this whole run takes long enough that
            # losing an hour of progress to a mid-run crash would be a
            # real cost, not just a style nitpick.
            f.flush()

            if i % 200 == 0 or i == total:
                print(f"  ...{i}/{total} processed "
                      f"({fetched} fetched, {already_had_image} already had images, "
                      f"{failed} failed)")

    print(f"Done. {fetched} images fetched, {already_had_image} already had images, "
          f"{failed} failed/missing. Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill primary image URLs onto an existing Etsy listings CSV."
    )
    parser.add_argument("--in", dest="in_path", type=str, default="data/raw/listings.csv",
                         help="Input CSV (from etsy_collector.py) to read listing ids from")
    parser.add_argument("--out", dest="out_path", type=str,
                         default="data/raw/listings_with_images.csv",
                         help="Output CSV path (created fresh, or resumed if it already exists)")
    args = parser.parse_args()
    backfill(args.in_path, args.out_path)