import csv
import io
import random
import string
import sys
import time

import modal

stub = modal.Stub("big-data-csv")
image = modal.Image.debian_slim().pip_install(
    [
        "boto3",
        "dask",
        "pandas",
        "pyarrow",
        "s3fs",
    ]
)


def random_alphanum_str(min_len: int, max_len: int) -> str:
    s_len = random.randrange(min_len, max_len)
    return "".join(random.choices(string.ascii_letters + string.digits, k=s_len))


def fake_csv_data(size_mb: int):
    """u/kawaii_kebab's analysis problem had the schema 'email STRING, password STRING."""
    domains = {}
    csvfile = io.StringIO()
    spamwriter = csv.writer(
        csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )

    approx_entry_bytes = 45
    entries_per_mb = (1024 * 1024) / approx_entry_bytes
    required_entries = int(size_mb * entries_per_mb)
    domains = {
        "gmail.com": 0.25,
        "hotmail.com": 0.2,
        "yahoo.com": 0.2,
        "outlook.com": 0.15,
        "proton.me": 0.15,
        "foo.com": 0.05,
    }
    print(f"Producing {required_entries} lines of (email, password) CSV data.")
    for dom in random.choices(
        population=list(domains.keys()),
        weights=list(domains.values()),
        k=required_entries,
    ):
        local_part = random_alphanum_str(min_len=5, max_len=25)
        email = f"{local_part}@{dom}"
        password = random_alphanum_str(min_len=5, max_len=25)
        spamwriter.writerow([email, password])
    data = csvfile.getvalue()
    csvfile.close()
    return data


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def upload_part(bucket, key, upload_id, part_num, size_mb):
    import boto3

    s3_resource = boto3.resource("s3")
    print(f"Uploading part {part_num} for upload ID {upload_id}")
    upload_part = s3_resource.MultipartUploadPart(
        bucket,
        key,
        upload_id,
        part_num,
    )

    part_data = fake_csv_data(size_mb=size_mb)
    print(f"Part {part_num} is {sys.getsizeof(part_data)} bytes")
    part_response = upload_part.upload(
        Body=part_data,
    )
    return (part_num, part_response)


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def upload_fake_csv(desired_mb: int):
    import boto3

    bucket_name = "temp-big-data-csv"
    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=bucket_name)

    print(boto3.client("sts").get_caller_identity())

    key = f"{desired_mb}_mb.csv"
    multipart_upload = s3_client.create_multipart_upload(
        ACL="private",
        Bucket=bucket_name,
        Key=key,
    )

    upload_id = multipart_upload["UploadId"]
    print(f"Upload ID: {upload_id}")
    upload_size_mb = 100  # Constrained by how fast Python can produce fake data?
    num_uploads = desired_mb // upload_size_mb

    uploads = [
        (bucket_name, key, upload_id, i, upload_size_mb)
        for i in range(1, num_uploads + 1)
    ]

    parts = []
    for (part_number, part_response) in upload_part.starmap(uploads):
        parts.append({"PartNumber": part_number, "ETag": part_response["ETag"]})

    print("Completing upload...")
    result = s3_client.complete_multipart_upload(
        Bucket=bucket_name,
        Key=key,
        MultipartUpload={"Parts": parts},
        UploadId=multipart_upload["UploadId"],
    )
    print(result)
    print("✅ Done")


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def count_by_filter_slow(csv_file_path) -> int:
    import pandas as pd

    count = 0
    with pd.read_csv(
        csv_file_path,
        chunksize=10_000,
        engine="c",
        delimiter=" ",
        quotechar="|",
        quoting=csv.QUOTE_MINIMAL,
        names=["email", "password"],
        header=None,
    ) as reader:
        for i, chunk in enumerate(reader):
            count += chunk["email"].str.endswith("@gmail.com").count()
            if i % 100 == 0:
                print(f"Processed {i} chunks,")
    return count


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def process_block(i, df):
    partition = df.partitions[i]
    count = (
        partition[partition["email"].str.endswith("@gmail.com")]["email"]
        .count()
        .compute()
    )
    print(f"Counted {count} in csv partition {i}")
    return count


@stub.function(image=image, secret=modal.Secret.from_name("personal-aws-user"))
def count_by_filter_fast(bucket: str, key: str) -> int:
    import boto3
    import pandas as pd
    from dask.dataframe.io import read_csv

    s3 = boto3.client("s3")
    response = s3.head_object(Bucket=bucket, Key=key)
    size = response["ContentLength"]
    print(f"CSV file is {size} bytes")

    csv_file_path = f"s3://{bucket}/{key}"
    df = read_csv(
        urlpath=csv_file_path,
        blocksize="128 MiB",
        sep=" ",
        names=["email", "password"],
    )

    print(f"Created dataframe has {df.npartitions} partitions.")
    return sum(
        process_block.starmap(
            ((i, df) for i in range(df.npartitions)), order_outputs=False
        )
    )


if __name__ == "__main__":
    with stub.run():
        # upload_fake_csv(desired_mb=10_000)
        print("Running fast...")
        start = time.time()
        count = count_by_filter_fast(bucket="temp-big-data-csv", key="10000_mb.csv")
        end = time.time()
        print(f"Returned {count=} in {end - start} seconds.")
