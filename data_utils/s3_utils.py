from __future__ import annotations

import gzip
import os
import pickle
import tempfile
from typing import Optional, Any

import boto3
from botocore.config import Config


class S3DataFrameStore:
    """Upload/Download pandas DataFrame to/from S3 with pickle+gzip."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        gzip_compress: bool = True,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.gzip_compress = gzip_compress

        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.s3 = session.client(
            "s3",
            config=Config(
                retries={"max_attempts": 10, "mode": "standard"},
                connect_timeout=10,
                read_timeout=300,
            ),
        )

    def _key(self, path: str) -> str:
        """path is like 'fha/test/df_a' or 'fha/test/df_a.pkl.gz'"""
        path = path.strip("/")

        # normalize extension
        if self.gzip_compress:
            if not path.endswith(".pkl.gz"):
                path += ".pkl.gz"
        else:
            if not path.endswith(".pkl"):
                path += ".pkl"

        return f"{self.prefix}/{path}" if self.prefix else path

    def upload_df(self, path: str, df: Any, *, overwrite: bool = True) -> str:
        key = self._key(path)

        if not overwrite:
            # light existence check
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                raise FileExistsError(f"s3://{self.bucket}/{key}")
            except Exception as e:
                # if it's not "exists", continue; otherwise re-raise
                msg = str(e)
                if "404" not in msg and "NotFound" not in msg and "NoSuchKey" not in msg:
                    # could be AccessDenied etc.
                    pass

        # write to temp file (stable for big df)
        suffix = ".pkl.gz" if self.gzip_compress else ".pkl"
        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, "df" + suffix)

            if self.gzip_compress:
                with gzip.open(local_path, "wb") as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
                extra = {"ContentType": "application/octet-stream", "ContentEncoding": "gzip"}
            else:
                with open(local_path, "wb") as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
                extra = {"ContentType": "application/octet-stream"}

            self.s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra)

        return f"s3://{self.bucket}/{key}"

    def download_df(self, path: str) -> Any:
        key = self._key(path)
        resp = self.s3.get_object(Bucket=self.bucket, Key=key)
        body = resp["Body"]

        try:
            if self.gzip_compress:
                with gzip.GzipFile(fileobj=body, mode="rb") as gz:
                    return pickle.load(gz)
            else:
                return pickle.load(body)
        finally:
            body.close()

#
# import pandas as pd
#
# store = S3DataFrameStore(
#     bucket="fengheai-jingbin-data",
#     prefix="fha/test",              # 可选：统一前缀
#     region_name="ap-southeast-1",   # 建议写上 bucket 所在 region
#     # profile_name="jingbin",       # 如果你用 profile
# )
#
# df = pd.DataFrame({"a":[1,2], "b":[3,4]})
#
# uri = store.upload_df("df_a", df)      # -> s3://bucket/fha/test/df_a.pkl.gz
# df2 = store.download_df("df_a")
# print(uri, df2.head())