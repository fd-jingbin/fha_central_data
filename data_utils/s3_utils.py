import gzip
import os
import pickle
import tempfile
from typing import Any, Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


class S3DataFrameStore:
    """
    存/取 pandas DataFrame
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        region_name: Optional[str] = None,
        gzip_compress: bool = True,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.gzip_compress = gzip_compress

        session = boto3.session.Session(region_name=region_name)
        self.s3 = session.client(
            "s3",
            config=Config(
                retries={"max_attempts": 10, "mode": "standard"},
                connect_timeout=10,
                read_timeout=300,
            ),
        )

        # 让大文件更稳：超过 64MB 自动 multipart
        MB = 1024 * 1024
        self.transfer_cfg = TransferConfig(
            multipart_threshold=64 * MB,
            multipart_chunksize=64 * MB,
            max_concurrency=4,
            use_threads=True,
        )

    def _key(self, name: str) -> str:
        name = name.strip().replace("\\", "/")
        ext = "pkl.gz" if self.gzip_compress else "pkl"
        key = f"{name}.{ext}" if not name.endswith(f".{ext}") else name
        return f"{self.prefix}/{key}" if self.prefix else key

    def save_df(self, name: str, df: Any, *, overwrite: bool = True) -> str:
        key = self._key(name)

        if not overwrite:
            # 轻量检查存在性
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                raise FileExistsError(f"s3://{self.bucket}/{key}")
            except self.s3.exceptions.NoSuchKey:
                pass
            except Exception as e:
                # head_object 不同错误码表现不一致，简单起见：不存在就继续，其他就抛
                if "404" not in str(e) and "NotFound" not in str(e):
                    raise

        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, "df_dump.pkl" + (".gz" if self.gzip_compress else ""))

            if self.gzip_compress:
                with gzip.open(local_path, "wb") as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
                extra = {"ContentType": "application/octet-stream", "ContentEncoding": "gzip"}
            else:
                with open(local_path, "wb") as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
                extra = {"ContentType": "application/octet-stream"}

            self.s3.upload_file(
                Filename=local_path,
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=extra,
                Config=self.transfer_cfg,
            )

        return f"s3://{self.bucket}/{key}"

    def load_df(self, name: str) -> Any:
        key = self._key(name)
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
