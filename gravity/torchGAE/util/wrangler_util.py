from multiprocessing.sharedctypes import Value
import awswrangler as wr
import os 

def read_parquet(bucket, prefix, key, n_cores):
    if os.cpu_count() < n_cores:
        raise ValueError("The number of CPU's specified exceed the amount available")

    df = wr.s3.read_parquet(
        path=f"s3://{bucket}/{prefix}/{key}/",
        dataset=True,
        path_ignore_suffix=["_SUCCESS"],
        use_threads=os.cpu_count()
    )

    return df

def write_parquet(df, bucket, prefix, key, n_cores):
    if os.cpu_count() < n_cores:
        raise ValueError("The number of CPU's specified exceed the amount available")
    
    wr.s3.to_parquet(
        df=df,
        path=f"s3://{bucket}/{prefix}/{key}",
        dataset=True,
        mode="overwrite",
        use_threads=os.cpu_count()
    )