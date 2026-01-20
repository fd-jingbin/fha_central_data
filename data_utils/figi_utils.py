import json
import urllib.request
import os
import pandas as pd
from tornado.httpclient import HTTPError

import data_config as da_cfg
import data_utils.utils as fl_ut


def _api_call(payload):
    headers = {"Content-Type": "application/json"}
    if da_cfg.OPENFIGI_API_KEY:
        headers["X-OPENFIGI-APIKEY"] = da_cfg.OPENFIGI_API_KEY

    req = urllib.request.Request(
        url=da_cfg.OPENFIGI_BASE_URL,
        data=bytes(json.dumps(payload), encoding="utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def get_figi_mapping(figi_list=None, chunk_size=100):

    fl_ut.exist_create_folder(da_cfg.BBG_DEFAULT_PICKLE_DIR)
    theo_pkl_dir = str(os.path.join(da_cfg.BBG_DEFAULT_PICKLE_DIR, da_cfg.FIGI_MAPPING_PKL_FN))

    if figi_list is None:
        return pd.read_pickle(theo_pkl_dir)

    else:

        try:
            existing_df = pd.read_pickle(theo_pkl_dir)
            existing_set = set(existing_df.figi.unique())
            new_figi_list = [x for x in figi_list if x not in existing_set]
        except FileNotFoundError:
            new_figi_list = figi_list
            existing_df = pd.DataFrame(columns=['figi', 'securityDescription', 'exchCode', 'inputFIGI'])

        all_results = []

        # Split into chunks of 500
        try:
            for i in range(0, len(new_figi_list), chunk_size):
                print(f'Retrieving {i} to {i + chunk_size} out of {len(new_figi_list)}')
                chunk = new_figi_list[i:i+chunk_size]
                payload = [{"idType": "ID_BB_GLOBAL", "idValue": f} for f in chunk]
                results = _api_call(payload)

                # Map each result to the input FIGI explicitly
                for input_figi, r in zip(chunk, results):
                    if "data" in r and r["data"]:
                        d = r["data"][0]  # take first match
                        all_results.append({
                            "inputFIGI": input_figi,
                            "figi": d.get("figi"),
                            "securityDescription": d.get("securityDescription"),
                            "exchCode": d.get("exchCode")
                        })
                    else:
                        all_results.append({
                            "inputFIGI": input_figi,
                            "figi": None,
                            "securityDescription": None,
                            "exchCode": None
                        })
        except HTTPError:
            print('Connection Lost, please retry!')

        new_df = pd.DataFrame(all_results)
        final_out = pd.concat([existing_df, new_df], ignore_index=True)
        final_out.dropna().to_pickle(theo_pkl_dir)

        return final_out
