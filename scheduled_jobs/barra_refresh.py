import sys
sys.path.append(r'/')
sys.path.append(r'C:\Python\fha-research')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from logging_utils.email_utils import notify_on_failure
import data_utils.barra_data as ba_helper

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("data_modules.barra_data").propagate = True
logging.getLogger("report_modules.factor_momentum_report").propagate = True


@notify_on_failure(to="jingbin@fengheasia.com",
                   capture_level=logging.INFO,  # or logging.DEBUG for deeper traces
                   include_logs=True,
                   max_log_chars=60_000)
def factor_data_update():
    ba_api = ba_helper.BarraDataLoader()
    ba_api.daily_update()

factor_data_update()